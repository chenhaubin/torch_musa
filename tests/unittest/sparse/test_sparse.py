"""Test sparse operators."""

# pylint: disable=missing-function-docstring, unused-import

from numbers import Number
import pytest
import torch
from torch.testing import make_tensor
from torch_musa import testing

ATOL, RTOL = 1e-5, 1e-5


class TestSparse:
    """
    Test suit of torch sparse operators.
    """

    def _gen_sparse(self, sparse_dim, nnz, with_size, dtype, device, coalesced):
        if isinstance(with_size, Number):
            with_size = [with_size] * sparse_dim

        x, i, v = self.gen_sparse_tensor(
            with_size, sparse_dim, nnz, not coalesced, dtype=dtype, device=device
        )

        if not coalesced:
            self.assert_uncoalesced(x)

        return x, i, v

    def assert_equal(self, golden, result, equal_nan=False):
        if isinstance(golden, int) and isinstance(result, int):
            assert golden == result
            return
        if isinstance(golden, list) and isinstance(result, list):
            assert len(golden) == len(result)
            for idx, g in enumerate(golden):
                self.assert_equal(g, result[idx])
            return
        rtol, atol = ATOL, RTOL
        matches = torch.isclose(
            golden, result, rtol=rtol, atol=atol, equal_nan=equal_nan
        )
        assert torch.all(matches)

    def assert_uncoalesced(self, x):
        """
        Test if a tensor is uncoalesced.  This is used to ensure
        correctness of the uncoalesced tensor generation algorithm.
        """
        assert not x.is_coalesced()
        existing_indices = set()
        indices = x._indices()
        for i in range(x._nnz()):
            index = str(indices[:, i])
            if index in existing_indices:
                return
            existing_indices.add(index)

    def gen_sparse_tensor(self, size, sparse_dim, nnz, is_uncoalesced, device, dtype):
        # Assert not given impossible combination, where the sparse dims have
        # empty numel, but nnz > 0 makes the indices containing values.
        if not (all(size[d] > 0 for d in range(sparse_dim)) or nnz == 0):
            raise AssertionError(
                f"invalid arguments: size={size}, sparse_dim={sparse_dim}, nnz={nnz}"
            )

        v_size = [nnz] + list(size[sparse_dim:])
        v = make_tensor(v_size, device=device, dtype=dtype, low=-1, high=1)
        i = torch.rand(sparse_dim, nnz, device=device)
        i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
        i = i.to(torch.long)
        if is_uncoalesced:
            i_1 = i[:, : (nnz // 2), ...]
            i_2 = i[:, : ((nnz + 1) // 2), ...]
            i = torch.cat([i_1, i_2], 1)
        x = torch.sparse_coo_tensor(i, v, torch.Size(size), dtype=dtype, device=device)

        if not is_uncoalesced:
            x = x.coalesce()
        else:
            x = x.detach().clone()._coalesced_(False)
        return x, x._indices().clone(), x._values().clone()

    @testing.test_on_nonzero_card_if_multiple_musa_device(1)
    @pytest.mark.parametrize("dtype", testing.get_float_types())
    @pytest.mark.parametrize("coalesced", [True])
    def test_basic(self, dtype, coalesced):
        device = torch.musa.current_device()

        def test_shape(sparse_dims, nnz, with_size):
            if isinstance(with_size, Number):
                with_size = [with_size] * sparse_dims
            x, i, v = self._gen_sparse(
                sparse_dims, nnz, with_size, dtype, device, coalesced
            )
            self.assert_equal(i, x._indices())
            self.assert_equal(v, x._values())
            self.assert_equal(x.ndimension(), len(with_size))
            self.assert_equal(
                x.coalesce()._nnz(), nnz if x.is_coalesced() else nnz // 2
            )
            self.assert_equal(list(x.size()), with_size)

            # Test .indices() and .values()
            if not coalesced:
                with self.assertRaisesRegex(
                    RuntimeError, "Cannot get indices on an uncoalesced tensor"
                ):
                    x.indices()
                with self.assertRaisesRegex(
                    RuntimeError, "Cannot get values on an uncoalesced tensor"
                ):
                    x.values()
            else:
                self.assert_equal(x.indices(), x._indices())
                self.assert_equal(x.values(), x._values())

        test_shape(3, 10, 100)
        test_shape(3, 10, [100, 100, 100])
        test_shape(3, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(3, 0, [0, 0, 100, 5, 5, 5, 0])

        # Make sure that coalesce handles duplicate indices correctly
        i = torch.tensor(
            [[9, 0, 0, 0, 8, 1, 1, 1, 2, 7, 2, 2, 3, 4, 6, 9]], device=device
        )
        v = torch.tensor(
            [[idx**2, idx] for idx in range(i.size(1))], dtype=dtype, device=device
        )
        x = torch.sparse_coo_tensor(
            i, v, torch.Size([10, 2]), dtype=dtype, device=device
        )
        self.assert_equal(x.coalesce()._nnz(), 9)
