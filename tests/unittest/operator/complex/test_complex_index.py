""" Tests for complex index operations. """

# pylint: disable=W0621

import copy
import torch
import pytest
from torch_musa import testing


torch.manual_seed(41)

index_select_configs = [
    [(10,), 0, 5],
    [(10, 5), 1, 3],
    [(10, 5, 3), 2, 2],
    [(10, 5, 1, 3), 1, 1],
    [(10, 5, 1, 3, 5), 4, 3],
    [(10, 5, 1, 3, 1, 2, 7), 3, 2],

    [(4, 128), 0, 2],
    [(4, 128), 0, 8],
    [(4, 128, 256,), 1, 64],

    [(4, 128, 256), 1, 0],
]
dtypes = [torch.complex64, torch.complex128]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", index_select_configs)
@pytest.mark.parametrize("dtype", dtypes)
def test_index_select(config, dtype):
    """ Test for torch.index_select operator. """
    input_data = {}
    indexed_dim = config[1]
    indexed_dim_size = config[0][indexed_dim]
    input_data["input"] = torch.empty(config[0]).uniform_(-10, 10).to(dtype)
    input_data["dim"] = indexed_dim
    input_data["index"] = torch.randint(0, indexed_dim_size, (config[2],))

    test = testing.OpTest(func=torch.index_select, input_args=input_data)
    test.check_result()
    test.check_grad_fn()


def get_indices(inputs):
    """ Generate indices and values for index_put test. """
    indices = []
    values = []
    for input_self in inputs:
        indice = []  # tuple of LongTensor, and its length must match the input's dim
        # the number of items must not exceed input items
        num_item = min(10, input_self.numel())
        shape = input_self.shape
        for dim in tuple(range(input_self.dim())):
            max_idx = shape[dim]
            if max_idx == 0:
                continue
            indice.append(torch.randint(max_idx, (num_item,)))

        indices.append(tuple(indice))
        if num_item == 0:
            values.append(torch.rand(1, 1))
        else:
            values.append(torch.randn(num_item))
    return [indices, values]


input_data = testing.get_raw_data() + [
    torch.rand(10, 10, 2, 2, 1) > 0.5,
]

[indices, values] = get_indices(input_data)

input_datas = []
for i, data in enumerate(input_data):
    input_datas.append({"input": data, "indices": indices[i], "values": values[i]})

ind_dtypes = [torch.int64]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("ind_dtype", ind_dtypes)
def test_index_put(input_data, dtype, ind_dtype):
    """ Test for torch.index_put operator. """
    if testing.get_musa_arch() < 22 and dtype == torch.bfloat16:
        return

    input_data["input"] = input_data["input"].to(dtype)
    input_data["indices"] = [x.to(ind_dtype) for x in input_data["indices"]]
    input_data["values"] = input_data["values"].to(dtype)
    input_data["accumulate"] = True
    test = testing.OpTest(func=torch.index_put, input_args=input_data)
    test.check_result()
    test.check_grad_fn()
    inplace_input = copy.deepcopy(input_data)
    test = testing.InplaceOpChek(
        func_name=torch.index_put.__name__ + "_",
        self_tensor=inplace_input["input"],
        input_args={
            "indices": inplace_input["indices"],
            "values": inplace_input["values"],
            "accumulate": True,
        },
    )
    test.check_address()
    test.check_res()
