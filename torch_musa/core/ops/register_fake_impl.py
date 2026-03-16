"""For operators that lack of FakeTensor implementation,
use torch.library.register_fake to complete registration
"""

from typing import Tuple
import torch


# pylint: disable-all
@torch.library.register_fake("aten::_scaled_dot_product_attention_flash_musa")
def scaled_dot_product_attention_flash_musa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Register FakeTensor impl for aten::_scaled_dot_product_attention_flash_musa"""

    output_shape = query.shape[:-1] + (value.shape[-1],)
    dropout_mask_shape = (0,)

    if query.dim() == 3:
        num_heads, query_seq_len, _ = query.shape
        logsumexp_shape = (num_heads, query_seq_len)
        key_seq_len = key.shape[1]

        if dropout_p > 0.0:
            dropout_mask_shape = (num_heads, query_seq_len, key_seq_len)

    else:
        batch_size, num_heads, query_seq_len, _ = query.shape
        logsumexp_shape = (batch_size, num_heads, query_seq_len)
        key_seq_len = key.shape[2]

        if dropout_p > 0.0:
            dropout_mask_shape = (batch_size, num_heads, query_seq_len, key_seq_len)

    output = torch.empty(output_shape, dtype=query.dtype, device="meta")
    logsumexp = torch.empty(logsumexp_shape, dtype=torch.float32, device="meta")
    dropout_mask = torch.empty(dropout_mask_shape, dtype=torch.bool, device="meta")

    return output, logsumexp, dropout_mask


@torch.library.register_fake("aten::_scaled_dot_product_attention_flash_musa_backward")
def scaled_dot_product_attention_flash_musa_backward(
    grad_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    logsumexp: torch.Tensor,
    dropout_mask: torch.Tensor,
    is_causal: bool = False,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Register FakeTensor impl for aten::_scaled_dot_product_attention_flash_musa_backward"""

    grad_query_shape = query.shape
    grad_key_shape = key.shape
    grad_value_shape = value.shape
    grad_attn_mask_shape = (0,)
    grad_attn_mask_dtype = torch.float32

    if attn_mask is not None:
        grad_attn_mask_shape = attn_mask.shape
        grad_attn_mask_dtype = attn_mask.dtype

    grad_query = torch.empty(grad_query_shape, dtype=query.dtype, device="meta")
    grad_key = torch.empty(grad_key_shape, dtype=key.dtype, device="meta")
    grad_value = torch.empty(grad_value_shape, dtype=value.dtype, device="meta")
    grad_attn_mask = torch.empty(
        grad_attn_mask_shape, dtype=grad_attn_mask_dtype, device="meta"
    )

    return grad_query, grad_key, grad_value, grad_attn_mask
