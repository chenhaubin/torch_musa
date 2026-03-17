"""Test Inductor Triton Conv2d"""

import os
import pytest
import torch
from torch import nn
from torch._inductor import config as inductor_config

from torch_musa.testing.base_test_tool import _HAS_TRITON


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class SimpleConv2dModule(nn.Module):
    """
    standard convolution module with batch normalization and activation.
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn2d = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn2d(self.conv(x)))


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not installed")
class TestInductorTritonConv2d:
    """Test class of InductorTritonConv2d"""

    def test_inductor_triton_conv2d(self):
        """
        Test method of InductorTritonConv2d, this test ensures the inductor
        fused conv2d model accuracy matches golden.
        """
        device = "musa"
        dtype = torch.float16
        model = SimpleConv2dModule(3, 64, 3)
        # typical input: batch of 1, 640x640 RGB image
        x = torch.randn(1, 3, 640, 640)

        # Tune on triton_musa MMA:
        os.environ["ENABLE_MUSA_MMA"] = "1"

        with torch.no_grad():
            golden = model(x)

            model.to(dtype).to(device)
            x = x.to(dtype).to(device)
            y = model(x).cpu().float()
            torch.musa.synchronize()

            torch.testing.assert_close(golden, y, atol=1e-2, rtol=1e-3)  # atol=1e-3

            with inductor_config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_pointwise": True,
                    "max_autotune_gemm": True,
                    # use Triton mm|conv kernel only:
                    "max_autotune_gemm_backends": "TRITON",
                    "max_autotune_conv_backends": "TRITON",
                    "max_autotune_gemm_search_space": "DEFAULT",
                }
            ):
                compiled_model = torch.compile(
                    model,
                    backend="inductor",
                    mode="max-autotune",
                    fullgraph=True,
                    dynamic=False,
                )

                y = compiled_model(x).cpu().float()
                torch.musa.synchronize()

                torch.testing.assert_close(golden, y, atol=1e-2, rtol=1e-3)  # atol=1e-3
