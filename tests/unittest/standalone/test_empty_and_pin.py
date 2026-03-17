"""test empty && pin"""

import torch


def test_empty_and_pin():
    """Standalone for device lazy init"""
    x = torch.empty([4, 4])
    y = x.pin_memory("musa")
    assert y.is_pinned()
