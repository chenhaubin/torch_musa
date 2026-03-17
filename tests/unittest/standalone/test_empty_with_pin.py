"""test empty(pin=True)"""

import torch


def test_empty_and_pin():
    """Standalone for device lazy init"""
    x = torch.empty([4, 4], pin_memory=True)
    assert x.is_pinned()
