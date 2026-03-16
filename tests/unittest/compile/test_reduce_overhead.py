# pylint: disable=W0621, C0103

"""Unit tests for reduce-overhead compilation mode"""

import os
import random
import torch
import numpy as np
import pytest


DEVICE = "musa"
N, D_in, H, D_out = 64, 4096, 2048, 1024
NUM_STEPS = 5


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.musa.manual_seed(seed)
    torch.musa.manual_seed_all(seed)


class SimpleModel(torch.nn.Module):
    """Simple Model"""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(D_in, 4 * H)
        self.fc2 = torch.nn.Linear(4 * H, 4 * H)
        self.fc3 = torch.nn.Linear(4 * H, 4 * H)
        self.relu = torch.nn.ReLU()
        self.classifier = torch.nn.Linear(4 * H, D_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x


class TinyConvBNNet(torch.nn.Module):
    """ " TinyConvBNNet"""

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
        )

        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@pytest.fixture
def data():
    set_seed()
    inputs = [torch.randn(N, D_in, device=DEVICE) for _ in range(NUM_STEPS)]
    targets = [torch.randn(N, D_out, device=DEVICE) for _ in range(NUM_STEPS)]
    return inputs, targets


@pytest.fixture
def models():
    set_seed()
    base = SimpleModel().to(DEVICE)
    reduce = SimpleModel().to(DEVICE)
    reduce.load_state_dict(base.state_dict())
    reduce = torch.compile(reduce, mode="reduce-overhead")
    return base, reduce


def test_reduce_overhead_train_consistency(data, models):
    """Test consistency between base and reduce-overhead models in training mode"""
    inputs, targets = data
    base_model, reduce_model = models

    criterion = torch.nn.MSELoss()
    opt_base = torch.optim.SGD(base_model.parameters(), lr=1e-3)
    opt_reduce = torch.optim.SGD(reduce_model.parameters(), lr=1e-3)

    for step, (x, y) in enumerate(zip(inputs, targets)):
        opt_base.zero_grad()
        out_base = base_model(x)
        loss_base = criterion(out_base, y)
        loss_base.backward()
        opt_base.step()

        opt_reduce.zero_grad()
        out_reduce = reduce_model(x)
        loss_reduce = criterion(out_reduce, y)
        loss_reduce.backward()
        opt_reduce.step()

        # check forward
        torch.testing.assert_close(
            out_base,
            out_reduce,
            atol=1e-6,
            rtol=1e-6,
            msg=f"Output mismatch at train step {step}",
        )

        # check loss
        torch.testing.assert_close(
            loss_base,
            loss_reduce,
            atol=1e-6,
            rtol=1e-6,
            msg=f"Loss mismatch at train step {step}",
        )


def test_reduce_overhead_infer_consistency(data, models):
    """Test consistency between base and reduce-overhead models in inference mode"""
    inputs, _ = data
    base_model, reduce_model = models

    base_model.eval()
    reduce_model.eval()

    with torch.no_grad():
        for step, x in enumerate(inputs):
            out_base = base_model(x)
            out_reduce = reduce_model(x)

            torch.testing.assert_close(
                out_base,
                out_reduce,
                atol=1e-6,
                rtol=1e-6,
                msg=f"Output mismatch at infer step {step}",
            )


@pytest.fixture
def resnet_data():
    """Generate data for TinyConvBNNet tests"""
    set_seed()
    inputs = [torch.randn(4, 3, 224, 224, device=DEVICE) for _ in range(NUM_STEPS)]
    targets = [torch.randint(0, 1000, (4,), device=DEVICE) for _ in range(NUM_STEPS)]
    return inputs, targets


@pytest.fixture
def resnet_models():
    set_seed()
    base = TinyConvBNNet().to(DEVICE)
    reduce = TinyConvBNNet().to(DEVICE)
    reduce.load_state_dict(base.state_dict())
    reduce = torch.compile(reduce, mode="reduce-overhead")
    return base, reduce


def test_reduce_overhead_TinyConvBNNet_train_consistency(resnet_data, resnet_models):
    """Test consistency between base and reduce-overhead TinyConvBNNet in training mode"""
    inputs, targets = resnet_data
    base_model, reduce_model = resnet_models

    criterion = torch.nn.CrossEntropyLoss()
    opt_base = torch.optim.SGD(base_model.parameters(), lr=1e-3)
    opt_reduce = torch.optim.SGD(reduce_model.parameters(), lr=1e-3)

    for step, (x, y) in enumerate(zip(inputs, targets)):
        opt_base.zero_grad()
        out_base = base_model(x)
        loss_base = criterion(out_base, y)
        loss_base.backward()
        opt_base.step()

        opt_reduce.zero_grad()
        out_reduce = reduce_model(x)
        loss_reduce = criterion(out_reduce, y)
        loss_reduce.backward()
        opt_reduce.step()

        # loss
        torch.testing.assert_close(
            loss_base,
            loss_reduce,
            atol=1e-3,
            rtol=1e-3,
            msg=f"TinyConvBNNet loss mismatch at train step {step}",
        )


def test_reduce_overhead_TinyConvBNNet_infer_consistency(resnet_data, resnet_models):
    """Test consistency between base and reduce-overhead TinyConvBNNet in inference mode"""
    inputs, _ = resnet_data
    base_model, reduce_model = resnet_models

    base_model.eval()
    reduce_model.eval()

    with torch.no_grad():
        for step, x in enumerate(inputs):
            out_base = base_model(x)
            out_reduce = reduce_model(x)

            torch.testing.assert_close(
                out_base,
                out_reduce,
                atol=1e-6,
                rtol=1e-6,
                msg=f"TinyConvBNNet output mismatch at infer step {step}",
            )
