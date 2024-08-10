"""
Example, fitting le-net to the MNIST dataset
a la 

before running the file, do:
```bash
make data-mnist
```

to download the MNIST dataset.
"""

from __future__ import annotations

import csv
import dataclasses
import functools
import logging
import pathlib
from typing import Iterator

import skinnygrad
import skinnygrad.optimizers
from skinnygrad import autograd, layers

MNIST_ROOT = pathlib.Path("data/mnist")
MNIST_RESOLUTION = (28, 28)
MNIST_IN_CHANNELS = 1
MNIST_INTENSITY_RANGE = 0, 255
MNIST_ENCODING = "utf-8"

logging.basicConfig(level=logging.INFO, format="%(message)s", filename="logs/lenet-mnist-train.log")
logger = logging.getLogger(__name__)


def train(
    train_path: pathlib.Path = MNIST_ROOT / "mnist_train.csv",
    test_path: pathlib.Path = MNIST_ROOT / "mnist_test.csv",
    batch_size: int = 32,
    learning_rate: float = 0.01,
    epochs: int = 10,
) -> None:
    train_data = DataLoader(load_mnist_images(train_path), batch_size=batch_size)
    lenet = LeNet()
    optimizer = skinnygrad.optimizers.SGD(lenet.params(), lr=learning_rate)
    while train_data.epoch <= epochs:
        pixels, labels = next(train_data)
        with optimizer:
            pred = lenet(pixels)
            loss = autograd.binary_cross_entropy(labels, pred).mean()
            loss.backprop()
        logger.info(f"epoch: {train_data.epoch:5.4f} | step: {train_data.step:<9} | loss: {loss.realize():.9f}")


### Model ###
class LeNet:
    def __init__(self) -> None:
        self.c1 = layers.ConvLayer(
            in_channels=MNIST_IN_CHANNELS,
            out_channels=9,
            kernel_shape=(5, 5),
            padding=(2, 2),
            bias=True,
            activation=autograd.relu,
        )
        self.s2 = layers.PoolingLayer(
            kernel_shape=(2, 2),
            stride=2,
            method="mean",
        )
        self.c3 = layers.ConvLayer(
            in_channels=self.c1.output_channels,
            out_channels=16,
            kernel_shape=(5, 5),
            padding=False,
            bias=True,
            activation=autograd.relu,
        )
        self.s4 = layers.PoolingLayer(
            kernel_shape=(2, 2),
            stride=2,
            method="mean",
        )
        self.c5 = layers.ConvLayer(
            in_channels=self.c3.output_channels,
            out_channels=120,
            kernel_shape=(5, 5),
            padding=False,
            bias=True,
            activation=autograd.relu,
        )
        self.f6 = layers.FFLayer(
            input_size=self.c5.output_channels,
            output_size=84,
            bias=True,
            activation=autograd.relu,
        )
        self.f7 = layers.FFLayer(
            input_size=self.f6.output_size,
            output_size=10,
            bias=True,
            activation=functools.partial(autograd.softmax, axes=1),
        )

    def __call__(self, image_batch: skinnygrad.Tensor) -> skinnygrad.Tensor:
        out = image_batch
        for op in self.ops():
            out = op(out)
        return out

    def params(self) -> Iterator[skinnygrad.Tensor]:
        return (p for op in self.ops() if isinstance(op, layers.HasParameters) for p in op.params())

    def ops(self) -> tuple[layers.SupportsForward, ...]:
        return (self.c1, self.s2, self.c3, self.s4, self.c5, layers.flatten_except_batch_dim, self.f6, self.f7)


### Data ###
@dataclasses.dataclass
class Mnist:
    labels: skinnygrad.Tensor  # (n, 10)
    pixels: skinnygrad.Tensor  # (n, 1, 28, 28)


def load_mnist_images(path: pathlib.Path) -> Mnist:
    with path.open(mode="r", encoding=MNIST_ENCODING) as file:
        reader = csv.reader(file)
        next(reader)  # throw away headers
        labels, pixels = list[list[int]](), list[list[list[list[float]]]]()
        for label, *pixel_values in reader:
            labels.append([0] * 10)
            labels[-1][int(label)] = 1
            pixels.append(
                [
                    [
                        [int(p) / MNIST_INTENSITY_RANGE[1] for p in pixel_values[i : i + MNIST_RESOLUTION[0]]]
                        for i in range(0, len(pixel_values), MNIST_RESOLUTION[0])
                    ]
                ]
            )
        return Mnist(skinnygrad.Tensor(labels), skinnygrad.Tensor(pixels))


class DataLoader:
    def __init__(self, mnist: Mnist, batch_size: int) -> None:
        """NOTE: assumes already shuffled data"""
        self.epoch: float = 0
        self.step: int = -1
        self.batch_size = batch_size
        self.mnist = mnist

    def __next__(self) -> tuple[skinnygrad.Tensor, skinnygrad.Tensor]:
        self.step += 1
        self.epoch = self.step / (self.mnist.pixels.shape.dims[0] / self.batch_size)
        loc = ((start := self.step * self.batch_size, start + self.batch_size),)
        return self.mnist.pixels[loc], self.mnist.labels[loc]


if __name__ == "__main__":
    train()
