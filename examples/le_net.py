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

import argparse
import csv
import dataclasses
import functools
import itertools
import logging
import pathlib
import random
from typing import Iterator

import numpy as np

import skinnygrad
import skinnygrad.optimizers
from skinnygrad import autograd, layers, tensors

DATADIR = pathlib.Path("data/mnist")
MNIST_IN_CHANNELS = 1
MNIST_RESOLUTION = (28, 28)
MNIST_INTENSITY_RANGE = 0, 255
MNIST_ENCODING = "utf-8"

np.random.seed(42)
random.seed(42)
logging.basicConfig(
    level=logging.INFO,
    filename="logs/lenet-mnist-train.log",
    format="%(asctime)s | %(message)s",
)
logger = logging.getLogger(__name__)


### Model ###
class LeNet:
    def __init__(self) -> None:
        self.c1 = layers.ConvLayer(
            in_channels=MNIST_IN_CHANNELS,
            out_channels=6,
            kernel_shape=(5, 5),
            padding=(2, 2),
            bias=True,
            activation=autograd.tanh,
        )
        self.s2 = layers.PoolingLayer(
            kernel_shape=(2, 2),
            stride=2,
            method="max",
        )
        self.c3 = layers.ConvLayer(
            in_channels=self.c1.output_channels,
            out_channels=16,
            kernel_shape=(5, 5),
            padding=False,
            bias=True,
            activation=autograd.tanh,
        )
        self.s4 = layers.PoolingLayer(
            kernel_shape=(2, 2),
            stride=2,
            method="max",
        )
        self.c5 = layers.ConvLayer(
            in_channels=self.c3.output_channels,
            out_channels=120,
            kernel_shape=(5, 5),
            padding=False,
            bias=True,
            activation=autograd.tanh,
        )
        self.f6 = layers.FFLayer(
            input_size=self.c5.output_channels,
            output_size=84,
            bias=True,
            activation=autograd.tanh,
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

        random.shuffle(data := list(zip(labels, pixels)))
        labels, pixels = zip(*data)
        return Mnist(skinnygrad.Tensor(labels), skinnygrad.Tensor(pixels))


class DataLoader:
    def __init__(self, mnist: Mnist, batch_size: int) -> None:
        """NOTE: assumes already shuffled data"""
        self.epoch: float = 0
        self.step: int = -1
        self.dataset_size = mnist.pixels.shape.dims[0]
        self.batch_size = batch_size
        self.mnist = mnist

    def __iter__(self) -> Iterator[tuple[skinnygrad.Tensor, skinnygrad.Tensor]]:
        curr_epoch = int(self.epoch)
        while int(self.epoch) == curr_epoch:
            yield next(self)

    def __next__(self) -> tuple[skinnygrad.Tensor, skinnygrad.Tensor]:
        self.step += 1
        self.epoch = self.step / (self.dataset_size / self.batch_size)
        start = (self.step * self.batch_size) % self.dataset_size
        end = min(self.dataset_size, start + self.batch_size)
        loc = ((start, end),)
        return self.mnist.pixels[loc], self.mnist.labels[loc]


def main():
    parser = argparse.ArgumentParser(description="Train a LeNet model on the MNIST dataset.")
    parser.add_argument("--train_path", type=pathlib.Path, default=DATADIR / "mnist_train.csv", help="train file.")
    parser.add_argument("--test_path", type=pathlib.Path, default=DATADIR / "mnist_test.csv", help="test file.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train the model for.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for training.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=256, help="Batch size for evaluation.")
    parser.add_argument("--eval_every_n_steps", type=int, default=50, help="Evaluate every n steps.")
    args = parser.parse_args()
    train(
        train_path=args.train_path,
        test_path=args.test_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_every_n_steps=args.eval_every_n_steps,
    )


def train(
    train_path: pathlib.Path,
    test_path: pathlib.Path,
    epochs: int,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    eval_every_n_steps: int,
) -> None:
    lenet = LeNet()
    optimizer = skinnygrad.optimizers.SGD(lenet.params(), lr=learning_rate)
    train_data = DataLoader(load_mnist_images(train_path), batch_size=train_batch_size)
    test_data = DataLoader(load_mnist_images(test_path), batch_size=eval_batch_size)
    run_eval_flag = itertools.cycle(itertools.chain([True], [False] * (eval_every_n_steps - 1)))
    eval_loss: float = 1e9
    while train_data.epoch <= epochs:
        train_loss = train_step(train_data, optimizer, lenet)
        eval_loss = eval_model(test_data, lenet) if next(run_eval_flag) else eval_loss
        epoch, step, train_loss_val = train_data.epoch, train_data.step, train_loss.realize()
        log_step(epoch, step, train_loss_val, eval_loss)  # type: ignore


def train_step(dataloader: DataLoader, optimizer: skinnygrad.optimizers.Optimizer, lenet: LeNet) -> tensors.Tensor:
    batch_pixels, batch_labels = next(dataloader)
    with optimizer:
        batch_pred = lenet(batch_pixels)
        loss = autograd.binary_cross_entropy(batch_labels, batch_pred).mean()
        loss.backprop()
    return loss


def eval_model(dataloader: DataLoader, lenet: LeNet) -> float:
    batch_losses: list[float] = []
    for batch, batch_labels in dataloader:
        corrects = np.argmax(batch_labels.realize(), axis=1) == np.argmax(lenet(batch).realize(), axis=1)
        batch_losses.append(np.sum(corrects))
    return sum(batch_losses) / dataloader.dataset_size


def log_step(epoch: float, step: int, train_bce: float, eval_acc: float) -> None:
    logger.info(
        " | ".join(
            (
                f"epoch: {epoch:5.4f}",
                f"step: {step:<9}",
                f"train_bce_loss: {train_bce:.9f}",
                f"eval_accuracy: {eval_acc:.9f}",
            )
        )
    )


if __name__ == "__main__":
    main()
