# SkinnyGrad
![python](https://img.shields.io/badge/python-3.11%5E-blue.svg) ![pypi](https://img.shields.io/pypi/v/skinnygrad.svg) ![license](https://img.shields.io/github/license/ArthurBook/skinnygrad) ![tests](https://github.com/ArthurBook/skinnygrad/actions/workflows/tests.yaml/badge.svg)


**SkinnyGrad** is a tensor autodifferentiation library that I wrote as a side project for fun and learning. By default, a computational graph is built and evaluated lazily with [NumPy](https://github.com/numpy/numpy). GPU acceleration is also available with the [CuPy backend extension](./extensions/cupy_engine/). At ~1300 lines, skinnygrad is written with simplicity and extensibility in mind. It nevertheless covers a [good subset](./src/skinnygrad/tensors.py) of the features of a `torch.Tensor`. Kudos to [tinygrad](https://github.com/tinygrad/tinygrad) which inspired the RISC-like design of mapping all operations to [19 low level ops](./src/skinnygrad/llops.py) that the runtime engine optimizes and executes.

# Try it out!
```bash
pip install skinnygrad
```
```python
import skinnygrad

a = tensors.Tensor(((1, 2, 3)))
b = tensors.Tensor(10)
x = tensors.Tensor(((4,), (5,), (6,)))
y = a @ x + b
print(y)
# <skinnygrad.tensors.Tensor(
#   <skinnygrad.llops.Symbol(UNREALIZED <Op(ADD)>, shape=(1, 1))>,
#   self.requires_grad=False,
#   self.gradient=None,
# )>
print(y.realize())
# [[42]]
```

# LeNet-5 as a convergence test
As an end-to-end test for the engine, I replicated the [LeNet-5 paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) -- a convolutional neural network (CNN) designed for handwritten digit recognition. Trained on [MNIST](https://yann.lecun.com/exdb/mnist/), the model recovers 98% accuracy on the evaluation set after about 5 epochs. With a batch size of 64 it takes a few minutes per training epoch (60k images) using the CuPy GPU acceleration backend on a Nvidia A100 GPU. The code for the experiment can be found in the [examples folder](./examples/le_net.py).

## BONUS: The computational graph pass built up by the skinnygrad engine for LeNet-5
![lenet-fwd](./static/lenet-forward.png)
