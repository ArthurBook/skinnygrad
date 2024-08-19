# SkinnyGrad
![pypi](https://img.shields.io/pypi/v/configmate.svg) 

**SkinnyGrad** is a side project implementing a tensor autodifferentiation library with a torch-like interface from scratch in Python.
By default, a computational graph is built and evaluated lazily with [NumPy](https://github.com/numpy/numpy), and GPU acceleration can be switched on with the [CuPy backend extension](./extensions/cupy_engine/).

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

# LeNet-5 demo
To test the engine, I've replicated the [LeNet-5 paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) and trained it for 5 epochs, achieving same the 98% accuracy on the MNIST eval set [here](./examples/le_net.py). The forward pass built up by the skinnygrad engine is shown below:
![lenet-fwd](./static/lenet_forward.png)

# Acknowledgements
The design choices are inspired by [tinygrad](https://github.com/tinygrad/tinygrad).
