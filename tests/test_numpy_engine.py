""" 
test the numpy backend
"""

from skinnygrad import config, llops, runtime


def test_numpy_engine_add_reduce() -> None:
    with config.Configuration(engine=runtime.NumPyEngine()):
        x = llops.ControlOps.LOAD([1, 2, 3])
        y = llops.ControlOps.LOAD([1, 2, 3])
        z = llops.BinaryOps.MUL(x, y)  # 1, 4, 9
        z_sum = llops.ReduceOps.SUM(z, axes=(0,))  # 14
        res = z_sum.realize().to_python()
    assert res == 14


def test_numpy_engine_assignment() -> None:
    with config.Configuration(engine=runtime.NumPyEngine()):
        x = llops.ControlOps.LOAD([1, 2, 3])
        x = llops.ControlOps.ASSIGN(x, llops.ControlOps.LOAD([4, 5, 6]))
        x_val = x.realize().to_python()
    assert x_val == [4, 5, 6]


def test_numpy_engine_assign_full_loop() -> None:
    with config.Configuration(engine=runtime.NumPyEngine()):
        x = llops.ControlOps.LOAD([1, 2, 3])
        x = llops.ControlOps.ASSIGN(x, llops.BinaryOps.ADD(x, llops.ControlOps.LOAD([1, 1, 1])))
        x = llops.ControlOps.ASSIGN(x, llops.BinaryOps.ADD(x, llops.ControlOps.LOAD([1, 1, 1])))
        x_val = x.realize().to_python()
    assert x_val == [3, 4, 5]


if __name__ == "__main__":
    test_numpy_engine_add_reduce()
    test_numpy_engine_assign_full_loop()
