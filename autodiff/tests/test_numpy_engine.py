""" 
test the numpy backend
"""

from autodiff import config, graph, llops
from autodiff.engines.packaged_engines import numpy_engine


def test_numpy_engine_add_reduce() -> None:
    with config.Configuration(backend=numpy_engine.NUMPY_ENGINE):
        x = graph.Symbol.load([1, 2, 3])
        y = graph.Symbol.load([1, 2, 3])
        z = x.do(llops.BinaryOps.MUL, y)  # 1, 4, 9
        z_sum = z.do(llops.ReduceOps.SUM, axis=0)  # 14
        res = z_sum.realize().to_python()
    assert res == 14


def test_numpy_engine_assignment() -> None:
    with config.Configuration(backend=numpy_engine.NUMPY_ENGINE):
        x = graph.Symbol.load([1, 2, 3])
        x = x.do(llops.MemOps.ASSIGN, graph.Symbol.load([4, 5, 6]))
        x_val = x.realize().to_python()
    assert x_val == [4, 5, 6]


def test_numpy_engine_assign_full_loop() -> None:
    with config.Configuration(backend=numpy_engine.NUMPY_ENGINE):
        x = graph.Symbol.load([1, 2, 3])
        x = x.do(llops.MemOps.ASSIGN, x.do(llops.BinaryOps.ADD, graph.Symbol.load([1, 1, 1])))
        x = x.do(llops.MemOps.ASSIGN, x.do(llops.BinaryOps.ADD, graph.Symbol.load([1, 1, 1])))
        x_val = x.realize().to_python()
    assert x_val == [3, 4, 5]


def test_numpy_engine_garbagecollector() -> None:
    ## TODO
    ...


if __name__ == "__main__":
    test_numpy_engine_add_reduce()
    test_numpy_engine_garbagecollector()
    test_numpy_engine_assign_full_loop()
