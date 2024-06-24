"""
These are the low level ops that the backend must support.
Tribute to [Chief Keef - Opps](https://www.youtube.com/watch?v=0XbrR1veyyI)
"""

import enum


@enum.global_enum
class MemOps(enum.Enum):
    """special ops for memory manipulation"""

    NOOP = enum.auto()  # do nothing. this should be used for realized vertices
    INIT = enum.auto()  # create the tensor from an init instruction
    TO_PYTHON = enum.auto()  # convert to pure python standard list of list repr
    ASSIGN = enum.auto()  # elementwise assign from one tensor to another (shapes must match)


@enum.global_enum
class UnaryOps(enum.Enum):
    """elementwise apply f(a:M)->b:M"""

    NEG = enum.auto()  # turn the value negative


@enum.global_enum
class BinaryOps(enum.Enum):
    """elementwise apply f(a:A,b:A)->c:A"""

    ADD = enum.auto()  # addition a+b
    MUL = enum.auto()  # multiplication a*b


@enum.global_enum
class TernaryOps(enum.Enum):
    """elementwise apply f(a:A,b:A,c:A)->d:A"""

    WHERE = enum.auto()  # where a, take b, else c


@enum.global_enum
class ReduceOps(enum.Enum):
    """reduce a along axis=int f(a:A)->b:B"""

    SUM = enum.auto()  # sum along axis
    MAX = enum.auto()  # max along axis
