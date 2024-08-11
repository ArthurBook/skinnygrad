from typing import Any
from skinnygrad import llops, runtime
import cupy


class CuPyEngine(runtime.SequentialEngine[cupy.ndarray]):
    __OPS_MAP__ = {
        llops.Ops.READ: cupy.array,
        llops.Ops.ASSIGN: lambda tgt, src, i: (cupy.copyto(tgt, src, where=i, casting="unsafe"), tgt)[1],
        llops.Ops.RESHAPE: lambda src, newshape: cupy.reshape(src, newshape.dims),
        llops.Ops.BROADCAST: lambda src, newshape: cupy.broadcast_to(src, newshape.dims),
        llops.Ops.PERMUTE: cupy.transpose,
        llops.Ops.SELECT: lambda arr, loc: arr[*(i if isinstance(i, int) else slice(*i) for i in loc)],
        llops.Ops.PAD: lambda src, pads, pad_val: cupy.pad(src, pad_width=pads, constant_values=(pad_val,)),
        llops.Ops.INV: lambda x: cupy.reciprocal(x.astype(float)),
        llops.Ops.EXP: cupy.exp,
        llops.Ops.LOG: cupy.log,
        llops.Ops.NEG: cupy.negative,
        llops.Ops.ADD: cupy.add,
        llops.Ops.MUL: cupy.multiply,
        llops.Ops.POW: cupy.power,
        llops.Ops.EQ: cupy.equal,
        llops.Ops.LESS: cupy.less,
        llops.Ops.SUM: cupy.sum,
        llops.Ops.AMAX: cupy.amax,
        llops.Ops.WHERE: cupy.where,
    }

    def execute(self, op: llops.Op, *args: cupy.ndarray | Any) -> cupy.ndarray:
        return self.__OPS_MAP__[op](*args)

    def to_python(self, objref: cupy.ndarray) -> runtime.PyArrayRepresentation:
        return objref.tolist()


missing_ops = [op for op in llops.Ops if op not in CuPyEngine.__OPS_MAP__]
assert not missing_ops, f"Missing ops: {missing_ops}"
