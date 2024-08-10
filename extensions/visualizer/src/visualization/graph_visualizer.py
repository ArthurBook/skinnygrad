import os
import pathlib
from typing import Any

import pygraphviz as pgv
from visualization import formatting

from skinnygrad import callbacks, llops

FORMATS_DIR = pathlib.Path(__file__).parent / "formats"


class GraphVisualizer(callbacks.OnSymbolInitCallBack, callbacks.OnCtxExitCallBack):
    graph_format = formatting.ElementFormatter(
        (None, formatting.FormatSpec.from_path(FORMATS_DIR / "graph.json")),
    )
    opnode_formatter = formatting.ElementFormatter(
        (None, formatting.FormatSpec.from_path(FORMATS_DIR / "base.json")),
        (None, formatting.FormatSpec.from_path(FORMATS_DIR / "opnode.json")),
    )
    datanode_formatter = formatting.ElementFormatter(
        (None, formatting.FormatSpec.from_path(FORMATS_DIR / "base.json")),
        (None, formatting.FormatSpec.from_path(FORMATS_DIR / "datanode.json")),
    )
    edge_formatter = formatting.ElementFormatter(
        (None, formatting.FormatSpec.from_path(FORMATS_DIR / "base.json")),
        (None, formatting.FormatSpec.from_path(FORMATS_DIR / "edge.json")),
    )

    def __init__(self, output_path: str | os.PathLike) -> None:
        self.output_path = output_path
        self._graph = pgv.AGraph(**self.graph_format[None])
        self._subgraph: pgv.AGraph | None = None

    def write_dot(self, path: str | os.PathLike) -> None:
        self._graph.draw(path, prog="dot", format="png")

    def on_symbol_creation(self, symbol: llops.Symbol) -> None:
        match symbol.op:
            case llops.Ops.ASSIGN:
                tgt, _ = symbol.symbol_args.values()
                self._add_opnode(symbol)
                self._add_data_to_op_edges(symbol, symbol)
                self._add_op_to_data_edge(symbol, tgt)
            case _:
                self._add_opnode(symbol)
                self._add_datanode(symbol)
                self._add_op_to_data_edge(symbol, symbol)
                self._add_data_to_op_edges(symbol, symbol)

    def _add_opnode(self, op: llops.Symbol) -> None:
        op_fmt = self.opnode_formatter.get_fmt(type(op.op), op.op)
        op_label = self._format_opname(op)
        self.cur_graph.add_node(_get_op_nodename(op), label=op_label, **op_fmt)

    def _format_opname(self, op: llops.Symbol) -> str:
        match op.op:
            case llops.Ops.BROADCAST | llops.Ops.RESHAPE:
                return f"{op.op.name}(shape={op.non_symbol_args['shape'].dims})"  # type: ignore
            case llops.Ops.PERMUTE:
                return f"{op.op.name}(order={op.non_symbol_args['order']})"
            case llops.Ops.SUM | llops.Ops.AMAX:  # reduce
                return f"{op.op.name}(axes={op.non_symbol_args['axes']})"
            case _:
                return f"{op.op.name}"

    def _add_datanode(self, op: llops.Symbol) -> None:
        data_nodename, data_fmt = _get_data_nodename(op), self.datanode_formatter.get_fmt()
        self.cur_graph.add_node(data_nodename, label=str(op.shape.dims), **data_fmt)  # data

    def _add_op_to_data_edge(self, src_op: llops.Symbol, tgt_data: llops.Symbol) -> None:
        op_nodename, data_nodename = _get_op_nodename(src_op), _get_data_nodename(tgt_data)
        self.cur_graph.add_edge(op_nodename, data_nodename, **self.edge_formatter.get_fmt())

    def _add_data_to_op_edges(self, src_data: llops.Symbol, tgt_op: llops.Symbol) -> None:
        op_nodename, edge_fmt = _get_op_nodename(tgt_op), self.edge_formatter.get_fmt()
        for in_edge in src_data.symbol_args.values():
            self.cur_graph.add_edge(_get_data_nodename(in_edge), op_nodename, **edge_fmt)

    def on_ctx_exit(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.write_dot(self.output_path)

    @property
    def cur_graph(self) -> pgv.AGraph:
        return self._subgraph if self._subgraph is not None else self._graph


def _get_op_nodename(symbol: llops.Symbol) -> str:
    return f"{id(symbol)}"


def _get_data_nodename(symbol: llops.Symbol) -> str:
    return f"{id(symbol)}_data"
