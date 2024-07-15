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
        self.graph = pgv.AGraph(**self.graph_format[None])

    def write_dot(self, path: str | os.PathLike) -> None:
        self.graph.draw(path, prog="dot", format="png")

    def on_symbol_creation(self, symbol: llops.Symbol) -> None:
        op_fmt = self.opnode_formatter.get_fmt(type(symbol.op), symbol.op)
        op_nodename = _get_op_nodename(symbol)
        self.graph.add_node(op_nodename, label=symbol.op.name, **op_fmt)  # node
        data_fmt = self.datanode_formatter.get_fmt()
        data_nodename = _get_data_nodename(symbol)
        self.graph.add_node(data_nodename, label=str(symbol.shape.dims), **data_fmt)  # data
        edge_fmt = self.edge_formatter.get_fmt()
        self.graph.add_edge(op_nodename, data_nodename, **edge_fmt)
        for in_edge in symbol.symbol_args.values():
            self.graph.add_edge(_get_data_nodename(in_edge), op_nodename, **edge_fmt)

    def on_ctx_exit(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.write_dot(self.output_path)


def _get_op_nodename(symbol: llops.Symbol) -> str:
    return f"{id(symbol)}"


def _get_data_nodename(symbol: llops.Symbol) -> str:
    return f"{id(symbol)}_data"
