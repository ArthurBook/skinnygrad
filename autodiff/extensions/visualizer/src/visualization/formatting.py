"""
- Modify the __post_init__ to and add a callback that the config can set
    - The graphviz hook will look something like:
        1) add op node with id(symbol); for each src in symbol.srcs: add in edges
        2) add f'{id(src)}_data' node with shape symbol.shape
        
- Graph context that describes what subgraph (if any) we are in
"""

from __future__ import annotations

import collections
import functools
import json
import os
import pathlib
from typing import Any, Hashable, Self


class ElementFormatter(collections.defaultdict[Hashable, "FormatSpec"]):
    def __init__(self, *format_specs: tuple[Hashable, FormatSpec]) -> None:
        super().__init__(FormatSpec)
        for key, spec in format_specs:
            self[key].update(spec)

    def get_fmt(self, *key: Hashable) -> FormatSpec:
        key = key if isinstance(key, tuple) else (key,)
        base = self.get(None, FormatSpec())
        return functools.reduce(lambda a, b: a | self.get(b, FormatSpec()), key, base)


class FormatSpec(collections.UserDict[str, Any]):
    @classmethod
    def from_path(cls, path: str | os.PathLike) -> Self:
        return cls(**json.loads(pathlib.Path(path).read_text()))
