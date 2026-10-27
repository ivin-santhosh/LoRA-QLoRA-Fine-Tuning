"""Small, deterministic tensor graph IR, interpreter and a guarded fusion pass.

The implementation is intentionally a teaching-scale CPU reference runtime, not a
compiler for arbitrary PyTorch models. No network I/O, code generation or eval.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Mapping

MAX_DIM = 512
MAX_ELEMENTS = 262144
OPS = {"matmul", "add_bias", "relu", "linear_relu"}


@dataclass(frozen=True)
class Node:
    name: str
    op: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class Graph:
    nodes: tuple[Node, ...]
    output: str

    def validate(self) -> None:
        known: set[str] = set()
        outputs: set[str] = set()
        for node in self.nodes:
            if not node.name.isidentifier():
                raise ValueError(f"invalid node name: {node.name!r}")
            if node.name in known:
                raise ValueError(f"duplicate node name: {node.name}")
            if node.op not in OPS:
                raise ValueError(f"unsupported operation: {node.op}")
            expected = 3 if node.op == "linear_relu" else 2 if node.op in ("matmul", "add_bias") else 1
            if len(node.inputs) != expected:
                raise ValueError(f"{node.op}: expected {expected} inputs")
            if node.name in node.inputs:
                raise ValueError(f"self-referential node: {node.name}")
            for dependency in node.inputs:
                if dependency in outputs and dependency not in known:
                    raise ValueError(f"unavailable dependency: {dependency}")
            known.add(node.name)
            outputs.add(node.name)
        if self.output not in known:
            raise ValueError("graph output must name an operation node")
        # References not defined as a node are external feeds, so a missing
        # reference is reported during execution, not graph construction.
        for node in self.nodes:
            for dependency in node.inputs:
                if dependency in known and all(n.name != dependency for n in self.nodes[: self.nodes.index(node)]):
                    raise ValueError(f"forward reference/cycle: {dependency}")


def _matrix(value: object, label: str) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or not value or len(value) > MAX_DIM:
        raise ValueError(f"{label} must be a nonempty, bounded 2D matrix")
    width = None
    rows: list[list[float]] = []
    for original in value:
        if not isinstance(original, (list, tuple)) or not original or len(original) > MAX_DIM:
            raise ValueError(f"{label}: invalid matrix row")
        if width is None:
            width = len(original)
        if width != len(original):
            raise ValueError(f"{label}: ragged matrix")
        if len(value) * len(original) > MAX_ELEMENTS:
            raise ValueError(f"{label}: element budget exceeded")
        row = [float(x) for x in original]
        if not all(math.isfinite(x) for x in row):
            raise ValueError(f"{label}: non-finite input")
        rows.append(row)
    return rows


def _bias(value: object, cols: int) -> list[float]:
    if not isinstance(value, (tuple, list)) or len(value) != cols:
        raise ValueError(f"bias must have {cols} elements")
    result = [float(x) for x in value]
    if not all(math.isfinite(x) for x in result):
        raise ValueError("bias contains non-finite entries")
    return result


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    if len(a[0]) != len(b):
        raise ValueError(f"matmul shape mismatch ({len(a)}, {len(a[0])}) x ({len(b)}, {len(b[0])})")
    # Transposition gives sequential reads of the second matrix.
    bt = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def add_bias(a: list[list[float]], bias: list[float]) -> list[list[float]]:
    return [[x + b for x, b in zip(row, bias)] for row in a]


def relu(a: list[list[float]]) -> list[list[float]]:
    return [[max(0.0, x) for x in row] for row in a]


def fused_linear_relu(a: list[list[float]], b: list[list[float]], bias: list[float]) -> list[list[float]]:
    if len(a[0]) != len(b):
        raise ValueError("fused matmul shape mismatch")
    bt = list(zip(*b))
    return [[max(0.0, sum(x * y for x, y in zip(row, col)) + bias[j])
             for j, col in enumerate(bt)] for row in a]


def execute(graph: Graph, feeds: Mapping[str, object]) -> tuple[list[list[float]], dict[str, int]]:
    graph.validate()
    names = {node.name for node in graph.nodes}
    if names.intersection(feeds):
        raise ValueError("external feeds cannot shadow operation names")
    cache: dict[str, object] = {}
    allocated_elements = 0
    for node in graph.nodes:
        inputs: list[object] = []
        for name in node.inputs:
            if name in cache:
                inputs.append(cache[name])
            elif name in feeds:
                inputs.append(feeds[name])
            else:
                raise ValueError(f"missing feed {name!r} required by {node.name}")
        if node.op == "matmul":
            result = matmul(_matrix(inputs[0], node.name), _matrix(inputs[1], node.name))
        elif node.op == "add_bias":
            inp = _matrix(inputs[0], node.name)
            result = add_bias(inp, _bias(inputs[1], len(inp[0])))
        elif node.op == "relu":
            result = relu(_matrix(inputs[0], node.name))
        elif node.op == "linear_relu":
            lhs, rhs = _matrix(inputs[0], node.name), _matrix(inputs[1], node.name)
            result = fused_linear_relu(lhs, rhs, _bias(inputs[2], len(rhs[0])))
        else:
            raise AssertionError(node.op)
        if not all(math.isfinite(x) for row in result for x in row):
            raise OverflowError("non-finite result")
        allocated_elements += sum(len(row) for row in result)
        cache[node.name] = result
    return cache[graph.output], {"operation_nodes": len(graph.nodes), "intermediate_output_elements": allocated_elements}


def fuse_matmul_bias_relu(graph: Graph) -> Graph:
    """Fuse only when both intermediate nodes have a single consumer.

    No aliasing, graph-reordering, or cross-output transformations are attempted.
    """
    graph.validate()
    lookup = {node.name: node for node in graph.nodes}
    consumers: Counter[str] = Counter(dep for node in graph.nodes for dep in node.inputs)
    # The output is observable even if no node consumes it.
    consumers[graph.output] += 1
    remove: set[str] = set()
    replacements: dict[str, Node] = {}
    for node in graph.nodes:
        if node.op != "relu":
            continue
        add = lookup.get(node.inputs[0])
        if not add or add.op != "add_bias" or consumers[add.name] != 1:
            continue
        mul = lookup.get(add.inputs[0])
        if not mul or mul.op != "matmul" or consumers[mul.name] != 1:
            continue
        # Enforce the ordered pattern and single-use intermediates.
        remove.update((add.name, mul.name))
        replacements[node.name] = Node(node.name, "linear_relu", (*mul.inputs, add.inputs[1]))
    optimized = Graph(tuple(replacements.get(n.name, n) for n in graph.nodes if n.name not in remove), graph.output)
    optimized.validate()
    return optimized


def demo_graph() -> Graph:
    return Graph((Node("mm", "matmul", ("x", "w")),
                  Node("bias", "add_bias", ("mm", "b")),
                  Node("activation", "relu", ("bias",))), "activation")
