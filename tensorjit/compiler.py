"""Fuse a tensor DAG and render runtime-sized C loops.

No fast-math, SIMD schedule search, GPU lowering, or advanced GEMM optimization is
implied. C uses int64 address arithmetic; checked capacities bound every allocation.
"""
from collections import Counter
from dataclasses import dataclass
from tensorjit.graph import Dim, Expr, toposort


@dataclass(frozen=True)
class Kernel:
  node: Expr
  dependencies: tuple[Expr, ...]
  source: str


@dataclass
class Plan:
  inputs: tuple[Expr, ...]
  outputs: tuple[Expr, ...]
  kernels: tuple[Kernel, ...]
  symbols: tuple[Dim, ...]
  names: dict[Expr, str]
  source: str


def compile_plan(outputs, fuse=True):
  outputs = tuple(outputs)
  if not outputs or any(not isinstance(x, Expr) for x in outputs): raise ValueError("nonempty tensor outputs required")
  if len(set(outputs)) != len(outputs): raise ValueError("duplicate outputs are unsupported")
  order = toposort(outputs)
  inputs = tuple(node for node in order if node.op == "input")
  if len({node.arg for node in inputs}) != len(inputs): raise ValueError("input names must be unique")
  symbols = {}
  for node in order:
    for dim in node.shape:
      if isinstance(dim, Dim):
        if dim.name in symbols and symbols[dim.name] != dim: raise ValueError("same symbol name has inconsistent bounds")
        symbols[dim.name] = dim
  symbols = tuple(symbols[name] for name in sorted(symbols))
  dimension_names = {dim: f"dim_{i}" for i, dim in enumerate(symbols)}
  counts = Counter(source for node in order for source in node.sources)
  stages = {node for node in order if node.op not in ("input", "constant") and
            (not fuse or node.op in ("sum", "matmul") or counts[node] > 1 or node in outputs)}
  stages.update(node for node in outputs if node.op in ("input", "constant"))
  names = {node: f"buf_{i}" for i, node in enumerate(order)}
  def dim_code(dim): return dimension_names[dim] if isinstance(dim, Dim) else str(dim)
  def offset(shape, indices):
    if not shape: return "0"
    expression = f"(int64_t)({indices[0]})"
    for dim, index in zip(shape[1:], indices[1:]): expression = f"(({expression})*({dim_code(dim)})+({index}))"
    return expression
  def map_indices(node, indices):
    return ["0" if d == 1 else i for d, i in zip(node.shape, indices[len(indices)-len(node.shape):])]

  kernels = []
  for stage in (node for node in order if node in stages):
    dependencies, lines = set(), []
    def expression(node, indices):
      if node.op == "constant":
        # repr always includes a decimal/exponent for Python float.
        return f"{node.arg!r}f"
      if node.op == "input" or (node in stages and node is not stage):
        dependencies.add(node)
        return f"{names[node]}[{offset(node.shape, indices)}]"
      if node.op in ("add", "mul"):
        args = [expression(source, map_indices(source, indices)) for source in node.sources]
        return f"({args[0]} {'+' if node.op == 'add' else '*'} {args[1]})"
      if node.op == "relu": return f"fmaxf({expression(node.sources[0], indices)}, 0.0f)"
      if node.op == "transpose": return expression(node.sources[0], indices[::-1])
      if node.op == "expand_dims": return expression(node.sources[0], indices[:node.arg]+indices[node.arg+1:])
      if node.op == "broadcast": return expression(node.sources[0], map_indices(node.sources[0], indices))
      if node.op == "positive": return f"({expression(node.sources[0], indices)} > 0.0f ? 1.0f : 0.0f)"
      raise ValueError(f"cannot inline {node.op}; expected a materialization boundary")
    indices = [f"i_{i}" for i in range(len(stage.shape))]
    depth = 1
    for index, dim in zip(indices, stage.shape):
      lines.append("  "*depth + f"for (int32_t {index} = 0; {index} < {dim_code(dim)}; {index}++) {{")
      depth += 1
    indent = "  "*depth
    if stage.op in ("sum", "matmul"):
      if stage.op == "sum":
        source = stage.sources[0]
        source_indices = list(indices)
        source_indices.insert(stage.arg, "reduction")
        extent = source.shape[stage.arg]
        product = expression(source, source_indices)
      else:
        left, right = stage.sources
        extent = left.shape[1]
        product = f"({expression(left, [indices[0], 'reduction'])} * {expression(right, ['reduction', indices[1]])})"
      lines.extend([indent+"float accumulator = 0.0f;", indent+f"for (int32_t reduction = 0; reduction < {dim_code(extent)}; reduction++) {{",
                    indent+f"  accumulator += {product};", indent+"}"])
      value = "accumulator"
    else:
      value = expression(stage, indices)
    destination = f"out_{names[stage]}"
    lines.append(indent + f"{destination}[{offset(stage.shape, indices)}] = {value};")
    for i in reversed(range(len(stage.shape))): lines.append("  "*(i+1) + "}")
    ordered = tuple(node for node in order if node in dependencies)
    arguments = [f"const float *{names[node]}" for node in ordered] + [f"float *{destination}"]
    arguments.extend(f"int32_t {dimension_names[dim]}" for dim in symbols)
    kernel_name = f"kernel_{len(kernels)}"
    source = f"void {kernel_name}({', '.join(arguments)}) {{\n" + "\n".join(lines) + "\n}"
    kernels.append(Kernel(stage, ordered, source))
  source = "#include <stdint.h>\n#include <math.h>\n" + "\n\n".join(k.source for k in kernels)
  return Plan(inputs, outputs, tuple(kernels), symbols, names, source)
