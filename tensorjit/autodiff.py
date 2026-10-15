"""Reverse-mode differentiation of the pure dynamic tensor graph.

Differentiation precedes fusion and code generation. The shape expressions in
backward retain the original symbols; no concrete runtime extent is captured.
"""
from tensorjit.graph import Expr, constant, toposort


def unbroadcast(gradient, shape):
  while len(gradient.shape) > len(shape): gradient = gradient.sum(0)
  for axis, extent in enumerate(shape):
    if extent == 1 and gradient.shape[axis] != 1:
      gradient = gradient.sum(axis).expand_dims(axis)
  if gradient.shape != shape: raise ValueError("gradient shape does not match source")
  return gradient


def gradients(loss, wrt):
  if not isinstance(loss, Expr) or loss.shape != (): raise ValueError("loss must be a scalar expression")
  wrt = tuple(wrt)
  if any(not isinstance(node, Expr) or node.op != "input" for node in wrt): raise ValueError("differentiate with respect to input nodes")
  order = toposort((loss,))
  accumulated = {loss: constant(1)}
  for node in reversed(order):
    if node not in accumulated: continue
    gradient, sources = accumulated[node], node.sources
    if node.op in ("input", "constant"): continue
    if node.op == "add": contributions = (unbroadcast(gradient, sources[0].shape), unbroadcast(gradient, sources[1].shape))
    elif node.op == "mul":
      contributions = (unbroadcast(gradient*sources[1], sources[0].shape), unbroadcast(gradient*sources[0], sources[1].shape))
    elif node.op == "relu": contributions = (gradient*Expr("positive", sources[0].shape, sources),)
    elif node.op == "transpose": contributions = (gradient.transpose(),)
    elif node.op == "sum": contributions = (gradient.expand_dims(node.arg).broadcast_to(sources[0].shape),)
    elif node.op == "expand_dims": contributions = (gradient.sum(node.arg),)
    elif node.op == "broadcast": contributions = (unbroadcast(gradient, sources[0].shape),)
    elif node.op == "matmul": contributions = (gradient @ sources[1].transpose(), sources[0].transpose() @ gradient)
    elif node.op == "positive":
      raise ValueError("higher derivatives through the ReLU mask are unsupported")
    else: raise ValueError(f"missing derivative rule: {node.op}")
    for source, contribution in zip(sources, contributions):
      if contribution.shape != source.shape: raise ValueError("derivative rule produced wrong shape")
      accumulated[source] = accumulated[source]+contribution if source in accumulated else contribution
  return tuple(accumulated[node] if node in accumulated else constant(0).broadcast_to(node.shape) for node in wrt)
