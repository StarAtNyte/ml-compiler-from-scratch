"""Small dynamic-shape float32 compiler with fusion and a planned native runtime."""
from tensorjit.graph import Dim, Expr, input
from tensorjit.runtime import CompiledGraph
from tensorjit.autodiff import gradients
