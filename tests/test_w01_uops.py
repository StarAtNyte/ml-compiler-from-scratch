"""Week 1: UOps, hash consing, rewriting, symbolic, evaluation by rewriting."""
import math, pytest
from uopc.uop import UOp, Ops, ParamArg, CallArg, UOpError, range_
from uopc.dtype import dtypes
from uopc.rewrite import UPat, PatternMatcher, graph_rewrite
from uopc.symbolic import run, simplify, to_python, vmin, vmax, exec_alu
from uopc.uir import parse, to_uir

def stack(*xs): return UOp(Ops.STACK, tuple(x if isinstance(x, UOp) else UOp.const(x) for x in xs))
def param(slot, dtype=dtypes.i32, shape=()): return UOp(Ops.PARAM, (), ParamArg(slot, dtype, shape))

def test_hash_consing():
  a, b = UOp.const(1) + UOp.const(2), UOp.const(1) + UOp.const(2)
  assert a is b
  assert UOp.const(1) is not UOp.const(1.0) is not UOp.const(True)
  assert UOp.const(0.0) is not UOp.const(-0.0)

def test_shape_dtype_inference():
  s = stack(stack(1, 2, 3), stack(4, 5, 6))
  assert s.shape == (2, 3) and s.dtype == dtypes.i32
  assert s.index(UOp.const(1)).shape == (3,)
  assert (UOp.const(1) < UOp.const(2)).dtype == dtypes.bool
  with pytest.raises(UOpError): UOp.const(1) + UOp.const(1.0)
  with pytest.raises(UOpError): stack(1, 2) + stack(1, 2, 3)
  with pytest.raises(UOpError): stack(1, 2.0)

def test_const_fold_semantics():
  assert exec_alu(Ops.FLOORDIV, dtypes.i32, [-7, 2]) == -4          # floor, not truncation
  assert exec_alu(Ops.FLOORMOD, dtypes.i32, [-7, 2]) == 1
  assert exec_alu(Ops.ADD, dtypes.i32, [2**31 - 1, 1]) == -2**31     # wraps like int32
  assert exec_alu(Ops.CAST, dtypes.i32, [-2.7]) == -2                # C truncation
  assert exec_alu(Ops.ADD, dtypes.f32, [0.1, 0.2]) == pytest.approx(0.3, abs=1e-7)
  assert exec_alu(Ops.MUL, dtypes.f32, [16777216.0, 1.0]) == 16777216.0
  assert exec_alu(Ops.ADD, dtypes.f32, [16777216.0, 1.0]) == 16777216.0   # f32 rounding

def test_run_stack_program():
  a, b = stack(1, 2, 3), stack(10, 20, 30)
  assert to_python(run(a * b + a)) == [11, 42, 93]
  assert to_python(run(stack(a, b).index(UOp.const(1)).index(UOp.const(2)))) == 30
  w = (a < stack(2, 2, 2)).where(a, b)
  assert to_python(run(w)) == [1, 20, 30]

def test_call_inlining_and_scoping():
  p0 = param(0)
  sq = UOp(Ops.CALL, (p0 * p0, UOp.const(7)), CallArg("sq"))
  assert run(sq).arg == 49
  # nested: f(x) = sq(x) + x ; the inner PARAM 0 is a different scope from the outer one
  inner = UOp(Ops.CALL, (p0 * p0, p0 + 1), CallArg("sq"))
  outer = UOp(Ops.CALL, (inner + p0, UOp.const(3)), CallArg("f"))
  assert run(outer).arg == 16 + 3

def test_call_type_checks():
  with pytest.raises(UOpError): UOp(Ops.CALL, (param(0) + param(1), UOp.const(1)))

def test_symbolic_identities():
  x = param(0)
  assert simplify(x + 0) is x and simplify(x * 1) is x and simplify(x * 0).arg == 0
  assert simplify((x + 3) + 4) is simplify(x + 7)
  assert simplify((x * 4) // 4) is x
  assert simplify((x * 6 + 4) % 2).arg == 0

def test_symbolic_ranges():
  i, j = range_(4, 0), range_(8, 1)
  flat = i * 8 + j
  assert simplify(flat // 8) is i and simplify(flat % 8) is j
  assert simplify((flat // 8) * 8 + flat % 8) is simplify(flat)
  assert simplify(j // 8).arg == 0 and simplify(j % 8) is j
  assert simplify(i < 4).arg is True and simplify(i < 0).arg is False
  assert (vmin(flat), vmax(flat)) == (0, 31)

def test_custom_pattern_matcher():
  pm = PatternMatcher([(UPat(Ops.MUL, src=[UPat.var("x"), UPat.cvar("c", dtype=dtypes.i32)]),
                        lambda x, c: x + x if c.arg == 2 else None)])
  x = param(0)
  assert graph_rewrite(x * 2, pm) is x + x

def test_deep_graph_no_recursion_error():
  x = UOp.const(0)
  for i in range(20000): x = x + 1
  assert run(x).arg == 20000

def test_uir_roundtrip_and_syllabus_example():
  src = """
  %0 = buffer 65536 : dtype=f32 slot=0
  %1 = buffer 65536 : dtype=f32 slot=1
  %2 = reshape %0, (256, 256, 1)
  %3 = reshape %1, (256, 1, 256)
  %4 = mul %2, %3
  %5 = reduce %4 : op=add pop=1
  """
  g = parse(src)
  assert g.shape == (256, 256)
  assert parse(to_uir(g)) is g

def test_uir_evaluates():
  g = parse("""
  %0 = stack 1.0, 2.0, 3.0
  %1 = stack %0, %0
  %2 = mul %1, 2
  %3 = permute %2, (1, 0)
  %4 = reduce %3 : op=add pop=1
  """)
  assert to_python(run(g)) == [4.0, 8.0, 12.0]

def test_folding_matches_c_semantics():
  # f32 inputs are rounded like the C compiler rounds the literal 16777217.0f
  assert simplify(UOp.const(16777217.0) + UOp.const(2.0)).arg == 16777218.0
  # constants made by reassociation wrap like int32 arithmetic: 65536*65536 == 0 (mod 2^32)
  assert simplify((param(0) * 65536) * 65536).arg == 0
  # INT_MIN // -1 traps in C: leave it to run time
  assert exec_alu(Ops.FLOORDIV, dtypes.i32, [-2**31, -1]) is None

def test_inline_call_does_not_enter_nested_bodies():            # NOTES 2
  from uopc.symbolic import inline_call
  p0 = param(0)
  inner = UOp(Ops.CALL, (p0 * p0, p0 + 1), CallArg("sq"))
  outer = UOp(Ops.CALL, (inner + p0, UOp.const(3)), CallArg("f"))
  body = inline_call(outer)                  # inline ONLY the outer call; inner must survive intact
  assert body.src[0].op is Ops.CALL and body.src[0].src[0] is p0 * p0
  assert run(body).arg == 19

def test_cast_inf_to_int_is_not_folded():                         # NOTES 6
  assert exec_alu(Ops.CAST, dtypes.i32, [math.inf]) is None
  assert exec_alu(Ops.CAST, dtypes.i32, [math.nan]) is None

def test_float_constants_do_not_reassociate():                    # NOTES 7
  x = UOp(Ops.PARAM, (), ParamArg(0, dtypes.f32))
  assert simplify((x + 1.0) + 2.0) is not simplify(x + 3.0)
