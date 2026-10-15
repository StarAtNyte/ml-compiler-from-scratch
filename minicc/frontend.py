"""Lower a deliberately restricted Python AST to mutable-variable control-flow IR.

Python supplies lexing/parsing only. minicc defines its own i32 and buffer semantics.
Supported: assignments, integer expressions/comparisons, if, while, break, continue,
return, buffer indexing/stores. Unsupported Python syntax is rejected, never executed.
"""
import ast
from minicc.ir import Block, Function, Instruction, Kind, Parameter, Terminator, i32, prune


class Frontend:
  def __init__(self, source):
    module = ast.parse(source)
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
      raise ValueError("source must contain exactly one function")
    self.node = node = module.body[0]
    args = node.args
    if node.decorator_list or args.posonlyargs or args.kwonlyargs or args.defaults or args.kw_defaults or args.vararg or args.kwarg:
      raise ValueError("decorators, defaults, and special argument forms are unsupported")
    if node.returns is not None and (not isinstance(node.returns, ast.Name) or node.returns.id != "int"):
      raise ValueError("return annotation must be int")
    parameters = []
    for arg in args.args:
      if arg.annotation is None: kind = Kind.I32
      elif isinstance(arg.annotation, ast.Name) and arg.annotation.id in ("int", "buffer"):
        kind = Kind.BUFFER if arg.annotation.id == "buffer" else Kind.I32
      else: raise ValueError("parameter annotations must be int or buffer")
      parameters.append(Parameter(arg.arg, kind))
    if len({p.name for p in parameters}) != len(parameters): raise ValueError("duplicate parameter")
    self.function = Function(node.name, tuple(parameters), {"entry": Block()})
    self.buffers = {p.name for p in parameters if p.kind == Kind.BUFFER}
    self.current, self.counter, self.temp_counter = "entry", 0, 0
    self.loops = []

  @property
  def block(self): return self.function.blocks[self.current]

  def new_block(self, hint):
    self.counter += 1
    name = f"{hint}_{self.counter}"
    self.function.blocks[name] = Block()
    return name

  def emit(self, op, *args):
    self.temp_counter += 1
    name = f"$t{self.temp_counter}"
    self.block.instructions.append(Instruction(op, name, args))
    return name

  def scalar_name(self, node):
    if not isinstance(node, ast.Name) or node.id in self.buffers: raise ValueError("expected scalar variable")
    return node.id

  def address(self, node):
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name) or node.value.id not in self.buffers:
      raise ValueError("indexing requires a buffer parameter")
    return node.value.id, self.expression(node.slice)

  def expression(self, node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, bool): return i32(node.value)
    if isinstance(node, ast.Name): return self.scalar_name(node)
    if isinstance(node, ast.Subscript): return self.emit("load", *self.address(node))
    if isinstance(node, ast.UnaryOp):
      if isinstance(node.op, ast.USub): return self.emit("sub", 0, self.expression(node.operand))
      if isinstance(node.op, ast.Not): return self.emit("eq", self.expression(node.operand), 0)
    if isinstance(node, ast.BinOp):
      operations = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}
      if type(node.op) in operations:
        return self.emit(operations[type(node.op)], self.expression(node.left), self.expression(node.right))
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
      operations = {ast.Lt: "lt", ast.LtE: "le", ast.Eq: "eq", ast.NotEq: "ne", ast.Gt: "lt", ast.GtE: "le"}
      op = type(node.ops[0])
      if op in operations:
        left, right = self.expression(node.left), self.expression(node.comparators[0])
        if op in (ast.Gt, ast.GtE): left, right = right, left
        return self.emit(operations[op], left, right)
    raise ValueError(f"unsupported expression: {ast.dump(node)}")

  def statements(self, statements):
    for node in statements:
      if self.block.terminator is not None: break
      self.statement(node)

  def statement(self, node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
      value = self.expression(node.value)
      target = node.targets[0]
      if isinstance(target, ast.Name):
        self.block.instructions.append(Instruction("copy", self.scalar_name(target), (value,)))
      elif isinstance(target, ast.Subscript):
        self.block.instructions.append(Instruction("store", None, (*self.address(target), value)))
      else: raise ValueError("unsupported assignment target")
    elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
      name = self.scalar_name(node.target)
      operations = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}
      if type(node.op) not in operations: raise ValueError("unsupported augmented operation")
      self.block.instructions.append(Instruction(operations[type(node.op)], name, (name, self.expression(node.value))))
    elif isinstance(node, ast.Return) and node.value is not None:
      self.block.terminator = Terminator("return", (self.expression(node.value),))
    elif isinstance(node, ast.If):
      condition = self.expression(node.test)
      yes, no, merge = self.new_block("then"), self.new_block("else"), self.new_block("merge")
      self.block.terminator = Terminator("branch", (condition,), (yes, no))
      for name, statements in ((yes, node.body), (no, node.orelse)):
        self.current = name
        self.statements(statements)
        if self.block.terminator is None: self.block.terminator = Terminator("jump", targets=(merge,))
      self.current = merge
    elif isinstance(node, ast.While) and not node.orelse:
      head, body, end = self.new_block("while"), self.new_block("body"), self.new_block("end")
      self.block.terminator = Terminator("jump", targets=(head,))
      self.current = head
      condition = self.expression(node.test)
      self.block.terminator = Terminator("branch", (condition,), (body, end))
      self.loops.append((head, end))
      self.current = body
      self.statements(node.body)
      if self.block.terminator is None: self.block.terminator = Terminator("jump", targets=(head,))
      self.loops.pop()
      self.current = end
    elif isinstance(node, (ast.Break, ast.Continue)):
      if not self.loops: raise ValueError("break/continue outside a loop")
      self.block.terminator = Terminator("jump", targets=(self.loops[-1][1 if isinstance(node, ast.Break) else 0],))
    elif isinstance(node, ast.Pass): pass
    else: raise ValueError(f"unsupported statement: {ast.dump(node)}")

  def lower(self):
    self.statements(self.node.body)
    # Unreachable merge blocks may remain after both arms return. Give all blocks
    # a structural terminator, then prune. Reachable fallthrough is rejected.
    for block in self.function.blocks.values():
      if block.terminator is None: block.terminator = Terminator("missing_return")
    prune(self.function)
    if any(b.terminator.op == "missing_return" for b in self.function.blocks.values()):
      raise ValueError("function may fall through without returning")
    return self.function


def lower(source): return Frontend(source).lower()
