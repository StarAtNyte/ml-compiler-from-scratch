from minicc import compile_ir, compile_native, execute
from minicc.ir import format_ir
from minicc.machine import destroy_phis, allocate, execute_allocated

SOURCE = """
def sum_to(n: int) -> int:
    i = 0
    total = 0
    while i < n:
        if i < 3:
            total = total + i
        else:
            total = total + 2 * i
        i += 1
    return total
"""


def main():
  function = compile_ir(SOURCE)
  print(format_ir(function))
  allocation = allocate(destroy_phis(function), registers=2)
  result, counts = execute_allocated(allocation, 10)
  native = compile_native(SOURCE)
  assert result == execute(function, 10) == native(10) == 87
  print(f"result={result}, registers=2, spill slots={allocation.slots}, counters={counts}")
  print("SSA interpreter, allocated machine, and native C agree")


if __name__ == "__main__": main()
