"""Aligned interval reuse for a sequential runtime. Run: python -m labs.memory

Intervals are inclusive: a value read by kernel k cannot share a new output
written by kernel k. The caller must exclude aliases and externally retained buffers.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Temporary:
  name: str
  nbytes: int
  first: int
  last: int
  alignment: int = 64


@dataclass(frozen=True)
class Allocation:
  offset: int
  capacity: int


def align_up(value, alignment): return ((value + alignment - 1) // alignment) * alignment


def plan_memory(temporaries):
  temporaries = list(temporaries)
  if len({t.name for t in temporaries}) != len(temporaries): raise ValueError("duplicate name")
  if any(t.nbytes <= 0 or t.first < 0 or t.last < t.first or t.alignment <= 0 for t in temporaries):
    raise ValueError("invalid lifetime, size, or alignment")
  active, free, allocations, peak = [], [], {}, 0
  for t in sorted(temporaries, key=lambda t: (t.first, t.name)):
    still_active = []
    for last, allocation in active:
      if last < t.first: free.append(allocation)
      else: still_active.append((last, allocation))
    active = still_active
    candidates = [(a.capacity, i, a) for i, a in enumerate(free)
                  if a.capacity >= t.nbytes and a.offset % t.alignment == 0]
    if candidates:
      _, i, allocation = min(candidates)
      free.pop(i)
    else:
      offset = align_up(peak, t.alignment)
      allocation = Allocation(offset, t.nbytes)
      peak = offset + t.nbytes
    allocations[t.name] = allocation
    active.append((t.last, allocation))
  return allocations, peak


def verify_plan(temporaries, allocations):
  temporaries = list(temporaries)
  for i, a in enumerate(temporaries):
    aa = allocations[a.name]
    if aa.offset < 0 or aa.offset % a.alignment or aa.capacity < a.nbytes: raise ValueError("bad allocation")
    for b in temporaries[i+1:]:
      bb = allocations[b.name]
      lifetimes_overlap = max(a.first, b.first) <= min(a.last, b.last)
      storage_overlaps = max(aa.offset, bb.offset) < min(aa.offset+aa.capacity, bb.offset+bb.capacity)
      if lifetimes_overlap and storage_overlaps: raise ValueError(f"live storage overlap: {a.name}, {b.name}")


def main():
  temporaries = [Temporary("a", 1024, 0, 2), Temporary("b", 512, 1, 1), Temporary("c", 768, 3, 4)]
  allocations, peak = plan_memory(temporaries)
  verify_plan(temporaries, allocations)
  for name, allocation in allocations.items(): print(name, allocation)
  print(f"peak={peak} bytes; independent allocations={sum(t.nbytes for t in temporaries)} bytes")


if __name__ == "__main__": main()
