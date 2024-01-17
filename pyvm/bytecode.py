# bytecode.py

from enum import Enum
from dataclasses import dataclass
from typing import Optional
from value import Value, DType

class Opcode(Enum):
  PUSH = 1
  POP = 2
  ADD = 3
  SUB = 4
  MULT = 5
  DIV = 6

class Chunk:
  def __init__(self, op : Opcode, lop : Optional[Value] = None, rop : Optional[Value] = None):
    self.op = op
    self.lop = lop
    self.rop = rop

  def __bytes__(self):
    bytes_out = bytearray()

    b_op = bytes(self.op.value)
    b_left_op = self.lop.__bytes__() if self.lop else None
    b_right_op = self.rop.__bytes__() if self.rop else None
    bytes_out.extend(b_op)

    if b_left_op: bytes_out.extend(b_left_op)
    if b_right_op: bytes_out.extend(b_right_op)

    return bytes(bytes_out)
  
def serialize(chunk : Chunk) -> bytes:
  pass

def deserialize(buffer : bytes) -> Chunk:
  pass

