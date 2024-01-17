from value import Value, DType
from bytecode import Chunk, Opcode
from typing import List


class VM:
  def __init__(self, buffer : List[Chunk]):
    self.buffer : List[Chunk] = buffer
    self.stack : List[Value] = []
    self.sp = -1
    self.chunk = None 

  def add(self): self.stack.append(self.chunk.lop + self.chunk.rop)
  def sub(self): self.stack.append(self.chunk.lop - self.chunk.rop)
  def mult(self): self.stack.append(self.chunk.lop * self.chunk.rop)
  def div(self): self.stack.append(self.chunk.lop / self.chunk.rop)

  def push(self):
    assert self.chunk.lop, "vm:push: no value to push"
    self.stack.append(self.chunk.lop)

  def run_next(self):
    self.sp += 1
    assert len(self.buffer) > self.sp, "vm:run_next: no more elements in buffer"
    self.chunk = self.buffer[self.sp]
    match self.chunk.op:
      case Opcode.ADD: self.add()
      case Opcode.SUB: self.sub()
      case Opcode.MULT: self.mult()
      case Opcode.DIV: self.div()
      case Opcode.PUSH: self.push()

  def run_buffer(self):
    assert self.buffer, "vm:run_buffer: no buffer available"
    while self.sp <= len(self.buffer) - 2: self.run_next()

buffer : List[Chunk] = []
buffer.append(Chunk(Opcode.PUSH, Value(dtype=DType.INT, data=50)))

import random
import metalcompute as mc

import cProfile
runs = 10000
print(f'running bench on {runs} runs')
device = mc.Device()
for i in range(runs): buffer.append(Chunk(Opcode.ADD, Value(dtype=DType.FLOAT, data=random.randint(0,500), backend="metal", metal_device=device, implicit_cast=True),  
                                         Value(dtype=DType.FLOAT, data=random.randint(0,500), backend='metal', metal_device=device, implicit_cast=True)))

vm = VM(buffer=buffer)
print('')
print('metal:')
cProfile.run('vm.run_buffer()')
buffer = []
for i in range(runs): buffer.append(Chunk(Opcode.ADD, Value(dtype=DType.FLOAT, data=random.randint(0,500), backend="python", implicit_cast=True),  
                                         Value(dtype=DType.FLOAT, data=random.randint(0,500), backend='python', implicit_cast=True)))
vm = VM(buffer=buffer)
print('python:')
#cProfile.run('vm.run_buffer()')

buffer = []
for i in range(runs): buffer.append(Chunk(Opcode.ADD, Value(dtype=DType.FLOAT, data=random.randint(0,500), backend="cpu_c", implicit_cast=True),  
                                         Value(dtype=DType.FLOAT, data=random.randint(0,500), backend='cpu_c', metal_device=device, implicit_cast=True)))
vm = VM(buffer=buffer)
print('c:')
cProfile.run('vm.run_buffer()')
