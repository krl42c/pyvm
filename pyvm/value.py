# pyvm.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from array import array
import os

dbg = os.getenv('DEBUG') == '1'

class DType(Enum):
    INT = 0
    FLOAT = 1
    STRING = 2

class OPS(Enum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3

class Value:
    def __init__(self, dtype : DType, 
                 data : str | int | float, 
                 implicit_cast : bool = False, 
                 backend = 'python', 
                 metal_device = None):
        self.dtype : DType = dtype
        self.implicit_cast = implicit_cast
        self.backend = backend

        if self.backend == 'cpu_c':
            from ctypes import CDLL, c_float, c_int32
            self.c_ops = CDLL('gen/cpu_ops.so')
        elif self.backend == 'metal':
            import metalcompute as mc
            from kernels import metal_kernel
            from ctypes import c_int32, c_float
            assert metal_device, "value:init: trying to run on metal backend without metal device"
            self.device = metal_device
            self.add_kernel = self.device.kernel(metal_kernel).function("add")
            self.add_kernel_int = self.device.kernel(metal_kernel).function("addInt")

        if self.dtype == DType.INT: self.data = int(data) if self.backend == 'python' else c_int32(data)
        elif self.dtype == DType.FLOAT: self.data = float(data) if self.backend == 'python' else c_float(data)
        elif self.dtype == DType.STRING: self.data = str(data)

    @staticmethod
    def _assert_op(lvalue, rvalue):
        assert lvalue.dtype == rvalue.dtype, "value:assert_op: operands are not same dtype" if not lvalue.implicit_cast else rvalue._cast_to(lvalue.dtype)
    
    def _get_c_op(self, op): 
        assert self.backend == 'cpu_c', f"value::_get_c_op: trying to access c function without using c backend"
        assert self.dtype != DType.STRING, "strings currently not supported for c backend"
        if dbg: print(f'c_cpu_backend: running {op.name} on {self}')
        if op == OPS.ADD: return self.c_ops.i_add if self.dtype == DType.INT else self.c_ops.f_add
        elif op == OPS.SUB: return self.c_ops.i_sub if self.dtype == DType.INT else self.c_ops.f_sub
        elif op == OPS.MUL: return self.c_ops.i_mul if self.dtype == DType.INT else self.c_ops.f_mul
        elif op == OPS.DIV: return self.c_ops.i_div if self.dtype == DType.INT else self.c_ops.f_div

    def _cast_to(self, target : DType):
        assert target != self.dtype
        if self.dtype == DType.STRING:
            if target == DType.INT or target == DType.FLOAT:
                assert self.data.isdigit(), "value:_cast_to: non numeric only string cannot be casted"
                self.data = int(self.data) if target == DType.INT else float(self.data)
        if self.dtype == DType.INT or self.dtype == DType.FLOAT:
            self.data = str(self.data)
        self.dtype = target

    def __add__(self, right):
        Value._assert_op(self, right)
        if self.backend == 'cpu_c': return Value(self.dtype, self._get_c_op(OPS.ADD)(self.data, right.data))
        elif self.backend == 'metal':
            input_a = self.data
            input_b = right.data
            input_c = self.device.buffer(4)
            c_view = memoryview(input_c).cast('i' if self.dtype == DType.INT else 'f')
            self.add_kernel_int(1, input_a, input_b, input_c) if self.dtype == DType.INT else self.add_kernel(1, input_a, input_b, input_c)
            return Value(self.dtype, c_view[0], metal_device=self.device, backend='metal')
        out = Value(self.dtype, self.data + right.data)
        return out
    
    def __sub__(self, right):
        assert self.dtype != DType.STRING and right.dtype != DType.STRING, "value:__sub__: op not supported for string dtype"
        Value._assert_op(self, right)
        if self.backend == 'cpu_c': return Value(self.dtype, self._get_c_op(OPS.SUB)(self.data, right.data))
        out = Value(self.dtype, self.data - right.data)
        return out
 
    def __mul__(self, right):
        assert self.dtype != DType.STRING and right.dtype != DType.STRING, "value:__mul__: op not supported for string dtype"
        Value._assert_op(self, right)
        if self.backend == 'cpu_c': return Value(self.dtype, self._get_c_op(OPS.MUL)(self.data, right.data))
        out = Value(self.dtype, self.data * right.data)
        return out

    def __truediv__(self, right):
        assert self.dtype != DType.STRING and right.dtype != DType.STRING, "value:__truediv__: op not supported for string dtype"
        Value._assert_op(self, right)
        if self.backend == 'cpu_c': return Value(self.dtype, self._get_c_op(OPS.DIV)(self.data, right.data))
        out = Value(self.dtype, self.data / right.data)

    def __bytes__(self):
        bytes_out = bytearray()
        btype = bytes(self.dtype.value)
        bvalue = bytes(self.data) if self.dtype in (DType.FLOAT, DType.INT) else bytes(self.data, "utf-8")

        bytes_out.extend(btype) 
        bytes_out.extend(bvalue)

        return bytes(bytes_out)

    def __repr__(self):
        return repr(f'Value ({self.dtype} : {self.data})')

import metalcompute as mc 
device = mc.Device()
x = Value(DType.INT, 50, backend='metal', metal_device=device)
y = Value(DType.INT, 50, backend='metal', metal_device=device)

z = x + y
print(z)