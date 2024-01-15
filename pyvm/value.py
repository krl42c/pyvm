# pyvm.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from array import array
import os

dbg = os.getenv('DEBUG') == '1'

AVAILABLE_BACKENDS = ('python', 'cpu_c', 'metal')

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
                 data : str | int | float | List[str | int | float],  
                 implicit_cast : bool = False, 
                 backend = 'python', 
                 back = None,
                 metal_device = None):
        self.dtype : DType = dtype
        self.implicit_cast = implicit_cast
        self.backend = backend
        self.back : Backend = back
        self.data = data
        if isinstance(self.data, list): raise NotImplementedError('lists are not supported')
        self.move_to(self.backend, metal_device)

    def move_to(self, new_backend, metal_device = None):
        assert new_backend in AVAILABLE_BACKENDS, "backend is not supported"
        self.backend = new_backend

        if self.backend == 'cpu_c':
            from ctypes import CDLL, c_float, c_int32
            self.c_ops = CDLL('gen/cpu_ops.so')

        elif self.backend == 'metal':
            from ctypes import CDLL, c_float, c_int32
            if not self.back: self.back : MetalBackend = MetalBackend(metal_device)

        if self.dtype == DType.INT: self.data = int(self.data) if self.backend == 'python' else c_int32(self.data)
        elif self.dtype == DType.FLOAT: self.data = float(self.data) if self.backend == 'python' else c_float(self.data)
        elif self.dtype == DType.STRING: self.data = str(self.data)

    @staticmethod
    def _assert_op(lvalue, rvalue):
        # FIXME: fix casting (assign priority)
        if lvalue.implicit_cast: rvalue._cast_to(lvalue.dtype)
        else: assert lvalue.dtype == rvalue.dtype, "value:assert_op: operands are not same dtype" 
    
    def _get_c_op(self, op): 
        assert self.backend == 'cpu_c', f"value::_get_c_op: trying to access c function without using c backend"
        assert self.dtype != DType.STRING, "strings currently not supported for c backend"
        if dbg: print(f'c_cpu_backend: running {op.name} on {self}')
        if op == OPS.ADD: return self.c_ops.i_add if self.dtype == DType.INT else self.c_ops.f_add
        elif op == OPS.SUB: return self.c_ops.i_sub if self.dtype == DType.INT else self.c_ops.f_sub
        elif op == OPS.MUL: return self.c_ops.i_mul if self.dtype == DType.INT else self.c_ops.f_mul
        elif op == OPS.DIV: return self.c_ops.i_div if self.dtype == DType.INT else self.c_ops.f_div

    def _cast_to(self, target : DType):
        if self.dtype == target: return
        if self.dtype == DType.STRING:
            if target == DType.INT or target == DType.FLOAT:
                assert self.data.isdigit(), "value:_cast_to: non numeric only string cannot be casted"
                self.data = int(self.data) if target == DType.INT else float(self.data)
        if self.dtype == DType.INT or self.dtype == DType.FLOAT:
            self.data = str(self.data)
        self.dtype = target

    def __add__(self, right):
        if isinstance(self.data, list):
            # todo: arrays
            return
        Value._assert_op(self, right)
        if self.backend == 'cpu_c': return Value(self.dtype, self._get_c_op(OPS.ADD)(self.data, right.data))
        elif self.backend == 'metal':
            res = self.back.add(self.dtype, self, right)
            if dbg: print(f'running metal kernel on {self.back.device}, result {res}')
            return Value(self.dtype, res, backend='metal', back=self.back)
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

class Backend:
    def __init__(self, backend):
        self.backend = backend

class MetalBackend(Backend):

    def __init__(self, device):
        super().__init__('metal')

        import metalcompute as mc
        from kernels import metal_kernel
        from ctypes import c_int32, c_float

        self.device = device
        assert self.device, "MetalBackend::init: trying to run metal backend without metal device"
        self.kernels = {
            'add_f' : self.device.kernel(metal_kernel).function('add'),
            'add_i' : self.device.kernel(metal_kernel).function('addInt'),
            'mul_f' : self.device.kernel(metal_kernel).function('mul'),
            'mul_i' : self.device.kernel(metal_kernel).function('mulInt'),
            'sub_f' : self.device.kernel(metal_kernel).function('sub'),
            'sub_i' : self.device.kernel(metal_kernel).function('subInt'),
            'div_f' : self.device.kernel(metal_kernel).function('div'),
            'div_i' : self.device.kernel(metal_kernel).function('divInt')
        }

    def add(self, dtype : DType, left : Value, right : Value) -> memoryview:
        kernel = self.kernels['add_i'] if dtype == DType.INT else self.kernels['add_f']
        return self._call_kernel(dtype, kernel, left.data, right.data, self.device.buffer(4))
    def sub(self, dtype : DType, left : Value, right : Value):
        kernel = self.kernels['sub_i'] if dtype == DType.INT else self.kernels['sub_f']
        return self._call_kernel(dtype, kernel, left.data, right.data, self.device.buffer(4))
    def div(self, dtype : DType, left : Value, right : Value):
        kernel = self.kernels['div_i'] if dtype == DType.INT else self.kernels['div_F']
        return self._call_kernel(dtype, kernel, left.data, right.data, self.device.buffer(4))
    def mul(self, dtype : DType, left : Value, right : Value):
        kernel = self.kernels['mul_i'] if dtype == DType.INT else self.kernels['mul_f']
        return self._call_kernel(dtype, kernel, left.data, right.data, self.device.buffer(4))

    def _call_kernel(self, dtype : DType, kernel, a, b, c) -> memoryview: 
        view = memoryview(c).cast('i' if dtype == DType.INT else 'f')
        kernel(1,a,b,c)
        return view[0]
    

import metalcompute as mc 
device = mc.Device()
x = Value(DType.INT, 50, backend='metal', metal_device=device)
y = Value(DType.INT, 50, backend='metal', metal_device=device)

z = x + y
print(z)