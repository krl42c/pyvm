# pyvm.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from array import array
from ctypes import c_float, c_int32
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
            from ctypes import CDLL 
            if not self.back: self.back : CBackend = CBackend()
        elif self.backend == 'metal':
            from ctypes import CDLL
            if not self.back: self.back : MetalBackend = MetalBackend(metal_device)

        if self.dtype == DType.INT: self.data = int(self.data) if self.backend == 'python' else c_int32(self.data)
        elif self.dtype == DType.FLOAT: self.data = float(self.data) if self.backend == 'python' else c_float(self.data)
        elif self.dtype == DType.STRING: self.data = str(self.data)

    @staticmethod
    def _assert_op(lvalue, rvalue):
        if lvalue.implicit_cast and rvalue.implicit_cast: 
            to_cast, no_cast = (lvalue, rvalue) if lvalue.dtype.value > rvalue.dtype.value else (rvalue, lvalue)
            to_cast._cast_to(no_cast.dtype)
        if dbg: print("lvalue.type", lvalue.dtype)
        if dbg: print("rvalue.type", rvalue.dtype)
        else: assert lvalue.dtype == rvalue.dtype, "value:assert_op: operands are not same dtype" 
    
    def _cast_to(self, target : DType):
        if self.dtype == target: return
        if target == DType.FLOAT:
            if self.dtype == DType.INT:
                self.data = float(self.data) if self.back == 'python' else c_float(self.data)
            elif self.dtype == DType.STRING:
                assert self.data.isdigit(), "non casteable string"
                self.data = float(self.data) if self.back == 'python' else c_float(self.data)
        if target == DType.INT:
            if self.dtype == DType.FLOAT:
                self.data = int(self.data) if self.back == 'python' else c_int32(int(self.data.value))
            elif self.dtype == DType.STRING:
                assert self.data.isdigit(), "non casteable string"
                self.data = int(self.data) if self.back == 'python' else c_int32(int(self.data.value))

    def __add__(self, right):
        Value._assert_op(self, right)
        if self.back: return Value(self.dtype, self.back.add(self.dtype, self, right), back=self.back, backend=self.backend) 
        return Value(self.dtype, self.data + right.data)
    
    def __sub__(self, right):
        assert self.dtype != DType.STRING and right.dtype != DType.STRING, "value:__sub__: op not supported for string dtype"
        Value._assert_op(self, right)
        if self.back: return Value(self.dtype, self.back.sub(self.dtype, self, right), back=self.back, backend=self.backend) 
        return Value(self.dtype, self.data - right.data)
 
    def __mul__(self, right):
        assert self.dtype != DType.STRING and right.dtype != DType.STRING, "value:__mul__: op not supported for string dtype"
        Value._assert_op(self, right)
        if self.back: return Value(self.dtype, self.back.mul(self.dtype, self, right), back=self.back, backend=self.backend) 
        return Value(self.dtype, self.data * right.data)

    def __truediv__(self, right):
        assert self.dtype != DType.STRING and right.dtype != DType.STRING, "value:__truediv__: op not supported for string dtype"
        Value._assert_op(self, right)
        if self.back: return Value(self.dtype, self.back.div(self.dtype, self, right), back=self.back, backend=self.backend) 
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
        if dbg: print(f'running metal kernel on {self.device}, result {view[0]}')
        return view[0]

class CBackend(Backend):
    def __init__(self):
        super().__init__('c')
        from ctypes import CDLL, c_float, c_int32
        self.c_ops = CDLL('gen/cpu_ops.so')
    
    def _get_c_op(self, dtype, op): 
        assert dtype != DType.STRING, "strings currently not supported for c backend"
        if dbg: print(f'c_cpu_backend: running {op.name}')
        if op == OPS.ADD: return self.c_ops.i_add if dtype == DType.INT else self.c_ops.f_add
        elif op == OPS.SUB: return self.c_ops.i_sub if dtype == DType.INT else self.c_ops.f_sub
        elif op == OPS.MUL: return self.c_ops.i_mul if dtype == DType.INT else self.c_ops.f_mul
        elif op == OPS.DIV: return self.c_ops.i_div if dtype == DType.INT else self.c_ops.f_div

    def add(self, dtype : DType, left : Value, right : Value): return self._get_c_op(dtype, OPS.ADD)(left.data, right.data)
    def sub(self, dtype : DType, left : Value, right : Value): return self._get_c_op(dtype, OPS.SUB)(left.data, right.data)
    def mul(self, dtype : DType, left : Value, right : Value): return self._get_c_op(dtype, OPS.MUL)(left.data, right.data)
    def div(self, dtype : DType, left : Value, right : Value): return self._get_c_op(dtype, OPS.DIV)(left.data, right.data)


import metalcompute as mc 
device = mc.Device()
x = Value(DType.INT, 50, backend='metal', metal_device=device, implicit_cast=True)
y = Value(DType.FLOAT, 50, backend='metal', metal_device=device, implicit_cast=True)

z = x + y
print(z)