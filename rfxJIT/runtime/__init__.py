"""Runtime prototypes for rfxJIT."""

from rfxJIT.runtime.executor import execute_lowered_kernel
from rfxJIT.runtime.interpreter import execute_kernel
from rfxJIT.runtime.queue import KernelDispatchQueue

__all__ = ["KernelDispatchQueue", "execute_kernel", "execute_lowered_kernel"]
