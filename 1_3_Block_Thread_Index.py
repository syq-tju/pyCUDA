import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
#include<stdio.h>
__global__ void block_thread()
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  printf("tid = %d, bid = %d\\n", tid, bid);
}
""")

block_thread = mod.get_function("block_thread")
block_thread(block=(2, 2, 1), grid=(2, 1))
