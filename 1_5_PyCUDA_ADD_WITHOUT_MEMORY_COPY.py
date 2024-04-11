import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

module = SourceModule("""
__global__ void add_without_memory_copy(float *c, float *a, float *b, int N)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) { 
        c[i] = a[i] + b[i];
    }
}
""")

add_without_memory_copy = module.get_function("add_without_memory_copy")

N = 1000000
h_a = np.random.randn(N).astype(np.float32)
h_b = np.random.randn(N).astype(np.float32)
h_c = np.empty_like(h_a)

block_size = 1024
grid_size = (N + block_size - 1) // block_size

add_without_memory_copy(drv.Out(h_c), drv.In(h_a), drv.In(h_b), np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1)) 

for i in range(5):
  print(h_a[i]," + ",h_b[i]," = ",h_c[i])
