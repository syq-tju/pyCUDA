import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

module = SourceModule("""
#include <stdio.h>
__global__ void add(float *a, float *b, float *c)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] = a[i] + b[i];
}
""")
N = 100000000  
add = module.get_function("add")
h_a = np.random.randn(N).astype(np.float32)
h_b = np.random.randn(N).astype(np.float32)
d_a = drv.mem_alloc(h_a.nbytes)
d_b = drv.mem_alloc(h_b.nbytes)
d_c = drv.mem_alloc(h_a.nbytes)
drv.memcpy_htod(d_a, h_a)
drv.memcpy_htod(d_b, h_b)

block_size = 1024
grid_size = (N + block_size - 1) // block_size  # 计算所需的grid大小
add(d_a, d_b, d_c, block=(block_size, 1, 1), grid=(grid_size, 1))

h_c = np.empty_like(h_a)
drv.memcpy_dtoh(h_c, d_c)
print(h_c)