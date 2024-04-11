import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

mod = SourceModule("""
__global__ void add(int *c, int *a, int *b, int N)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
    *c = *a + *b;
    }
}
""")

add = mod.get_function("add")
N = 100000000

start = drv.Event()
end = drv.Event()


h_a = np.random.randint(0, 100, N).astype(np.int32)
h_b = np.random.randint(0, 100, N).astype(np.int32)
d_c = np.empty_like(h_a)
h_c = np.empty_like(h_a)

block_size = 1024
grid_size = (N + block_size - 1) // block_size

start.record()
add(drv.Out(d_c), drv.In(h_a), drv.In(h_b), np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1, 1))
end.record()
end.synchronize()
print("Addition of %d elements of GPU in %fs seconds." % (N, start.time_till(end) * 0.001))

start= time.time()
for i in range(0,N):
    h_c[i] = h_a[i] + h_b[i]
end = time.time()
print("Addition of %d elements of CPU in %fs seconds." % (N, end - start))
