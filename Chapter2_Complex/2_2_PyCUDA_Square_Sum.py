import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

N = 100000
h_a = np.random.randn(N).astype(np.float32)
h_b = np.random.randn(N).astype(np.float32)

module = SourceModule("""
__global__ void square_sum(float *sum, float *a, float *b, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        float val_a = a[index];
        float val_b = b[index];
        atomicAdd(sum, val_a * val_a + val_b * val_b); // 正确计算平方和
    }
}
""")

# 从模块中获取kernel函数
square_sum_kernel = module.get_function("square_sum")

# 为结果分配单个浮点数的空间并初始化为0
sum_gpu = np.array([0], dtype=np.float32)
sum_gpu_device = drv.mem_alloc(sum_gpu.nbytes)
drv.memcpy_htod(sum_gpu_device, sum_gpu)

# 计算grid和block的大小
block_size = 1024
grid_size = (N + block_size - 1) // block_size

# 调用kernel函数，这次直接传递设备内存地址
square_sum_kernel(sum_gpu_device, drv.In(h_a), drv.In(h_b), np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1))

# 将结果从GPU复制回CPU
drv.memcpy_dtoh(sum_gpu, sum_gpu_device)

# 打印结果
print("The square sum of the array is:", sum_gpu[0])

# 假设 h_a, h_b 是输入数组，sum_gpu[0] 是CUDA计算得到的结果

# 在Python中使用NumPy进行同样的计算
sum_np = np.sum(h_a * h_a + h_b * h_b)

# 比较CUDA和NumPy的结果，允许一定的误差范围
if np.allclose(sum_gpu[0], sum_np, atol=1e-3):
    print("The CUDA and NumPy results are close enough! Verification passed.")
    print("CUDA result:", sum_gpu[0], "NumPy result:", sum_np)
else:
    print("The results differ! Verification failed.")
    print("CUDA result:", sum_gpu[0], "NumPy result:", sum_np)


