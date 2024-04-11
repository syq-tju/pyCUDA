import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
#include <stdio.h>
__global__ void helloworld(void)
{
  printf("Hello World, from PyCUDA!");
}
""")

helloworld = mod.get_function("helloworld")
helloworld(block=(2,2,1), grid=(2,1))