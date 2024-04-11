import pycuda.driver as drv
import pycuda.autoinit
#drv.init()

print("%d devices found." % drv.Device.count())

for i in range(drv.Device.count()):
  dev = drv.Device(i)
  device = drv.Device(i)
  print("Device %d: %s" % (i, device.name()))
  print("Compute Capability: %d.%d" % device.compute_capability())
  print("Memory: %d GB" % (device.total_memory() / 1024 / 1024 / 1024))
  print("Clock Rate: %d GHz" % (device.clock_rate/1000))
  print("Multiprocessors: %d" % device.multiprocessor_count)
  print("Max Threads Per Multiprocessor: %d" % device.max_threads_per_multiprocessor)
  print("Max Threads Per Block: %d" % device.max_threads_per_block)
  print("Max Shared Memory Per Block: %d" % device.max_shared_memory_per_block)

  # 获取最大线程块维度
  max_block_dim_x = device.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_X)
  max_block_dim_y = device.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_Y)
  max_block_dim_z = device.get_attribute(drv.device_attribute.MAX_BLOCK_DIM_Z)
  print("Max Block Dimensions: (%d, %d, %d)" % (max_block_dim_x, max_block_dim_y, max_block_dim_z))

  attributes = [(str(prop), value)
    for prop, value in list(dev.get_attributes().items())]
  attributes.sort()
  n = 0
  for name, value in attributes:
    n += 1
    print("Attribute %d: %s = %s" % (n, name, value))

  #print("Attributes: %s" % attributes)


