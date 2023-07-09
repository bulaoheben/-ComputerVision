import tensorflow as tf

# 检查GPU是否可用
print("GPU Available:", tf.test.is_gpu_available())

# 输出GPU设备信息
print("GPU Devices:")
for device in tf.config.list_physical_devices("GPU"):
    print(device)
