import torch
import time
import subprocess

def run_task_on_device(device_id):
     # 设置当前设备
     torch.cuda.set_device(device_id)
     
     # 直接在对应的 CUDA 设备上创建 Tensor
     x = torch.Tensor([0]).cuda(device_id)

     x = x + x

     # time.sleep(0.00000000001)  # 模拟任务运行时间
     

def main():
     # 获取可用的 CUDA 设备数量
     device_count = torch.cuda.device_count()

     if device_count == 0:
          print("No CUDA devices found.")
          return
     
     while True:
          start_time = time.time()
          # 运行3分钟
          while time.time() - start_time < 360:
               for device_id in range(device_count):
               # for device_id in [0,]:
                    run_task_on_device(device_id)
          

          # 暂停10分钟
          time.sleep(600) # 600秒 = 10分钟
          print(f"Device {device_id}: 2 minutes passed, now resting for 10 minutes.")



def get_free_devices_from_nvidia_smi(threshold=5):
    """
    使用 nvidia-smi 检测 GPU 占用率
    """
    free_devices = []
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
    usage_list = [int(x) for x in result.decode().strip().split("\n")]
     
    for device_id, usage in enumerate(usage_list):
        if usage < threshold: # GPU 使用率低于阈值
            free_devices.append(device_id)
    return free_devices          
          
if __name__ == "__main__":
    # print(get_free_devices_from_nvidia_smi())
    main()