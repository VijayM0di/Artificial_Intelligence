# import platform
# if platform.system() == 'Linux':
#     pv.start_xvfb()
    
from pynvml.smi import nvidia_smi
import os

def connect_to_gpu():
    nvsmi = nvidia_smi.getInstance()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    import torch
    # torch.cuda.set_device(0)
    # print(nvsmi.DeviceQuery('memory.used,memory.free'))
    total_free_memory = 0
    print(os.environ.get('CUDA_VISIBLE_DEVICES'))
    for i in range(torch.cuda.device_count()):
        print(f'Device {i}: {torch.cuda.get_device_name(i)}')
        print(f"memory Usage: {torch.cuda.memory_allocated(i)}")

    gpu = 0
    for i in nvsmi.DeviceQuery('memory.free, memory.used')['gpu']:
        total_free_memory = i['fb_memory_usage']['free']
        total_used_memory = i['fb_memory_usage']['used']
        if torch.cuda.is_available():
            print("GPU present")
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            torch.cuda.set_device(gpu)
            print(f'total_free_memory: {total_free_memory}, total_used_memory:  {total_used_memory}')
            if total_free_memory < 1000:
                resource = f"cuda:{gpu}"
                print(resource)
                print("GPU is available and being used")
                return torch.device(resource)
            else:
                gpu += 1
        else:
            return torch.device("cpu")

    # return ['mps' if torch.backends.mps.is_available() else 'cpu'][0]
    return 0

# device = connect_to_gpu()