import os
import time
import psutil

from pynvml.smi import nvidia_smi

from helpers.log_manager import logger

nvsmi = nvidia_smi.getInstance()
cpu_utilisations = []
gpu_utilisations = [[] for _ in range(4)]
memory_usages = []

start_time = time.time()
duration = 120

while time.time() - start_time < duration:
    cpu_utilisations.append(psutil.cpu_percent(interval=1))
    count = 0
    # gpu_outputs = os.popen('nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits').readlines()
    for i in nvsmi.DeviceQuery('memory.free, memory.used')['gpu']:
        total_free_memory = i['fb_memory_usage']['free']
        total_used_memory = i['fb_memory_usage']['used']
        use_percent = (total_used_memory / (total_free_memory + total_used_memory)) * 100
    # for idx, gpu_output in enumerate(gpu_outputs):
        gpu_utilisations[count].append(float(use_percent))
        count += 1

    memory_info = psutil.virtual_memory()
    memory_usages.append(memory_info.percent)

    time.sleep(1)

avg_cpu_util = sum(cpu_utilisations) / len(cpu_utilisations)
avg_gpu_utils = [sum(gpu_util) / len(gpu_util) for gpu_util in gpu_utilisations]
avg_memory_usage = sum(memory_usages) / len(memory_usages)

logger.debug("Average CPU Util: ", avg_cpu_util)
logger.debug("Average GPU Utils: ", avg_gpu_utils)
logger.debug("Average Memory Util: ", avg_memory_usage)