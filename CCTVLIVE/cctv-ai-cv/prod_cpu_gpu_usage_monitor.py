import os
import re

from dotenv import load_dotenv
from flask import Flask, jsonify
import psutil
import subprocess
import requests

import time
from flask_cors import CORS


app = Flask(__name__)
cors = CORS(app)
load_dotenv()


@app.route('/memory-usage')
def get_memory_usage():
    svmem = psutil.virtual_memory()
    memory_used = svmem.used / (1024 ** 2)
    memory_percentage = svmem.percent
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_percent = str(cpu_percent) + ' %'
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'])
    gpu_str = 'GPU {gpu_id}'
    gpu_data = result.decode('utf-8').split('\n')
    gpu_data = gpu_data[:4]
    i = 0
    all_data = []
    for item in gpu_data:
        gpu_percent = f'{gpu_str.format(gpu_id=i)}: {item}'
        all_data.append(gpu_percent)
        i += 1
    return jsonify({
        "Memory Used (MBs)": f"{memory_used:.2f}",
        "Memory Used (%)": f"{memory_percentage} %",
        "cpu_percent": cpu_percent,
        "gpu_percent": all_data
    })


@app.route('/storage-usage')
def get_storage_usage():
    total = 0
    free = 0
    for i in psutil.disk_partitions(all=True):
        try:
            dirs = i.mountpoint
            total += psutil.disk_usage(dirs).total // (2 ** 30)
            free += psutil.disk_usage(dirs).free // (2 ** 30)
        except:
            continue
    used = total - free
    used_per = round((100 * used / total), 2)
    free_per = round((100 * free / total), 2)
    return jsonify({
        "Total Storage Space (GBs)": total,
        "Storage Used (GBs)": used,
        "Storage Used (%)": f"{used_per} %",
        "Free Space (Gbs)": free,
        "Free Space (%)": f"{free_per} %"
    })
    

@app.route('/cpu-usage')
def get_cpu_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_percent = str(cpu_percent) + ' %'
    return jsonify(cpu_percent=cpu_percent)


@app.route('/gpu-usage')
def get_gpu_usage():
    try:
        # Use nvidia-smi to get GPU usage
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'])
        gpu_str = 'GPU {gpu_id}'
        gpu_data = result.decode('utf-8').split('\n')
        gpu_data = gpu_data[:4]
        i = 0
        all_data = []
        for item in gpu_data:
            gpu_percent = f'{gpu_str.format(gpu_id=i)}: {item}'
            all_data.append(gpu_percent)
            i += 1
        return jsonify(gpu_percent=all_data)
    except Exception as e:
        return jsonify(gpu_percent=None)



if __name__ == '__main__':
    FLASK_PORT = os.getenv('FLASK_PORT')
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=True)
