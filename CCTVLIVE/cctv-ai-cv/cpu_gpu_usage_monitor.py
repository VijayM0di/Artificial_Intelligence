import os
import re

from dotenv import load_dotenv
from flask import Flask, jsonify
import psutil
import subprocess
import requests

import time
from flask_cors import CORS
from settings import access_token, BACKEND_URL, USECASE_RETRIEVAL_URL, sudo_password


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


@app.route('/service-status')
def get_service_status_all():
    try:
        service_search_get = {

            'pageNumber': 0,
            'pagesize': 10000,
            'orderBy': [
                'ProductId'
            ]
        }

        response_search = requests.post(USECASE_RETRIEVAL_URL,json=service_search_get,
                                       headers={'Authorization': f'Bearer {access_token}'})
        data_search = []
        try:
            data_search = response_search.json()['data']
        except:
            pass

        service_list = []

        stream_distributor = 'stream_distributor'
        _, stream_distributor_flag = service_exists(stream_distributor)
        for item in data_search:
            dict_service = {}
            camera_id = item['productId']
            usecase_id = item['useCasesLinked']

            service_file = f"{camera_id}__{usecase_id}"

            stream_fetcher = 'stream_fetcher__'+camera_id

            if stream_distributor_flag == 1:
                _, stream_fetcher_flag = service_exists(stream_fetcher)
                if stream_fetcher_flag == 1:
                    dict_service, flag = service_exists(service_file)
                else:
                    dict_service['service_name'] = service_file
                    dict_service['service_status'] = "FAILURE"
                    dict_service['service_details'] = f"stream_fetcher service is not working for camera: {camera_id}"

            else:
                dict_service['service_name'] = service_file
                dict_service['service_status'] = "FAILURE"
                dict_service['service_details'] = "stream_distributor service is not working."

            service_list.append(dict_service)

        return jsonify(data=service_list)
    except Exception as e:
        return jsonify(data=e)


@app.route('/service-status/<usecase_id>/<camera_id>')
def get_service_status_camera(usecase_id, camera_id):
    try:
        dict_service = {}

        stream_distributor = 'stream_distributor'
        _, stream_distributor_flag = service_exists(stream_distributor)

        stream_fetcher = 'stream_fetcher__'+camera_id
        _, stream_fetcher_flag = service_exists(stream_fetcher)

        service_file = camera_id + '__' + usecase_id

        if stream_distributor_flag == 1:
            if stream_fetcher_flag == 1:
                dict_service, flag = service_exists(service_file)
            else:
                dict_service['service_name'] = service_file
                dict_service['service_status'] = "FAILURE"
                dict_service['service_details'] = f"stream_fetcher service is not working for camera: {camera_id}"

        else:
            dict_service['service_name'] = service_file
            dict_service['service_status'] = "FAILURE"
            dict_service['service_details'] = "stream_distributor service is not working."

        return jsonify(dict_service)
    except Exception as e:
        return jsonify(e)

@app.route('/service-status/<service_file>')
def get_service_status(service_file):
    try:
        dict_service, flag = service_exists(service_file)

        return jsonify(dict_service)
    except Exception as e:
        return jsonify(e)


def service_exists(service_file):
    dict_service = {}
    path = '/etc/systemd/system/'
    flag = None
    try:
        output = subprocess.check_output(['sudo', '-S', 'systemctl', 'status', service_file], input=sudo_password,
                                         stderr=subprocess.STDOUT, text=True)
        result_start = re.search('Active:', output)
        result_end = re.search(r"\)", output[int(result_start.end()):])
        dict_service['service_name'] = service_file
        dict_service['service_status'] = "OK"
        dict_service['service_details'] = output[result_start.end():result_start.end() + result_end.start() + 1].strip()
        flag = 1

    except subprocess.CalledProcessError as e:
        if e.returncode == 4:
            dict_service['service_name'] = service_file
            dict_service['service_status'] = "FAILURE"
            dict_service['service_details'] = 'Service does not exist'
            flag = 0
        else:
            result_start = re.search('Active:', e.output)
            result_end = re.search(r"\)", e.output[int(result_start.end()):])

            dict_service['service_name'] = service_file
            dict_service['service_status'] = "FAILURE"
            dict_service['service_details'] = e.output[result_start.end():result_start.end() + result_end.start() + 1].strip() + e.output
            flag = -1
    return [dict_service, flag]


if __name__ == '__main__':
    FLASK_PORT = os.getenv('FLASK_PORT')
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=True)
