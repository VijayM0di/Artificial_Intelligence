#!/usr/local/bin/python

import os
import subprocess
import re
import time

from helpers.log_manager import logger, error_logger
from common import RequestMethod, API
from settings import (USECASE_RETRIEVAL_URL, USECASE_PATH_ID, access_token, service_command, BACKEND_URL,
                      sudo_password, project_directory, environment_directory, project_python_directory)

dev_service_file_template = f'''
[program:FILE]
command={project_python_directory} COMMAND
directory={project_directory}
environment=PYTHONPATH={project_directory}
autostart=true
autorestart=true
stderr_logfile=/var/log/FILE.err.log
stdout_logfile=/var/log/FILE.out.log
priority=997
killasgroup=true
'''

new_service_file_template = f'''
[program:FILE]
command=/usr/local/bin/python COMMAND
directory=/app/
environment=PYTHONPATH=/app/
autostart=true
autorestart=true
stderr_logfile=/var/log/FILE.err.log
stdout_logfile=/var/log/FILE.out.log
priority=997
killasgroup=true
'''


def service_start_stop(service_file_name, flag):
    print(f'{service_file_name=}')
    print(f'{flag=}')

    variable_1 = None
    variable_2 = None
    variable_3 = None
    variable_4 = None
    variable_5 = None
    if flag == 0:
        variable_1 = 'stop'
        variable_2 = 'disable'
        variable_3 = 'inactive'
        variable_4 = -1
        variable_5 = 'stopped'

    if flag == 1:
        variable_1 = 'enable'
        variable_2 = 'restart'
        variable_3 = 'active'
        variable_4 = 1
        variable_5 = 'started'

    your_command = ''

    if service_file_name == 'stream_distributor':
        your_command = 'stream_distributor.py'

    elif service_file_name.split('__')[0] == 'stream_fetcher':
        camera_id = service_file_name.split('__')[1]
        camera_ip_url = f'{BACKEND_URL}/api/v1/cameraproducts/{camera_id}'
        response_1 = API(RequestMethod.GET, camera_ip_url, headers={'Authorization': f'Bearer {access_token}'})
        camera_ip = response_1.json()['ipAddress'].split(':')[0]
        your_command = 'stream_fetcher.py ' + camera_ip

    # service = dev_service_file_template.replace('FILE', service_file_name).replace('COMMAND', your_command)
    service = new_service_file_template.replace('FILE', service_file_name).replace('COMMAND', your_command)

    # service_file = f"{service_file_name}.ini"
    service_file = f"{service_file_name}.conf"

    # file_path = f'/etc/supervisord.d/{service_file}'
    file_path = f'/etc/supervisor/conf.d/{service_file}'
    # print(f'{file_path=}')

    check_file = os.path.isfile(file_path)
    # print(check_file)

    if check_file:
        pass
    else:
        with open(file_path, 'w') as file:
            file.write(service)

    print('Running Command = ', ['sudo', 'supervisorctl', 'reread'])
    process = subprocess.Popen(['supervisorctl', 'reread'], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate(input=sudo_password.encode())

    print('Running Command = ', ['sudo', 'supervisorctl', 'update'])
    process_1 = subprocess.Popen(['supervisorctl', 'update'], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output_1, error_1 = process_1.communicate(input=sudo_password.encode())

    print('Running Command = ', ['sudo', 'supervisorctl', variable_2, service_file_name])
    process_2 = subprocess.Popen(['supervisorctl', variable_2, service_file_name], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output_2, error_2 = process_2.communicate(input=sudo_password.encode())

    status, flag = service_exists(service_file_name)
    if flag == variable_4 and variable_3 in status:
        print(f'{variable_5}: {service_file_name}')
        logger.debug(f'{variable_5}: {service_file_name}')
    else:
        print(f'service: {service_file_name} was not {variable_5} properly. Error status: {status}')
        error_logger.error(f'service: {service_file_name} was not {variable_5} properly. Error status: {status}')
        quit()


def service_exists(service_file):
    time.sleep(5)
    status = None
    flag = None
    try:
        output = subprocess.check_output(['supervisorctl', 'status', service_file], input=sudo_password,
                                         stderr=subprocess.STDOUT, text=True)
        result_start = re.search('RUNNING', output)
        status = 'active'
        flag = 1

    except subprocess.CalledProcessError as e:
        if e.returncode == 4:
            status = "Service does not exist"
            flag = 0
        elif re.search('STOPPED', e.output):
            status = 'inactive'
            flag = -1
        else:
            status = 'failed'
            flag = -1

    return [status, flag]


service_true_get = {
    'pageNumber': 0,
    'pagesize': 999,
    'orderBy': [
        'ProductId'
    ]
}

service_start_stop('stream_distributor', 1)

response_true = API(RequestMethod.POST, USECASE_RETRIEVAL_URL, json=service_true_get,
                    headers={'Authorization': f'Bearer {access_token}'})
# print('response_true = ', response_true)
data_true = response_true.json()['data']

for item in data_true:
    camera_id = item['productId']
    usecase_id = item['useCasesLinked']
    id_camera = item['id']
    service_start_stop('stream_fetcher__' + camera_id, 1)
    camera_ip_url = f'{BACKEND_URL}/api/v1/cameraproducts/{camera_id}'

    response_1 = API(RequestMethod.GET, camera_ip_url, headers={'Authorization': f'Bearer {access_token}'})
    camera_ip = response_1.json()['ipAddress'].split(':')[0]
    usecase_path = USECASE_PATH_ID[usecase_id]

    your_command = usecase_path + ' ' + camera_ip
    service_file_name = f"{camera_id}__{usecase_id}"

    # service = dev_service_file_template.replace('FILE', service_file_name).replace('COMMAND', your_command)
    service = new_service_file_template.replace('FILE', service_file_name).replace('COMMAND', your_command)

    # service_file = f"{service_file_name}.ini"
    service_file = f"{service_file_name}.conf"

    # file_path = f'/etc/supervisord.d/{service_file}'
    file_path = f'/etc/supervisor/conf.d/{service_file}'

    check_file = os.path.isfile(file_path)
    if check_file:
        pass
    else:
        with open(file_path, 'w') as file:
            file.write(service)

    process = subprocess.Popen(['supervisorctl', 'reread'], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate(input=sudo_password.encode())

    process_1 = subprocess.Popen(['supervisorctl', 'update'], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output_1, error_1 = process_1.communicate(input=sudo_password.encode())

    process_2 = subprocess.Popen(['supervisorctl', 'restart', service_file_name], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output_2, error_2 = process_2.communicate(input=sudo_password.encode())

    status, flag = service_exists(service_file_name)
    if flag == 1 and 'active' in status:
        print(f'Started: {service_file_name}')
        camera_restarted = f'{BACKEND_URL}/api/v1/camerausecases/{id_camera}/set-camera-usecase-has-restarted'
        response_2 = API(RequestMethod.POST, camera_restarted, headers={'Authorization': f'Bearer {access_token}'})

    elif flag == -1 and 'failed' in status:
        print(f'reset-failed: {service_file_name}')
        logger.debug(f'reset-failed: {service_file_name}')
        process_3 = subprocess.Popen(['supervisorctl', 'status', service_file_name], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output_3, error_3 = process_3.communicate(input=sudo_password.encode())
    else:
        error_logger.error(f'service: {service_file_name} did not run properly. Error status: {status}')

    # break
