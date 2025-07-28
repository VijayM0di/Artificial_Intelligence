import os
import re
import subprocess

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from common import RequestMethod, API
from settings import USECASE_PATH_ID, access_token, service_command, BACKEND_URL, sudo_password, project_directory, \
    environment_directory, project_python_directory

app = Flask(__name__)
cors = CORS(app)
load_dotenv()

API_KEY_HEADER = os.environ.get('API_KEY_HEADER')
API_KEY_VALUE = os.environ.get('API_KEY_VALUE')
sudo_password = os.environ.get('SUDO_PASSWORD')

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
stopsignal=SIGTERM
stopwaitsecs=10
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
stopsignal=SIGTERM
stopwaitsecs=10
'''


def service_start_fetcher(service_file_name, action):
    print(f"Stream Fetcher = {service_file_name}")

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

        process = subprocess.Popen(['supervisorctl', 'reread'], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate(input=sudo_password.encode())

        process_1 = subprocess.Popen(['supervisorctl', 'update'], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output_1, error_1 = process_1.communicate(input=sudo_password.encode())

    if action == 'start':
        start_service(service_file_name)
    else:
        stop_service(service_file_name)

    status = check_status(service_file_name)

    return status


@app.route('/ai-services/<camera_id>/<usecase_id>/<action_type>', methods=['GET'])
def get_service_status_camera(camera_id, usecase_id, action_type):
    try:
        if API_KEY_HEADER not in request.headers:
            return {"message": "Invalid API key header"}, 400

        if not request.headers.get("X-Api-Key") == API_KEY_VALUE:
            return {"message": "Invalid API key value"}, 401

        camera_ip = get_camera_ip(camera_id)
        usecase_path = USECASE_PATH_ID[usecase_id]
        your_command = usecase_path + ' ' + camera_ip

        service_file_name = f"{camera_id}__{usecase_id}"
        service = new_service_file_template.replace('FILE', service_file_name).replace('COMMAND', your_command)

        # service_file = f"{service_file_name}.ini"
        service_file = f"{service_file_name}.conf"

        # file_path = f'/etc/supervisord.d/{service_file}'
        file_path = f'/etc/supervisor/conf.d/{service_file}'
        print(f'{service_file=}')

        check_file = os.path.isfile(file_path)
        if check_file:
            pass
        else:
            with open(file_path, 'w') as file:
                file.write(service)

            print('New service detected = ', service_file)
            # print('Rereading Processes')
            process = subprocess.Popen(['supervisorctl', 'reread'], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate(input=sudo_password.encode())

            # print('Updating Processes')
            process_1 = subprocess.Popen(['supervisorctl', 'update'], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output_1, error_1 = process_1.communicate(input=sudo_password.encode())

            # print('Making Process Inactive')
            process_1 = subprocess.Popen(['supervisorctl', 'stop', service_file_name], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output_1, error_1 = process_1.communicate(input=sudo_password.encode())

        ctx = {}

        if action_type == 'status':
            status = check_status(service_file_name)
            ctx['status'] = status
            ctx['message'] = f'Service is {status}'
        elif action_type == 'start':
            start_service(service_file_name)
            stream_fetcher = service_start_fetcher('stream_fetcher__' + camera_id, action_type)
            status = check_status(service_file_name)
            ctx['status'] = status
            ctx['message'] = f'Service is {status}, Stream Fetcher Status is {stream_fetcher}'
        elif action_type == 'stop':
            stop_service(service_file_name)
            stream_fetcher = service_start_fetcher('stream_fetcher__' + camera_id, action_type)
            status = check_status(service_file_name)
            ctx['status'] = status
            ctx['message'] = f'Service is {status}, Stream Fetcher Status is {stream_fetcher}'
        else:
            return {'message': 'Action type not found'}, 400

        return jsonify(ctx)
    except Exception as e:
        return jsonify(e)


def get_camera_ip(camera_id):
    camera_ip_url = f'{BACKEND_URL}/api/v1/cameraproducts/{camera_id}'
    response = API(RequestMethod.GET, camera_ip_url, headers={'Authorization': f'Bearer {access_token}'})
    camera_ip = response.json()['ipAddress'].split(':')[0]
    return camera_ip


def check_status(service_file):
    status = None
    try:
        output = subprocess.check_output(['supervisorctl', 'status', service_file], input=sudo_password,
                                         stderr=subprocess.STDOUT, text=True)

        if re.search('RUNNING', output):
            status = 'active'
        elif re.search('STARTING', output):
            status = 'started'
        else:
            status = 'unknown'

    except subprocess.CalledProcessError as e:

        if e.returncode == 4:
            status = "does not exist"
        elif re.search('STOPPED', e.output):
            status = 'inactive'
        else:
            status = 'failed'

    return status


def start_service(service_file):
    process = subprocess.Popen(['supervisorctl', 'start', service_file], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    output, error = process.communicate(input=sudo_password.encode())


def stop_service(service_file):
    process = subprocess.Popen(['supervisorctl', 'stop', service_file], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    output, error = process.communicate(input=sudo_password.encode())


if __name__ == '__main__':
    FLASK_PORT = os.getenv('AI_SERVICES_API_PORT')
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=True)
