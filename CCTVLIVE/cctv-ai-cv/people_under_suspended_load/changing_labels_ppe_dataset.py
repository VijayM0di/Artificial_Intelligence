from os import walk

og_path = '../TRUCK_DETECTION/container_custom_data'

f = []
for (dirpath, dirnames, filenames) in walk(og_path):
    for item in dirnames:
        current_path = f'{og_path}/{item}/labels'
        for dir_path, dir_name, file_names in walk(current_path):
            for file in file_names:
                current_file = f'{current_path}/{file}'
                with open(f'{current_file}', 'r') as label_file:
                    lines = label_file.readlines()
                for line in lines.copy():
                    new_line = line.replace(line[0], '1', 1)
                    lines.remove(line)
                    lines.append(new_line)
                with open(f'{current_file}', 'w') as label_file:
                    lines = label_file.writelines(lines)
