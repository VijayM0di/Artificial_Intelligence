import os
import shutil

for filename in os.listdir('person_dataset'):
    if filename.endswith('.jpg'):
        if os.path.exists(os.path.join('person_dataset', filename.split('.jpg')[0] + '.txt')):
            shutil.copy(os.path.join('person_dataset', filename), 'dataset/images')
            shutil.copy(os.path.join('person_dataset', filename.split('.jpg')[0] + '.txt'), 'dataset/labels')


for filename in os.listdir(os.path.join(os.getcwd(), 'PPE-Dataset-v3i/valid/labels')):
    with open(os.path.join(os.getcwd(), f'PPE-Dataset-v3i/valid/labels/{filename}'), 'r') as f:
        lines = f.readlines()
        for line in lines.copy():
            lines.remove(line)
            class_map = {'2': '0', '3': '1', '0': '1'}
            if line[0] in class_map:
                line = class_map[line[0]] + line[1:]
            lines.append(line)
            print(line)
        outfile = open(os.path.join(os.getcwd(), f'PPE-Dataset-v3i/valid/labels/{filename}'), 'w')
        outfile.writelines(lines)
