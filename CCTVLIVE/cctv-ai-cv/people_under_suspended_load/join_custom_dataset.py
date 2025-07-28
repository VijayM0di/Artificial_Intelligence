import os.path
import shutil
from os import walk

og_path = '../TRUCK_DETECTION/container_custom_data'

f = []
for (dirpath, dirnames, filenames) in walk(og_path):
    if filenames:
        for file in filenames:
            if file.endswith('txt') and 'classes' not in file:
                # print(file)
                # shutil.copy(os.path.join(dirpath, file), '../TRUCK_DETECTION/container_custom_data/')
                img_file = file.rstrip('txt') + 'jpg'
                # shutil.copy(os.path.join(dirpath, img_file), '../TRUCK_DETECTION/container_custom_data/')
                f.append(file)
                f.append(img_file)


og_path = 'datasets/custom_dataset'

for (dirpath, dirnames, filenames) in walk(og_path):
    if filenames:
        for file in filenames:
            if file in f:
                print(file)
                os.remove(os.path.join(dirpath, file))