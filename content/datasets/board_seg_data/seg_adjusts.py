import os
from random import random
from shutil import move

train = ''
val = ''

for name in os.listdir(f'images/train'):

    image_path = f'images/train/{name}'
    label_path = f'labels/train/{name[:-3]+"txt"}'

    if random() <0.15:
        move(
            f'images/train/{name}',
            f'images/val/{name}'
        )
        move(
            f'labels/train/{name[:-3] + "txt"}',
            f'labels/val/{name[:-3] + "txt"}'
        )
        val += f'images/val/{name}\n'
    else:
        train += f'images/train/{name}\n'

with open('train.txt', 'w') as file:
    file.writelines(train)

with open('val.txt', 'w') as file:
    file.writelines(val)