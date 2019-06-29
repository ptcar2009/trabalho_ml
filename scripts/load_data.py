import torch
from skimage import io as image_input
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
image_path = '../data/words/'
ground_truth_path = '../data/ascii/words.txt'
saved_data_path = '../data/saved_data/'


class Loader(Dataset):
    def __init__(self):
        self.img_size = (128, 32)
        saved_file = saved_data_path + 'pt'
        if (os.path.isfile(saved_file)):
            self.data = torch.load(saved_file)
        else:
            self.data = LoadData()
            torch.save(self.data, saved_file)

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)


def LoadData():
    data = []
    with open(ground_truth_path) as gt_file:
        for cur_img, line in enumerate(gt_file):
            if cur_img % 500 == 0:
                print('Imagem {}'.format(cur_img))
            if line[0] != '#':
                parts = line.strip().split()
                split_info = parts[0].split('-')
                full_image_path = image_path + '/' + split_info[0] + '/' + '/'.join(
                    ['-'.join(split_info[: 1 + i]) for i in range(1, len(split_info)) if i != 2])
                if parts[1] != 'ok':
                    continue
                transcript = split_info[-1]
                try:
                    image = image_input.imread(full_image_path + '.png')

                    image = image.astype(np.float32) / 255
                except Exception as exception:
                    print(exception)
                    continue
                image = formatImage(image)
                data += [(transcript, image)]
    return data


def formatImage(image):
    x_size, y_size = (128, 32)
    y_cur_size, x_cur_size = image.shape
    max_ratio = max(x_cur_size / x_size, y_cur_size / y_size)
    x_cur_size, y_cur_size = (
        min(x_size, int(x_cur_size / max_ratio)), min(y_size, int(y_cur_size / max_ratio)))
    image = cv2.resize(image, (x_cur_size, y_cur_size))
    back = np.ones((y_size, x_size))
    back[:y_cur_size, :x_cur_size] = image
    return back
