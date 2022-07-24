import os
import os.path as osp
import glob
import numpy as np

from torch.utils import data
from torchvision import transforms as T
from tps_transformation import tps_transform
from PIL import Image
from utils import elastic_transform
import random
import torch
import cv2
import skimage.measure

def average_pool(mat, kernel_size):
    M, N = mat.shape[:2]
    K, L = kernel_size

    MK = M // K
    NL = N // L
    return mat[:MK*K, :NL*L].reshape(MK, K, NL, L, mat.shape[-1]).max(axis=(1, 3))



class DataSet(data.Dataset):

    def __init__(self, config, img_transform_gt, img_transform_sketch, mode):
        self.img_transform_gt = img_transform_gt
        self.img_transform_sketch = img_transform_sketch
        self.mode = mode
        # self.img_dir = osp.join(config['TRAINING_CONFIG']['IMG_DIR'], config['TRAINING_CONFIG']['MODE'])
        self.img_size = (config['MODEL_CONFIG']['IMG_SIZE'], config['MODEL_CONFIG']['IMG_SIZE'], 3)
        self.img_dir = config['TRAINING_CONFIG']['IMG_DIR']

        self.data_list = glob.glob(os.path.join(self.img_dir, mode, 'image', '*.jpg'))
        self.data_list = [x.split(os.sep)[-1].split('.')[0] for x in self.data_list]
        self.data_list = list(set(self.data_list))
        #random.seed(config['TRAINING_CONFIG']['CPU_SEED'])

        self.augment = config['TRAINING_CONFIG']['AUGMENT']

        self.dist = config['TRAINING_CONFIG']['DIST']
        if self.dist == 'uniform':
            self.a = config['TRAINING_CONFIG']['A']
            self.b = config['TRAINING_CONFIG']['B']
        else:
            self.mean = config['TRAINING_CONFIG']['MEAN']
            self.std = config['TRAINING_CONFIG']['STD']

    def __getitem__(self, index):
        fid = self.data_list[index]
        reference = Image.open(osp.join(self.img_dir, self.mode, 'image', '{}.jpg'.format(fid))).convert('RGB')
        sketch = Image.open(osp.join(self.img_dir, self.mode, 'sketch', '{}.jpg'.format(fid))).convert('L')

        if self.mode == 'train':
            if self.dist == 'uniform':
                noise = np.random.uniform(self.a, self.b, 3)
            else:
                noise = np.random.normal(self.mean, self.std, 3)
            reference = np.clip(np.array(reference) + noise, 0, 255)
        else:
            reference = np.array(reference)
        reference = Image.fromarray(reference.astype('uint8'))

        if self.augment == 'elastic':
            augmented_reference = elastic_transform(np.array(reference), 1000, 8, random_state=None)
            augmented_reference = Image.fromarray(augmented_reference)
        elif self.augment == 'tps':
            augmented_reference, map_inv = tps_transform(np.array(reference))
            augmented_reference = Image.fromarray(augmented_reference)
        else:
            augmented_reference = reference

        augmented_reference = self.img_transform_gt(augmented_reference)
        reference = self.img_transform_gt(reference)
        map_inv = cv2.resize(map_inv, (self.img_size[0], self.img_size[1]))
        map_inv = average_pool(map_inv, (16,16)) // 16
        map_inv = np.clip(map_inv.astype(int), 0, 15)
        
        if self.mode == 'train':
            return fid, augmented_reference, reference, self.img_transform_sketch(sketch), map_inv
        else:            
            return fid, augmented_reference, reference, self.img_transform_sketch(sketch)

    def __len__(self):
        """Return the number of images."""
        return len(self.data_list)


def get_loader(config):

    img_transform_gt = list()
    img_transform_sketch = list()
    img_size = config['MODEL_CONFIG']['IMG_SIZE']

    img_transform_gt.append(T.Resize((img_size, img_size)))
    img_transform_gt.append(T.ToTensor())
    img_transform_gt.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    img_transform_gt = T.Compose(img_transform_gt)

    img_transform_sketch.append(T.Resize((img_size, img_size)))
    img_transform_sketch.append(T.ToTensor())
    img_transform_sketch.append(T.Normalize(mean=(0.5), std=(0.5)))
    img_transform_sketch = T.Compose(img_transform_sketch)

    dataset = DataSet(config, img_transform_gt, img_transform_sketch, mode=config['TRAINING_CONFIG']['MODE'])
    config['TRAINING_CONFIG']['BATCH_SIZE'] = 1 if config['TRAINING_CONFIG']['MODE'] == 'val' else config['TRAINING_CONFIG']['BATCH_SIZE']
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['TRAINING_CONFIG']['BATCH_SIZE'],
                                  shuffle=(config['TRAINING_CONFIG']['MODE'] == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    return data_loader
