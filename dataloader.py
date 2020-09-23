import os
import os.path as osp
import glob
import numpy as np

from torch.utils import data
from torchvision import transforms as T
from PIL import Image
from utils import elastic_transform


class DataSet(data.Dataset):

    def __init__(self, config, img_transform):
        self.img_transform = img_transform
        self.img_dir = osp.join(config['TRAINING_CONFIG']['IMG_DIR'], config['TRAINING_CONFIG']['MODE'])
        self.img_size = (config['MODEL_CONFIG']['IMG_SIZE'], config['MODEL_CONFIG']['IMG_SIZE'], 3)

        self.data_list = glob.glob(os.path.join(self.img_dir, '*.png'))
        self.data_list = [x.split(os.sep)[-1].split('_')[0] for x in self.data_list]
        self.data_list = list(set(self.data_list))
        #random.seed(config['TRAINING_CONFIG']['CPU_SEED'])

        self.dist = config['TRAINING_CONFIG']['DIST']
        if self.dist == 'uniform':
            self.a = config['TRAINING_CONFIG']['A']
            self.b = config['TRAINING_CONFIG']['B']
        else:
            self.mean = config['TRAINING_CONFIG']['MEAN']
            self.std = config['TRAINING_CONFIG']['STD']

    def __getitem__(self, index):
        fid = self.data_list[index]
        reference = Image.open(osp.join(self.img_dir, '{}_color.png'.format(fid))).convert('RGB')
        sketch = Image.open(osp.join(self.img_dir, '{}_sketch.png'.format(fid))).convert('L')

        elastic_reference = elastic_transform(np.array(reference), 1000, 8, random_state=None)
        elastic_reference = Image.fromarray(elastic_reference)

        if self.dist == 'uniform':
            noise = np.random.uniform(self.a, self.b, np.shape(reference))
        else:
            noise = np.random.uniform(self.mean, self.std, np.shape(reference))

        reference = np.array(reference) + noise
        reference = Image.fromarray(reference.astype('uint8'))

        return fid, self.img_transform(elastic_reference), self.img_transform(reference), self.img_transform(sketch)

    def __len__(self):
        """Return the number of images."""
        return len(self.data_list)


def get_loader(config):

    img_transform_gt = list()
    img_size = config['MODEL_CONFIG']['IMG_SIZE']

    img_transform_gt.append(T.Resize((img_size, img_size)))
    img_transform_gt.append(T.ToTensor())
    img_transform_gt.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    img_transform = T.Compose(img_transform_gt)

    dataset = DataSet(config, img_transform)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['TRAINING_CONFIG']['BATCH_SIZE'],
                                  shuffle=(config['TRAINING_CONFIG']['MODE'] == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    return data_loader
