import os
import glob
import os.path as osp
import shutil

dataNum = 10000
image_folder_train = 'celebA{}/{}/image'.format(dataNum, 'train')
sketch_folder_train = 'celebA{}/{}/sketch'.format(dataNum, 'train')
image_folder_val = 'celebA{}/{}/image'.format(dataNum, 'val')
sketch_folder_val = 'celebA{}/{}/sketch'.format(dataNum, 'val')
os.makedirs(image_folder_train, exist_ok=True)
os.makedirs(sketch_folder_train, exist_ok=True)
os.makedirs(image_folder_val, exist_ok=True)
os.makedirs(sketch_folder_val, exist_ok=True)

source_image_folder = '/home/yichungc/Thesis/data/CelebA/Img/img_align_celeba'
source_sketch_folder = '/home/yichungc/Thesis/data/CelebA/Img/img_align_celeba_sketch'

source_images = glob.glob(osp.join(source_image_folder, '*.jpg'))
source_images.sort(key=lambda x:int(osp.basename(x.split('.')[0])))

source_sketches = glob.glob(osp.join(source_sketch_folder, '*.jpg'))
source_sketches.sort(key=lambda x:int(osp.basename(x.split('.')[0])))

source_images_train = source_images[:dataNum]
source_sketches_train = source_sketches[:dataNum]

source_images_val = source_images[dataNum:dataNum+1000]
source_sketches_val = source_sketches[dataNum:dataNum+1000]

for img, sketch in zip(source_images_train, source_sketches_train):
    shutil.copy2(img, image_folder_train)
    shutil.copy2(sketch, sketch_folder_train)

for img, sketch in zip(source_images_val, source_sketches_val):
    shutil.copy2(img, image_folder_val)
    shutil.copy2(sketch, sketch_folder_val)