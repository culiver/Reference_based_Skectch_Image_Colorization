import os
import glob
import os.path as osp
import shutil

split = 'val' 
image_folder = 'celebA1000/{}/image'.format(split)
sketch_folder = 'celebA1000/{}/sketch'.format(split)
os.makedirs(image_folder, exist_ok=True)
os.makedirs(sketch_folder, exist_ok=True)

source_image_folder = '/home/yichungc/Thesis/data/CelebA/Img/img_align_celeba'
source_sketch_folder = '/home/yichungc/Thesis/data/CelebA/Img/img_align_celeba_sketch'

source_images = glob.glob(osp.join(source_image_folder, '*.jpg'))
source_images.sort(key=lambda x:int(osp.basename(x.split('.')[0])))


source_sketches = glob.glob(osp.join(source_sketch_folder, '*.jpg'))
source_sketches.sort(key=lambda x:int(osp.basename(x.split('.')[0])))

if split == 'train':
    source_images = source_images[:1000]
    source_sketches = source_sketches[:1000]
elif split == 'val':
    source_images = source_images[1000:2000]
    source_sketches = source_sketches[1000:2000]

for img, sketch in zip(source_images, source_sketches):
    shutil.copy2(img, image_folder)
    shutil.copy2(sketch, sketch_folder)