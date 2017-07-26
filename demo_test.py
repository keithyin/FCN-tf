import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

from PIL import Image
from skimage.transform import resize, rescale
from skimage.io import imread, imsave

img_path = '/media/fanyang/workspace/DataSet/VOCdevkit/VOC2012/SegmentationClass/2007_000033.png'


img_obj = Image.open(img_path)
print(img_obj)

img = np.array(img_obj, dtype=np.uint8)


resized = resize(img, (10, 10), mode='constant', preserve_range=True).astype(np.uint8)

print(resized)

re_img_obj = Image.fromarray(resized, mode='P')
re_img_obj = re_img_obj.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)
print(re_img_obj)

re_img_obj.save('demo_img.png')

print(img.shape)
# a = np.array([1, 2, 3, 4, 2, 1, 2, 3])
# a[a == 2] = 0
# print(a)
