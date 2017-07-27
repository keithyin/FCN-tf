from skimage.io import imread
import os
import numpy as np

COLOR_MAPS = [[0, 0, 0], [128, 64, 0], [192, 128, 128], [192, 0, 128], [128, 192, 0], [128, 128, 0],
              [0, 128, 128], [0, 64, 0], [128, 0, 128], [192, 0, 0], [192, 128, 0], [0, 0, 128], [0, 128, 0],
              [64, 0, 0], [128, 128, 128], [64, 128, 0], [0, 192, 0], [64, 0, 128], [128, 0, 0], [64, 128, 128],
              [0, 64, 128]]
NUM_COLORS = 22


def generate_label(img_dir):
    imgs = os.listdir(img_dir)
    imgs_path = [os.path.join(img_dir, img) for img in imgs]
    num_image = len(imgs_path)
    color_maps = []
    for index_img, img_path in enumerate(imgs_path, start=1):
        print('processing image %d/%d' % (index_img, num_image))

        img_nd = imread(img_path)
        height, width, _ = img_nd.shape
        for i in range(height):
            for j in range(width):
                color = list(img_nd[i, j])
                if not (color in color_maps):
                    color_maps.append(color)
    print(color_maps)
    print("num colors %d " % (len(color_maps)))

def label2img(label):
    if len(label.shape) != 2:
        raise ValueError("lable must be 2-D array, the given is %d-D"%len(label.shape))

    img = []
    height, width = label.shape
    for i in range(height):
        row_color = []
        for j in range(width):
            row_color.append(COLOR_MAPS[label[i, j]])
        img.append(row_color)
    return np.array(img).astype(np.uint8)

def main():
    img_dir = '/media/fanyang/workspace/DataSet/VOCdevkit/VOC2012/SegmentationClass'
    generate_label(img_dir)

if __name__ == '__main__':
    main()
