# -*- coding=utf-8 -*-
from PIL import Image
import numpy as np
import os.path as osp
import os


def getFilelist(path, ext):
    """
    get files path which have specified extension as a list recursiveliy.
    path: str, directory path you want to search
    ext: str, extension

    output: list, the components are file path
    """
    t = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                t.append(os.path.join(root, file))
    return t


lr = "1e-4"
epoch = 16
method = "U-net_patch" # or U-net_patch

path = "/home/sora/project/ips/PSPnet2/result/5/"
#print(path)
for i in [5]:
    dataset = "label"
    impath = osp.join(path, dataset)
    print(impath)
    im_list = getFilelist(path, "PNG")
    for im in im_list:
        im_name = im.split("/")[-1]
        name, ext = osp.splitext(im_name)
        ext = ext.lower()
        im_name = name + ext
        save_path = "/".join([path, dataset, im_name])
        img = np.array(Image.open(im), dtype=int)
        img = img / 255
        label = img[:, :, 0] * 1 + img[:, :, 1] * 2 + img[:, :, 2] * 3
        mask = Image.fromarray(np.uint8(label))
        mask.save(save_path)
