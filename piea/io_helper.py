import re, pathlib
from PIL import Image
import numpy as np

def get_completed_names(tgt):
    import re
    for item in pathlib.Path(tgt).glob('**/*.npy'):
        m = re.search(r'^(.+).best_params.npy$', item.name)
        if m:
            yield m.group(1)
            

def generator(src, index_file, tgt, bsz, width, height):

    se = set(get_completed_names(tgt))

    with open(src + '/' + index_file) as f:
        fs = list(map(str.strip, f.readlines()))

    def filter_fs():
        for item in fs:
            if item in se:
                print('{}.best_params.npy exists, skipping....'.format(item))
            else:
                yield item

    fs = list(filter_fs())
    
    sz = len(fs)
    for i in range(0, sz, bsz):
        yield np.array(list(map(lambda  x: load(x, src, width, height), fs[i:i+bsz]))), fs[i:i+bsz]

# 8-bits
def load(fname, src, width, height, resize=1):
    img = Image.open(src + '/' + fname)
    if resize == 1:
        img = img.resize((width, height))

    img = np.array(img, dtype=np.float32)
    return img

# float32 (0~255) -> 8-bits
def filtimg(img):
    if len(img.shape) > 3:
        img = img[0,...]
    if img.dtype != np.uint8:
        img = img.astype(dtype=np.uint8)
    return img

def saveimgs(imgs, names, tgt):
    for name, img in zip(names, imgs):
        saveimg(name, filtimg(img), tgt)

# 8-bits
def saveimg(fname, img, tgt):
    im = Image.fromarray(img)
    im.save(tgt + '/' + fname)
    #scipy.misc.imsave('outfile.jpg', img)
