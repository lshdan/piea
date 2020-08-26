# -*- coding: utf-8 -*-

# this version is for NIMA loss

from __future__ import print_function
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Dropout, Input
import keras.applications.mobilenet as mobilenet
#from keras.applications.mobilenet import MobileNet
#from keras.applications.mobilenet import preprocess_input

import keras.applications.inception_resnet_v2 as resnet
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras.applications.inception_resnet_v2 import preprocess_input

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras import backend as K
from keras_preprocessing import image
from PIL import Image
import scipy
import scipy.misc
import os
import numpy as np
from piea.guider import Guider
from piea.enhancer import Enhancer
import argparse

import piea.io_helper as io_helper

from piea.filters import *

import logging

logger = logging.getLogger(__name__)


# !------------ Initialization ----------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Arguments for models')
    parser.add_argument('--guider', nargs='?', default='resnet',
                        help='one of mobilenet,resnet')
    parser.add_argument('--config', nargs='?', default='example',
                        help='configuration file, full name should be config_\%s.py')
    parser.add_argument('--resize', type=int, default=0,
                        help='whether resize the images, 0: no resize, 1: resize the original image, 2: resize the generated image')
    parser.add_argument('--bsz', type=int, default=1,
                        help='the number of images in one batch')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate for the filter parameters')
    parser.add_argument('--ite', type=int, default=20,
                        help='the number of iterations for explore the parameters')
    parser.add_argument('--src', nargs='?', default='/workspace/aesthetic-guided/input2', #None,
                        help='source image directory')
    parser.add_argument('--tgt', nargs='?', default='./dxyoutput', #None,
                        help='target image directory')
    parser.add_argument('--index', nargs='?', default='filelist', #None,
                        help='index file')
    parser.add_argument('--loss', nargs='?', default='2', #None,
                        help='loss type 1, 2')

    if argv is None:
        return parser.parse_args()
    else:
        return parser.parse_args(argv)


def init_logging(tgt):
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler('{}/log.txt'.format(tgt))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # logger.info('Program started')




def main(args):

    from piea.config import cfg

    print ('got guider:', args.guider)
    if args.guider == 'mobilenet':
        cfg['guider'] = mobilenet.MobileNet
        cfg['preprocess'] = mobilenet.preprocess_input
        cfg['weight_file'] = 'weights/mobilenet_weights.h5'
    elif args.guider == 'resnet':
        cfg['guider'] = resnet.InceptionResNetV2
        cfg['preprocess'] = resnet.preprocess_input
        cfg['weight_file'] = 'weights/inception_resnet_weights.h5'
    else:
        print('unknown guider')
        exit(1)
    # !----------------  Filters for image adjust-------------------
    BATCH_SIZE=args.bsz
    WIDTH=224
    HEIGHT=224
    CHANNEL=3
    img_size = np.ndarray((1, WIDTH, HEIGHT, CHANNEL))

    import pathlib
    pathlib.Path(args.tgt).mkdir(parents=True, exist_ok=True)

    #'''
    

    # g_lr = 1e0
    cfg.filters = [
        ExposureFilter, #1e-1
        GammaFilter, # 1e-1
        ImprovedWhiteBalanceFilter, #1e-1
        SaturationPlusFilter, #1e-1
        ToneFilter, #1e0
        ContrastFilter, #1e-1
        WNBFilter, #1e0
        ColorFilter, #1e1
    ]
    '''
    importance = [
        1., 0.1, 0.7, 1., 1., 0.6, 0.05, 1.
    ]
    #'''

    
    # !----------- training ----------------
    right_way_graph = tf.Graph()

    per_process_gpu_memory_fraction = float(os.getenv('PER_PROCESS_GPU_MEMORY_FRACTION', '0.2'))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)

    with right_way_graph.as_default(), tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)).as_default() as sess:
        #### build model ####
        input_img = tf.placeholder(tf.float32, (None, None, None, 3), name='input_image')
        ie = Enhancer(input_img, cfg, img_size)
        # pixel value of ie.output are in [0,1].
        # input pixel value of Image Guilder is required in [0,255.]

        formed_imgs = []
        toig = []
        for oimg in ie.outputs:
            formed_img = tf.clip_by_value(oimg *255., 0., 255.)
            next_img = formed_img
            if args.resize == 2:
                next_img = tf.image.resize_images(formed_img, (WIDTH,HEIGHT))
            formed_imgs.append(formed_img)
            #formed_img = ie.output * 255.
            pp_img = cfg['preprocess'](next_img)
            #print 'pp_img', np.min(pp_img), np.max(pp_img)
            toig.append(pp_img)

        ig = Guider(toig, cfg['guider'], cfg['weight_file'])
        gig = []        # step gradient
        '''
        loss = 0
        for i, out in enumerate(ig.outputs):
            target = 10 - out
            sloss = tf.reduce_sum(tf.square(target))
            gig.append(tf.gradients(sloss, ie.param_holders[i])[0])
            loss = loss + cfg.importance[i] * sloss
        '''
        #'''

        from tqdm import tqdm
        fake_target = np.zeros([10])
        fake_target[9] = 1


        if args.loss == '1':
            outs = ig.outputs
        else:
            outs = ig.output_sd

        def get_loss_grads_list():
            loss_list = []
            grads_list = []
            for idx, out in enumerate(outs):
                if args.loss == '1':
                    target = 10 - out
                else:
                    target = fake_target - out

                loss = tf.reduce_sum(tf.square(target)) / 10
                #'''
                grads = tf.gradients(loss, ie.param_holders[:idx+1])    # global gradient

                loss_list.append(loss)
                grads_list.append(grads)

            return loss_list, grads_list


        loss_list, grads_list = get_loss_grads_list()


        for imgs, names in tqdm(io_helper.generator(args.src, args.index, args.tgt, args.bsz, WIDTH, HEIGHT), 
                desc="Processing Image", unit=''):
            print ("-----------  %s  -----------" % names)
            
            ie.initParams()

            img = imgs / 255.

            for idx, grads in  tqdm(enumerate(grads_list), desc="Finetuning with filters", unit=''):
                best_score = -100000.
                for i in range(args.ite):
                    feed_dict = dict(zip(ie.param_holders, ie.params))
                    feed_dict[ie.input] = img
                    outputs, gds = sess.run([ig.outputs, grads], feed_dict=feed_dict)
                    ie.updateParams(gds)
                    
                    # if (outputs[-1] > best_score):
                    #     best_params[names[0]] = ie.params
                    #     best_score = outputs[-1]
                    # print (outputs)
                    # print('gds: {}'.format(gds))
                    # print('param[{}]: {}'.format(idx, ie.params[idx]))
                
                nimg = sess.run([formed_imgs[idx]], feed_dict={
                    **dict(zip(ie.param_holders, ie.params)),
                    ie.input: img
                    })
                io_helper.saveimgs(nimg, ['{}.{}.jpg'.format(names[0], idx)], args.tgt)
                
                logger.info('parameters for %s after filter %d: %s', names, idx, ie.params)



            # ie.params = best_params[names[0]]
            feed_dict = dict(zip(ie.param_holders, ie.params))
            feed_dict[ie.input] = img
            outputs, nimg = sess.run([ig.outputs, formed_imgs[-1]], feed_dict=feed_dict)
            io_helper.saveimgs(nimg, names, args.tgt)


            # best_params[names[0]] = ie.params
            dp = np.array({
                names[0]: ie.params
            })
            np.save(os.path.join(args.tgt,'{}.best_params.npy'.format(names[0])), dp)
            # break


if __name__ == "__main__":
    args = parse_args()
    init_logging(args.tgt)
    main(args)