import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
from piea.util import lrelu, rgb2lum, tanh_range, lerp
import cv2
import math


class Enhancer():
    def __init__(self, input, cfg, img_size):
        self.input = input
        self.param_holders = []
        self.filters = []
        self.params = []
        self.outputs = []
        self.cfg = cfg
        self.output = None
        self.lr = 0.1
        self.img_size = img_size
        self.model()

    def initParams(self):
        bsz = 1
        self.params = []
        for filter in self.filters: # generate params from the filter instances
            pn = filter.get_num_filter_parameters()
            #param = np.array(np.random.randn(bsz, pn), dtype=np.float32)
            param = np.zeros((bsz, pn), dtype=np.float32)
            self.params.append(param)
        self.lastv = None

        init_vals = {
            0: 0.3,
            3: -6.0,
            6: -6.0
        }

        for k, v in init_vals.items():
            self.params[k][:,0] = v

        return self.params

    def updateParams(self, grads, subset=None):
        # if (len(self.params) != len(grads)):
            # raise "wrong len of param(%d) and grad(%d)" % (len(self.params), len(grads))

        if subset == None:
            # print ('SGD')
            for i, g in enumerate(grads):
                if i == len(grads) - 1:
                    rate = 1
                else:
                    rate = 0.1


    # ExposureFilter, #1e-1
    # GammaFilter, # 1e-1
    # ImprovedWhiteBalanceFilter, #1e-1
    # SaturationPlusFilter, #1e-1
    # ToneFilter, #1e0
    # ContrastFilter, #1e-1
    # WNBFilter, #1e0
    # ColorFilter, #1e1

                lrs = [
                    1e-2,
                    1e-3, #
                    1e-3,
                    1e-2,
                    1e-2,
                    1e-2,
                    1e-3,
                    1e1,
                ]

                # ilr = g_lr


                ng =  np.linalg.norm(g)

                if ng > 1e-8:
                    self.params[i] -= rate * lrs[i] * g / ng
        else:
            print ('momentum')
            for i in subset:
                v = self.lr * grads[i]
                if self.lastv != None:
                    v = v + 0.9 * self.lastv
                self.lastv = v
                self.params[i] -= v

        return self.params

    def model(self):
        cimg = self.input
        img_size = self.img_size

        for _filter in self.cfg.filters:
            _ = tf.Variable(img_size)
            fins = _filter(_, self.cfg)
            self.filters.append(fins) # instance of filter
            pn = fins.get_num_filter_parameters()
            # initial param value
            # param holder
            param_holder = tf.placeholder(tf.float32, (None, pn), name='param_holder')
            self.param_holders.append(param_holder)

            realparam = fins.filter_param_regressor(param_holder)
            oimg = fins.process(cimg, realparam)
            cimg = oimg
            self.outputs.append(oimg)

        return self.outputs
