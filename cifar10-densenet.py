#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os
import tensorpack as tp

# from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

"""
CIFAR10 DenseNet example. See: http://arxiv.org/abs/1608.06993
Code is developed based on Yuxin Wu's ResNet implementation: https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet
Results using DenseNet (L=40, K=12) on Cifar10 with data augmentation: ~5.77% test error.

Running time:
On one TITAN X GPU (CUDA 7.5 and cudnn 5.1), the code should run ~5iters/s on a batch size 64.
"""

BATCH_SIZE = 64

class Model(tp.ModelDesc):
    def __init__(self, depth):
        super(Model, self).__init__()
        self.N = int((depth - 4)  / 3)
        self.growthRate =12

    def _get_input_vars(self):
        return [tp.InputVar(tf.float32, [None, 32, 32, 3], 'input'),
                tp.InputVar(tf.int32, [None], 'label')
               ]

    def _build_graph(self, input_vars):
        image, label = input_vars
        image = image / 128.0 - 1
        lays = []

        def conv(name, l, channel, stride):
            return tp.Conv2D(name, l, channel, 3, stride=stride,
                          nl=tf.identity, use_bias=False,
                          W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/channel)))
        def add_layer(name, l):
            print("l")
            print(l)
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                c = tp.BatchNorm('bn1', l)
                c = tf.nn.relu(c)
                c = conv('conv1', c, self.growthRate, 1)
            return c

        def add_transition(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                l = tp.BatchNorm('bn1', l)
                l = tf.nn.relu(l)
                l = tp.Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
                l = tp.AvgPooling('pool', l, 2)
            return l

        def get_inputs(lvl):
            num = (int(np.log(len(lays))/np.log(2)))
            print("num",num)
            print("len", len(lays))
            logIdx = [-(2**i) for i in range(num+1)]
            layLst = [lays[i] for i in logIdx]
            ret = []
            if lvl == 0:
                ret =  [k[0] for k in layLst]
            elif lvl == 1:
                ret = []
                for j,k in enumerate(layLst):
                    if len(k) == 2:
                        ret.append(k[1])
                    else:
                        name = "transitionLvl_1_num_"+str(j)
                        k.append(add_transition(name, k[0]))
                        ret.append(k[1])
            else:
                ret = []
                for j,k in enumerate(layLst):
                    if len(k) == 3:
                        ret.append(k[2])
                    elif len(k) == 2:
                        name = "transitionLvl_2_num"+str(j)
                        k.append(add_transition(name, k[1]))
                        ret.append(k[2])
                    elif len(k) == 1:
                        name = "transitionLvl_1_num_"+str(j)
                        k.append(add_transition(name, k[0]))
                        name = "transitionLvl_2_num"+str(j)
                        k.append(add_transition(name, k[1]))
                        ret.append(k[2])
            return tf.concat(3, ret)


        def dense_net(name):
            l = conv('conv0', image, 16, 1)
            lays.append(l)
            with tf.variable_scope('block1') as scope:

                for i in range(self.N):
                    inp = get_inputs(0)
                    l = add_layer('dense_layer.{}'.format(i), inp)
                    lays.append([l])
                inp = get_inputs(1)
                l = add_transition('transition1', inp)
                lays.append([None,l])

            with tf.variable_scope('block2') as scope:

                for i in range(self.N):
                    inp = get_inputs(1)
                    l = add_layer('dense_layer.{}'.format(i), inp)
                    lays.append([None,l])
                inp = get_inputs(2)
                l = add_transition('transition2', inp)
                lays.append([None,None,l])

            with tf.variable_scope('block3') as scope:

                for i in range(self.N):
                    inp = get_inputs(2)
                    l = add_layer('dense_layer.{}'.format(i), inp)
                    lays.append([None,None,l])

            inp = get_inputs(2)
            l = tp.BatchNorm('bnlast', linp)
            l = tf.nn.relu(l)
            l = tp.GlobalAvgPooling('gap', l)
            logits = tp.FullyConnected('linear', l, out_dim=10, nl=tf.identity)

            return logits
        print("lays")
        print(lays)
        logits = dense_net("dense_net")
        print(lays)

        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label)
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W
        wd_cost = tf.mul(1e-4, tp.regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary([('.*/W', ['histogram'])])   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = tp.dataset.Cifar10(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            tp.imgaug.CenterPaste((40, 40)),
            tp.imgaug.RandomCrop((32, 32)),
            tp.imgaug.Flip(horiz=True),
            #tp.imgaug.Brightness(20),
            #tp.imgaug.Contrast((0.6,1.4)),
            tp.imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            tp.imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = tp.AugmentImageComponent(ds, augmentors)
    ds = tp.BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = tp.PrefetchData(ds, 3, 2)
    return ds

def get_config():
    log_dir = 'train_log/cifar10-single-fisrt%s-second%s-max%s' % (str(args.drop_1), str(args.drop_2), str(args.max_epoch))
    tp.logger.set_logger_dir(log_dir, action='n')

    # prepare dataset
    dataset_train = get_data('train')
    step_per_epoch = dataset_train.size()
    dataset_test = get_data('test')

    sess_config = tp.get_default_sess_config(0.9)

    tp.get_global_step_var()
    lr = tf.Variable(0.1, trainable=False, name='learning_rate')
    tf.summary.scalar('learning_rate', lr)

    return tp.TrainConfig(
        dataflow=dataset_train,
        optimizer=tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
        callbacks=tp.Callbacks([
            tp.StatPrinter(),
            tp.ModelSaver(),
            tp.InferenceRunner(dataset_test,
                [tp.ScalarStats('cost'), tp.ClassificationError()]),
            tp.ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001)])
        ]),
        session_config=sess_config,
        model=Model(depth=args.depth),
        step_per_epoch=step_per_epoch,
        max_epoch=args.max_epoch,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    parser.add_argument('--drop_1',default=150, help='Epoch to drop learning rate to 0.01.') # nargs='*' in multi mode
    parser.add_argument('--drop_2',default=225,help='Epoch to drop learning rate to 0.001')
    parser.add_argument('--depth',default=40, help='The depth of densenet')
    parser.add_argument('--max_epoch',default=300,help='max epoch')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    # tp.SyncMultiGPUTrainer(config).train()
    tp.SimpleTrainer(config).train()
