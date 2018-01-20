import argparse
import os
import scipy.misc
import numpy as np
import tensorflow as tf
from model import gan

parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset_dir', dest='dataset_dir', default='celeba', help='path of the dataset')

parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='# images in batch')
parser.add_argument('--ngf', dest='ngf', type=int, default=32, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')

parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')

parser.add_argument('--phase', dest='phase', default='train', help='train')
parser.add_argument('--epoch', dest='epoch', type=int, default=300, help='# of epoch')
parser.add_argument('--from_epoch', dest='from_epoch', type=int, default=0, help='training from epoch #')

parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')

parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=50.0, help='weight on L1 term in objective')
parser.add_argument('--xent_lambda', dest='xent_lambda', type=float, default=50.0, help='weight on xent term in objective')
parser.add_argument('--max_seq_len', dest='max_seq_len', type=int, default=14, help='max length of the text sequence')

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    with tf.Session() as sess:
        model = gan(sess, args)
        model.train(args) if args.phase == 'train' \
            else model.test(args)

if __name__ == '__main__':
    args = parser.parse_args()
    main(None)
