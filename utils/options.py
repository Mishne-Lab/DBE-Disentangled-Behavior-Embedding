
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch Video Compression Training')

    parser.add_argument('-cf', '--config-dir', default='../DBE-Disentangled-Behavior-Embedding/configs', help='config dir')
    parser.add_argument('-sd', '--save-dir', default='../DBE-Disentangled-Behavior-Embedding/outputs', help='path to save')
    parser.add_argument('-n', '--name', default='', help='folder in save path')
    parser.add_argument('-r', '--resume', dest="resume", help='resume from checkpoint', action="store_true")
    parser.add_argument('-md', '--mode', default='test', help='test mode')

    parser.add_argument('--rnn_layer', default=1, type=int, metavar='N', help='hidden layer of rnn')
    parser.add_argument('--rnn_hidden', default=8, type=int, metavar='N', help='hidden dim of rnn')
    parser.add_argument('-l', '--criterion', default='recon', help='training criterion')

    parser.add_argument('-fpc', '--frames_per_clip', default=50, type=int, metavar='N', help='# of frames per clip')
    parser.add_argument('-fps', '--frame_rate', default=1, type=int, metavar='N', help='# of frames per second')
    parser.add_argument('-fs', '--frame_size', default=128, type=int, metavar='N', help='# of frames dimensions')
    parser.add_argument('-np', '--n_past', default=10, type=int, metavar='N', help='# of past frames to infer z0 (DBE only)')

    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-u', '--gpus', default='', help='index of specified gpus')
    parser.add_argument('-bs', '--batch-size', default=8, type=int)
    parser.add_argument('-ep', '--epochs', default=50, type=int, metavar='N', help='# of total epochs')
    parser.add_argument('-wp', '--warmups', default=50, type=int, metavar='N', help='# of total warm up epochs')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='# of data loaders (default: 16)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate (default: 0.001)')
    parser.add_argument('--rnn_lr', default=0, type=float, help='initial learning rate of rnn if necessary (default: 0.001)')
    parser.add_argument('--lr-step', default=10000, type=int, help='decrease lr every these iterations')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--anneal', default=0, type=float, help='turning point for linear annealing')
    parser.add_argument('--alpha', default=1.0, type=float, help='ratio of rnn loss')
    parser.add_argument('--validate-freq', default=1, type=int, help='validation frequency')

    args = parser.parse_args()

    return args