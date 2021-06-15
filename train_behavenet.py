import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.options import *
from utils.videos import *
from utils.criterions import *
from models.nets import BehaveNet

import os
import sys
import json
from collections import deque
from datetime import datetime

def train(args):

    torch.autograd.set_detect_anomaly(True)
    
    # path initialization
    if not args.resume:
        time = datetime.now()
        if not args.name:
            save_dir = os.path.join(args.save_dir, time.strftime('%Y-%m-%d-%H:%M:%S'))
        else:
            save_dir = os.path.join(args.save_dir, '{}-{}'.format(time.strftime('%Y-%m-%d-%H:%M:%S'), args.name))

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    else:
        save_dir = os.path.join(args.save_dir, args.name)
        print('save dir: ', save_dir)
        
    # redirect output
    file = open(os.path.join(save_dir, 'train_log.txt'), 'a')
    sys.stdout = file

    # data loading
    transform = ResizeVideo(args.frame_size)
    train_set = BehavioralVideo(args.frames_per_clip, frame_rate=args.frame_rate, interval=[args.start, args.end], transform=transform, mod='train', rand_crop=True, jitter=True, return_first_frame=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    
    # model building
    if not args.resume:

        with open(os.path.join(args.config_dir, 'BehaveNet_model_config.json'), 'r') as f:
            model_config = json.load(f)
        model_config["frames_per_clip"] = args.frames_per_clip
        model_config["criterion"] = args.criterion
        model_config["first_frame"] = False
        # reproductibility
        with open(os.path.join(save_dir, 'BehaveNet_model_config.json'), 'w') as f:
            json.dump(model_config, f)
    else:
        with open(os.path.join(save_dir, 'BehaveNet_model_config.json'), 'r') as f:
            model_config = json.load(f)
    
    print('Model: {}'.format(model_config["model"]))
    model = BehaveNet(model_config)

    # criterion
    criterion = globals()[model_config["criterion"]]

    # optimizer initialization
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)
    meter = deque(maxlen=25)
    print('Learning rate: {}'.format(args.lr))
    print('Batch size: {}'.format(args.batch_size))

    # checkpoint loading
    if args.resume:
        checkpoint = torch.load('{}/checkpoint.pth'.format(save_dir), map_location='cpu')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        start_epoch = 0

    # parallelism
    if args.device == 'cuda':
        if args.gpus:
            device_ids = [int(idx) for idx in list(args.gpus)]
            device = '{}:{}'.format(args.device, device_ids[0])
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
        else:
            device = args.device
            model = nn.DataParallel(model).to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    elif args.device == 'cpu':
        device = args.device

    # model training
    print('start training...')
    for epoch in range(start_epoch, start_epoch+args.epochs):

        print('epoch: ', epoch)
        model.train()

        for front, side in train_loader:

            front, side = front.to(device), side.to(device)
            output1, output2 = model(front, side)
            loss = criterion(output1, front) + criterion(output2, side)
            meter.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('train loss: {}'.format(sum(meter)/len(meter)))

        scheduler.step()
        # model validation
        if (epoch+1) % args.validate_freq == 0:   

            print('save model')
            torch.save(model.module.state_dict(), '{}/best_model.pth'.format(save_dir))

            # save checkpoint
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, '{}/checkpoint.pth'.format(save_dir))

    file.close()


if __name__ == "__main__":

    args = parse_args()
    print('exp: ', args.name)
    train(args)