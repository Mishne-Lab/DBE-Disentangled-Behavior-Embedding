import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.options import *
from utils.videos import *
from utils.criterions import *
from models.nets import DBE

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
    train_set = BehavioralVideo(args.frames_per_clip, frame_rate=args.frame_rate, transform=transform, mod='train', rand_crop=True, jitter=True, return_first_frame=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # model building
    if not args.resume:
        with open(os.path.join(args.config_dir, "{}_model_config.json".format(args.model)), 'r') as f:
            model_config = json.load(f)
        model_config["frames_per_clip"] = args.frames_per_clip
        model_config["criterion"] = args.criterion
        model_config["annealing"] = args.anneal
        model_config["first_frame"] = True
        # reproductibility
        with open(os.path.join(save_dir, "{}_model_config.json".format(args.model)), 'w') as f:
            json.dump(model_config, f)
    else:
        # sys.path.append(save_dir)
        with open(os.path.join(save_dir, '{}_model_config.json'.format(args.model)), 'r') as f:
            model_config = json.load(f)
    
    print('Model: {}'.format(model_config["model"]))
    model = DBE(model_config)

    # criterion
    criterion = globals()[model_config["criterion"]]

    # optimizer initialization
    optimizer = optim.Adam(model.parameters(), lr=args.lr) if args.rnn_lr == 0 else optim.Adam([{'params': [p for n,p in model.named_parameters() if 'rnn' not in n]}, 
                                                                                                {'params': model.rnn.parameters(), 'lr': args.rnn_lr}], 
                                                                                                lr=args.lr) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)
    meter = deque(maxlen=25)
    klg_meter = deque(maxlen=25)
    kls_meter = deque(maxlen=25)
    klc_meter = deque(maxlen=25)
    beta = model_config["beta"]
    print('Learning rate: {}'.format(args.lr))
    print('Batch size: {}'.format(args.batch_size))
    print('Beta: {}'.format(beta))

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
    print('device: ', device)

    # model training
    print('start training...')
    n_past, n_future = model_config["n_past"], args.frames_per_clip-model_config["n_past"]
    for epoch in range(start_epoch, start_epoch+args.epochs):

        print('epoch: ', epoch)
        model.train()

        for front, side in train_loader:

            front, side = front.to(device), side.to(device)
            input1, input2 = front, side

            front, side = front[:, 1:], side[:, 1:]
            (output1, output2), (mu, logvar), (mu_s0, logvar_s0), probs = model(input1, input2, n_past=n_past, n_future=n_future)

            loss = criterion(output1, front) + criterion(output2, side)
            meter.append(loss.item())
            klg_loss = kl(mu, logvar)
            klg_meter.append(klg_loss.item())
            kls_loss = kl(mu_s0, logvar_s0)
            kls_meter.append(kls_loss.item())
            if model_config["independent_cluster"]:
                prob1 = probs[0]
                klc_loss = kl_cat(prob1)
                klc_meter.append(klc_loss.item())
            else:
                (prob1, prob2) = probs # prob2 break
                klc_loss = kl_cat_general_log(prob1, prob2)
                klc_meter.append(klc_loss.item())

            if args.anneal > 0:
                anneal = beta_annealing(epoch, args.anneal)
                loss += (klg_loss * beta[0] + kls_loss * beta[1] + klc_loss * beta[2]) * anneal
            else:    
                loss += klg_loss * beta[0] + kls_loss * beta[1] + klc_loss * beta[2]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('train loss: {}, kl loss: {}, kls loss: {}, klc loss: {}'.format(sum(meter)/len(meter), sum(klg_meter)/len(klg_meter), sum(kls_meter)/len(kls_meter), sum(klc_meter)/len(klc_meter)))

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
                    'loss': validate_error,
                    }, '{}/checkpoint.pth'.format(save_dir))

    file.close()


if __name__ == "__main__":

    args = parse_args()
    print('exp: ', args.name)
    train(args)