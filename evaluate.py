import sys
import pickle
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.options import *
from utils.videos import *
from utils.criterions import *
from utils.analysis import *


def evaluate(args):

    save_dir = os.path.join(args.save_dir, args.name)
    assert os.path.exists(save_dir)
    sys.path.append(save_dir)

    # model building
    saved_model_config = glob('{}/*_model_config.json'.format(save_dir))[0]
    with open(saved_model_config, 'r') as f:
        model_config = json.load(f)
    model_config["frames_per_clip"] = args.frames_per_clip
    use_first_frame = model_config["first_frame"]

    # data loading
    transform = ResizeVideo(args.frame_size)
    mod = args.mode
    eval_set = BehavioralVideo(args.frames_per_clip, frame_rate=args.frame_rate, transform=transform, mod=mod, rand_crop=False, jitter=False, return_first_frame=use_first_frame)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # criterion & device
    device = args.device

    print('Model: {}'.format(model_config["model"]))
    from models.nets import BehaveNet, DisAE, DBE
    best_model = eval(model_config["model"])(model_config)

    best_dict = torch.load('{}/best_model.pth'.format(save_dir), map_location='cpu')
    best_model.load_state_dict(best_dict)

    # read criterion
    criterion = globals()[model_config["criterion"]]

    if args.device == 'cuda':
        if args.gpus:
            device_ids = [int(idx) for idx in list(args.gpus)]
            device = '{}:{}'.format(args.device, device_ids[0])
            best_model = nn.DataParallel(best_model, device_ids=device_ids).to(device)
        else:
            device = args.device
            best_model = nn.DataParallel(best_model).to(device)
    elif args.device == 'cpu':
        device = args.device

    with open('{}/{}_trials.pkl'.format(save_dir, mod), 'wb') as f:
        pickle.dump(eval_set.trials, f)

    best_model.eval()
    
    print('start evaluating...')
    latents, contents, states = [], [], []
    loss_track = []
    with torch.no_grad():
        for front, side in eval_loader:

            front, side = front.to(device), side.to(device)
            if model_config["model"] == 'DBE':
                (output1, output2), _, _, probs = best_model(front, side, n_past=args.n_past, n_future=args.frames_per_clip-args.n_past)
                states.append(probs[0].detach())
            else:
                output1, output2 = best_model(front, side)

            if use_first_frame:
                loss = criterion(output1, front[:, 1:]) + criterion(output2, side[:, 1:])
            else:
                loss = criterion(output1, front) + criterion(output2, side)
            loss_track.append(loss.item())

            latents.append(best_model.module.latent.detach())

        print('Saving latent embedding...')
        save_file(latents, file_dir='{}/{}_latents.pkl'.format(save_dir, mod))
        if len(states) > 0:
            save_file(states, file_dir='{}/{}_states.pkl'.format(save_dir, mod))

        print('loss: ', torch.mean(torch.FloatTensor(loss_track)))


if __name__ == "__main__":

    args = parse_args()
    evaluate(args)