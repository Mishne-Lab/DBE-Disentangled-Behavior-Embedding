import torch
import numpy as np
import os
import cv2
import pickle
import datetime
from tqdm import tqdm


def save_file(buffer, file_dir):
    buffer = torch.cat(buffer, dim=0).cpu().numpy()
    with open(file_dir, 'wb') as f:
        pickle.dump(buffer, f)

def convert_date(date_str):
    return datetime.date.strftime(datetime.datetime.strptime(date_str, '%Y-%m-%d'), "%b%d")

def generate_labels(gt, classes, full_length=2400):
    labels = np.zeros((len(gt), full_length), dtype=int)
    for i, (n, t) in enumerate(gt.items()):
        for j, k in enumerate(classes):
            if k in t.keys():
                labels[i, t[k]] = j+1
    return labels

def get_states(all_states, target_trials, gt):
    # get the state estimation of the labeled video frames
    all_states = all_states.argmax(2).copy()
    s = []
    trial_nums = [k.rsplit('/', 1)[1] for k in target_trials]
    for j, (k, v) in enumerate(gt.items()):
        idx = k.rsplit('_', 1)[1]
        if idx in trial_nums:
            s.append(all_states[trial_nums.index(idx)])
    return np.stack(s)

def generate_mapping(states, labels, n_states=None, n_labels=None):
    # generate a mapping from the state estimation to ground truth labeling
    assert states.shape==labels.shape
    mapping = []
    if n_states is None:
        n_states = states.max()+1
    if n_labels is None:
        n_labels = labels.max()+1
    for i in range(n_states):
        tar, max_count = 0, 0
        for j in range(1, n_labels):
            count = labels[(labels==j)&(states==i)].sum()
            if count > max_count:
                max_count, tar = count, j
        mapping.append(tar)
    states_map = states.copy()
    for i, c in enumerate(mapping):
        states_map[states==i] = c
    return states_map, mapping

def gather_motifs(trials, states, tar, start=750, resolution=400, win=0, video_dir='/data2/changhao/Dataset/Behavioral-Videos/Videos/'):
    images = []
    info = []
    for trial, indices in tqdm(zip(trials, states.argmax(-1))):
        img = np.zeros((resolution, resolution*2, 3))
        for k in range(1, indices.shape[0]-1):
            if indices[k-1] != tar and indices[k] == tar:
                k0 = k
                i0 = cv2.imread('{}/img_{:04d}.jpg'.format(os.path.join(video_dir, trial), start+(k+1)+win//2), 0)
                img[:, :, 1] = i0
                img[:, :, 2] = i0
            if indices[k-1] == tar and indices[k] != tar:
                if k - k0 <=3:
                    img = np.zeros((resolution, resolution*2, 3))
                    continue
                i1 = cv2.imread('{}/img_{:04d}.jpg'.format(os.path.join(video_dir, trial), start+(k+1)+win//2), 0)
                img[:, :, 0] = i1
                images.append(img)
                info.append([trial, k0, k])
            if img[:, :, 0].sum() > 0:
                img = np.zeros((resolution, resolution*2, 3))
    return images, info

def dlc_regression(latents, trials, dlcs, dlc_trials, markers, session):
    assert len(latents)==len(trials)
    dlc_trial_nums = [n.rsplit('_', 1)[1] for n in dlc_trials]
    dlc_indices = [dlc_trial_nums.index(t.rsplit('/', 1)[1]) for t in trials if session in t]
    dlc_session = dlcs[dlc_indices]
    X = np.stack([l for l, t in zip(list(latents), trials) if session in t], axis=0)
    Y = dlc_session[:, 750:1150]
    score, dlc_reg = [], []
    from sklearn.linear_model import LinearRegression
    for i in range(len(markers)):
        mask = (Y[:, :, 3*i+2].flatten()>0.95)
        x = X.reshape((-1, X.shape[-1]))[mask]
        y = Y[:, :, 3*i:3*i+2].reshape((-1, 2))[mask]
        reg = LinearRegression().fit(x, y)
        y_hat = reg.predict(X.reshape((-1, X.shape[-1]))).reshape((*X.shape[:2], 2))
        score.append(reg.score(x, y))
        dlc_reg.append(y_hat)
    return dlc_reg, score