import torch
import torchvision
from torchvision import io
from torchvision.datasets.video_utils import VideoClips
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset, DataLoader

# from utils import *
from utils.transforms import *

import os
import time
import json
import pickle
import h5py
import random
import cv2
from operator import itemgetter
from PIL import Image
from glob import glob
# from tqdm import tqdm


class BehavioralVideo(Dataset):
    '''
    new version dataset
    video data in /data2/BehaveLabeling/_name
    e.g. /data2/BehaveLabeling/PT3/EPC_front_PT3_2018-02-20_025.mp4
    '''
    anchor = 600
    # interval = [600, 1800]
    # interval = [750, 1100]
    # interval = [780, 1080]
    # interval = [850, 1050]
    interval = [750, 1150]
    with open("./configs/split.json", 'r') as f:
        split = json.load(f)
    with open("./configs/split_new.json", 'r') as f:
        split2 = json.load(f)

    def __init__(self, configs, frames_per_clip, frame_rate=4, view='comb', transform=None, semi=False, opf=False, motion=False, mod='train', rand_crop=False, jitter=True, return_first_frame=False, holdout=True):
        '''
        frames_per_clip: not used
        config: a list of session configs
        '''
        print('frame range: ', self.interval)
        self.mod = mod
        self.step = frame_rate
        self.fpc = frames_per_clip
        self.view = view
        self.semi = semi
        self.opf = opf
        self.motion = motion
        self.rand_crop = rand_crop
        self.jitter = jitter
        self.return_first_frame = return_first_frame

        # temporary setting for efficiency consideration
        configs = [config for config in configs if view in config["views"]]
        configs = [config for config in configs if config["resolution"]==[400, 800]]

        # data split
        self.configs = [config for config in configs if "{}-{}".format(config["name"], config["date"]) in self.split[mod]]

        # construct labels
        for k, config in enumerate(self.configs):
            config["session"] = k

        print('Total {} trials: '.format(mod), sum([config["trials"] for config in self.configs])) # 1241 in total

        # self-defined
        self.trials, self.ids = [], [] # id is the mice + session
        for config in self.configs:
            session_dir = os.path.join(config["root"], "{}-{}".format(config["name"], config["date"]))
            trials_of_session = [x[0] for x in os.walk(session_dir)][1:]
            self.trials += sorted(trials_of_session)
            self.ids += [config["session"]]*len(trials_of_session)
            print("Session {}-{} loaded: {}".format(config["name"], config["date"], len(trials_of_session)))

        self.transforms = torchvision.transforms.Compose([transform])
        print(self.trials)

        if semi:
            self.get_bda()

        if holdout:
            # if include holdout videos
            print("exclude holdout videos...")
            self.trials, self.ids = zip(*[(t, i) for t, i in zip(self.trials, self.ids) if t.split('/', 6)[-1] in self.split2[mod]])

    def get_clip(self, trial_dir, rand_crop=False, jitter=False):

        rand = random.randint(-self.step, self.step) if jitter else 0
        if rand_crop:
            start = self.interval[0] + random.randint(0, self.interval[1]-self.interval[0]-self.fpc*self.step)
        else:
            start = self.interval[0]
        start, end = rand+start, rand+start+self.fpc*self.step
        start = start-self.step if self.motion else start
        clip_indices = range(start, end, self.step)

        # t0 = time.time()
        raw_video_flow = sorted(glob('{}/*.jpg'.format(trial_dir)))
        try:
            video_flow = itemgetter(*clip_indices)(raw_video_flow)
        except:
            print('Fail to collect frames of video: ', trial_dir)
        # print(time.time()-t0)
        # t0 = time.time()
        # clip = [to_tensor(Image.open(frame))[0:1, :, :] for frame in video_flow] # kind of slow 1586392418.11
        if self.opf:
            clip = [cv2.imread(frame, 0) for frame in video_flow]
            clip = [to_tensor(cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)) for prvs, next in zip(clip, clip[1:]+clip[-1:])]
        else:
            clip = [to_tensor(cv2.imread(frame, 0)) for frame in video_flow] # t+1 frame for substraction for motion only
        # print(time.time()-t0)
        clip = torch.stack(clip, dim=0)
        
        if self.return_first_frame:
            first_frame_jitter = random.randint(-15, 15) if jitter else 0
            frame = '{}/img_{:04}.jpg'.format(trial_dir, self.anchor+first_frame_jitter)
            first_frame = to_tensor(cv2.imread(frame, 0))
            clip = torch.cat([first_frame.unsqueeze(dim=0), clip], dim=0)

        if self.semi:
            states = self.get_state(trial_dir, start, clip_indices)
            return clip, states
        return clip

    def __len__(self):
        # return self.video_clips.num_clips()
        return len(self.trials)

    def __getitem__(self, idx):

        # video, audio, info, video_idx = self.video_clips.get_clip(idx)
        # return video
        if self.semi:
            clip, states = self.get_clip(self.trials[idx], rand_crop=self.rand_crop, jitter=self.jitter)
        else:
            clip = self.get_clip(self.trials[idx], rand_crop=self.rand_crop, jitter=self.jitter)
        front, side = clip[:, :, :, 400:], clip[:, :, :, :400]

        if self.transforms is not None:
            front, side = self.transforms(front), self.transforms(side)

        # if self.return_first_frame:
        #     front, side, first_front, first_side = front[1:], side[1:], front[:1], side[:1]

        trial_name = '/'.join(self.trials[idx].rsplit('/', 2)[1:])

        if self.semi:
            # if self.return_first_frame:
            #     return (front, side), (first_front, first_side), self.ids[idx], [states]
            return (front, side), self.ids[idx], [states]
        else:
            # if self.return_first_frame:
            #     return (front, side), (first_front, first_side)
            return (front, side), self.ids[idx]
    
    def get_bda(self, bda_dir='/data2/changhao/Dataset/Behavioral-Videos/Analysis'):

        with open(os.path.join(bda_dir, 'reformed_bda.pkl'), 'rb') as f:
            bda = pickle.load(f)
        self.states = {session_name: s for session_name, s in bda.items() if session_name in self.split[self.mod]}

        # summerize all possible states
        self.all_states = {}
        for session_name, s in self.states.items():
            for vid_name, vid in s.items():
                if len(vid) > 0:
                    for i, (state, occur, timeind) in enumerate(vid):
                        if state in ['Tone', 'success', 'failure', 'success+1', 'failure+1']: # exampt 3 states
                            continue
                        if state not in self.all_states.keys():
                            self.all_states[state] = timeind[1] - timeind[0]
                        else:
                            self.all_states[state] += timeind[1] - timeind[0]
                        # self.states[session_name][vid_name][i][-1] = (timeind[0]/self.fps, timeind[1]/self.fps)
                # all_states += np.asarray(vid, dtype=object)[:, 0].tolist()
        # self.all_states = set(all_states)
        print('All possible states: ', self.all_states)

        # manually set mapping
        self.state_map = {'TableTurn': 0, 'Lift':1, 'LiftLeft':1, 'LiftRight':1, 'Grab':2, 'AtMouth':3, 'AtMouthNew':3, 'AtMouthNoPellet':3, 'BackToPerch':4, 'Chew':5}
        print('State mapping: ', self.state_map)

    def get_state(self, trial_dir, start=None, clip_indices=None):
        
        if start is None or clip_indices is None:
            start, end = self.interval[0], self.interval[0]+self.fpc*self.step
            clip_indices = range(start, end, self.step)
        states = {}
        session, idx = trial_dir.rsplit('/', 2)[1:]
        for state, _, (state_start, state_end) in self.states[session][idx]:
            if state in ['Tone', 'success', 'failure', 'success+1', 'failure+1']: # exampt 3 states
                continue
            if self.state_map[state] not in states.keys():
                states[self.state_map[state]] = []
            state_indices = range(state_start, state_end)
            states[self.state_map[state]] += [(i-start)//self.step for i in state_indices if i in clip_indices]

        return states

    def filter_trials(self):
        # only keep videos that have state labels
        filtered_trials, fitlered_ids = [], []
        for t, i in zip(self.trials, self.ids):
            if self.get_state(t):
                filtered_trials.append(t)
                fitlered_ids.append(i)
        self.trials, self.ids = filtered_trials, fitlered_ids


class WFCI(Dataset):

    interval = [5, 185]
    def __init__(self, configs, frames_per_clip, frame_rate=4, view='comb', transform=None, mod='train', rand_crop=False, jitter=True, return_first_frame=False, use_holdout=False):
        self.mod = mod
        self.step = frame_rate
        self.fpc = frames_per_clip
        self.view = view
        self.rand_crop = rand_crop
        self.jitter = jitter
        self.transform = transform
        self.return_first_frame = return_first_frame

        # temporary setting for efficiency consideration
        configs = [config for config in configs if view in config["views"]]
        if not use_holdout:
            configs = [config for config in configs if config["holdout"] is False]
        else:
            configs = [config for config in configs if config["holdout"] is True]
        self.configs = [config for config in configs if config["resolution"]==[128, 128]]

        # construct labels
        for k, config in enumerate(self.configs):
            config["session"] = k

        print('Total {} trials: '.format(mod), sum([config["trials"] for config in self.configs])) # 1241 in total

        # self-defined
        self.trials, self.ids = [], [] # id is the mice + session
        for config in self.configs:
            session = "{}_{}".format(config["name"], config["date"])
            with h5py.File('{}/musall_vistrained_{}_data.hdf5'.format(config["root"], session), "r", libver='latest', swmr=True) as f:
                trials_of_session = [(config["root"], session, trial) for trial in f['images'].keys()]
                self.trials += trials_of_session
            self.ids += [config["session"]]*len(trials_of_session)
            print("Session {}-{} loaded: {}".format(config["name"], config["date"], len(trials_of_session)))

        self.transforms = torchvision.transforms.Compose([transform])

        # data split
        if not use_holdout:
            if os.path.exists("./configs/wfci_split.json"):
                with open("./configs/wfci_split.json", 'r') as f:
                    split = json.load(f)
            else:
                n_trials = len(self.trials)
                perm = np.random.permutation(n_trials).tolist()
                split = {'train': perm[:round(0.8*n_trials)], 'validate': perm[round(0.8*n_trials):round(0.9*n_trials)], 'test': perm[round(0.9*n_trials):]}
                with open("./configs/wfci_split.json", 'w') as f:
                    json.dump(split, f)
            self.trials, self.ids = np.asarray(self.trials)[split[mod]], np.asarray(self.ids)[split[mod]]
        
    def __len__(self):
        return len(self.trials)

    def get_clip(self, trial_dir, rand_crop=False, jitter=False):
        
        rand = random.randint(-self.step, self.step) if jitter else 0
        if rand_crop:
            start = self.interval[0] + random.randint(0, self.interval[1]-self.interval[0]-self.fpc*self.step)
        else:
            start = self.interval[0]
        start, end = rand+start, rand+start+self.fpc*self.step
        clip_indices = range(start, end, self.step)

        root, session, trial = trial_dir
        # # old implementation
        # with h5py.File('{}/data.hdf5'.format(session_dir), "r", libver='latest', swmr=True) as f:
        #     clip = torch.from_numpy(f['images'][trial][:].astype(float) / 255.)
        with h5py.File('{}/musall_vistrained_{}_data.hdf5'.format(root, session), "r", libver='latest', swmr=True) as f:
            clip = torch.from_numpy(f['images'][trial][:].astype(float) / 255.)

        if 'mSM36' not in session:
            clip[:, :1] = torch.rot90(clip[:, :1], k=3, dims=(2,3))

        if self.return_first_frame:
            first_frame_jitter = random.randint(0, 10) if jitter else 0
            first_frame = clip[0+first_frame_jitter]
            clip = torch.cat([first_frame.unsqueeze(dim=0), clip[clip_indices]], dim=0)
        else:
            clip = clip[clip_indices]
        return clip.float()

    def __getitem__(self, idx):
        clip = self.get_clip(self.trials[idx], rand_crop=self.rand_crop, jitter=self.jitter)
        front, side = clip[:, :1], clip[:, 1:]

        if self.transform is not None:
            front, side = self.transforms(front), self.transforms(side)

        return (front, side), self.ids[idx]