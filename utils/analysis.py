import torch
import pickle


def save_file(buffer, file_dir):
    buffer = torch.cat(buffer, dim=0).cpu().numpy()
    with open(file_dir, 'wb') as f:
        pickle.dump(buffer, f)