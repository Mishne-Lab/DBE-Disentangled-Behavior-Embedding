# DBE: Disentangled-Behavior-Embedding

Official implementation of [Learning Disentangled Behavior Embeddings (NeurIPS 2021)](https://openreview.net/forum?id=ThbM9_6DNU).

## Environment requirement
The whole experiment process is based on PyTorch:
- PyTorch 1.3.0
- Torchvision 0.4.1

Note this is neither the minimum updated requirement nor the latest version.
Other versions may just work as well.

## Create video dataset
1. Add train and test videos directories to "./configs/split.json". 
2. Note that the videos are assumed to be multiview with 2 views concatenated horizontally.

## Train your models
There are multiple training scripts in this repo as we are trying different models.
1. The model parameters are stored in "./configs/model_name_model_configs.json". To change the architecture, edit the json file. The model config file will saved for each running together with training hyperparameters.
2. Video frames are by defaulted resized to 128 by 128. Use argparse to specify other data related setting, e.g. frame rate, frame per clip, crop range etc.
3. Run bash command to train a DBE model. An example: 

    ```bash
    python3 train_dbe.py -n name_of_exp -u id_of_gpu -l recon -bs num_of_batch_size -ep num_of_epochs --lr num_of_lr -fpc frame_per_clip
    ```
    The choice of these parameters depends on your computing power. The batch size is recommanded to be larger than 8. The suitable number of epochs depends on the size of the dataset. 
4. Results will be saved in "outputs/name_of_exp".

## Evaluate your models
1. Evaluate the trained model by specifying the name of experiment. Note that the time of the experiment being created is added as the prefix of the experiment name before training.

    ```bash
    python3 evaluate.py -u id_of_gpu -bs num_of_batch_size -n time-name_of_exp -fpc frame_per_clip -md test
    ```
2. The latent embeddings will be saved in the experiment directory. For DBE model, behavioral state estimation will also be saved.
