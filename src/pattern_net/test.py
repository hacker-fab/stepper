import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import pattern_net
import numpy as np
from datasets import AugmentedDataset, visualize_transformation
import os
import time


if __name__ == '__main__':
    # test inference time of trained model
    model_folder = "/home/louis/project/stepper/src/pattern_net/model"
    model = pattern_net()
    model.load_state_dict(torch.load(os.path.join(model_folder, 'pattern_net_model.pth')))
    start = time.time()
    for i in range(100):
        result = model(torch.randn(1, 3, 224, 224))
    end = time.time()
    total_time = end - start
    print(f"Total inference time for 100 samples: {total_time:.4f} seconds")
    print(f"Avg inference time per sample: {total_time / 100:.4f} seconds")