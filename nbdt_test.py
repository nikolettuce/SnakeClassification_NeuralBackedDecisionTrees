import sys
sys.path.insert(0, "src/util")
sys.path.insert(0, "src/model")

# imports
import torch
import torchvision
import os
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score
from PIL import Image

from nbdt.model import SoftNBDT
from nbdt.loss import SoftTreeSupLoss

from wn_utils import *
from graph import *
from dir_grab import *

from hierarchy import *


model = models.densenet121(pretrained=True)
# set features from classes, in this case 45
model.classifier = nn.Linear(model.classifier.in_features, 45)
input_size = 224 # densenet characteristic


# load model from teams folder
model_weights_path = "../teams/DSC180A_FA20_A00/a01group09/model_states/baseline_model.pt"
model_weights = torch.load(model_weights_path)
model.load_state_dict(model_weights)

#generate_hierarchy(dataset='snakes', arch='densenet121', model=model, method='induced')

# test hierarchy based on path
#test_hierarchy('snakes',"./data/hierarchies/snakes/graph-induced-densenet121.json")

generate_hierarchy_vis("./data/hierarchies/snakes/graph-induced-densenet121.json", 'snakes')