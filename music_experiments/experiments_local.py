import sys

sys.path.append("..")
sys.path.append("/share/home/francesco/concept_composers/DeepComposerClassification/")
sys.path.append("/share/home/francesco/concept_composers/")
sys.path.append("/share/home/francesco/InvertibleCE/")  # invertible ICE repo

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import torchvision
import numpy as np

import os
from pathlib import Path

# from torchsummary import summary

import numpy as np

# from methods.activations import HandleActivations

from DeepComposerClassification.resnet import resnet50
from tcav.tcav_utils import assemble_concept, get_tensor_from_filename
from DeepComposerClassification.generator import Generator
from tools.data_loader import MIDIDataset

from config import concepts_path, splits_root
import partitura

import ICE.ModelWrapper
import ICE.Explainer
import ICE.utils
import shutil

concepts_path = os.path.join(concepts_path, "npy")

device = "cuda:0"  ## TODO: import or configure (also used in trainer)

import random
import pandas as pd

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

model_loc = (
    "/share/cp/projects/concept_composers/experiments/kim2020/training/2202180920/model"
)
model_name = "resnet50_valloss_1.3208494186401367_acc_0.9237896919964558.pt"

model = resnet50(in_channels=2, num_classes=13)
checkpoint = torch.load(os.path.join(model_loc, model_name), map_location=device)
state_dict = {
    k.replace("module.", ""): checkpoint["model.state_dict"][k]
    for k in checkpoint["model.state_dict"].keys()
}

model.load_state_dict(state_dict)
model.to(device)
model.eval()

batch_size = 40


target_composers = [
    "Alexander Scriabin",
    "Claude Debussy",
    "Domenico Scarlatti",
    "Franz Liszt",
    "Franz Schubert",
    "Frédéric Chopin",
    "Johann Sebastian Bach",
    "Johannes Brahms",
    "Joseph Haydn",
    "Ludwig van Beethoven",
    "Robert Schumann",
    "Sergei Rachmaninoff",
    "Wolfgang Amadeus Mozart",
]

target_classes = [5, 6, 9, 12]
classes_names = [target_composers[i] for i in target_classes]
n_components = 6
layer_name = "layer4"
title = (
    "ConcComp_{}_{}_[".format(layer_name, n_components) + "_".join(classes_names) + "]"
)

print("title:{}".format(title))
print("target_classes:{}".format(target_classes))
print("classes_names:{}".format(classes_names))
print("n_components:{}".format(n_components))
print("layer_name:{}".format(layer_name))


model = model.cuda()
model.eval()
wm = ICE.ModelWrapper.PytorchModelWrapper(
    model,
    batch_size=batch_size,
    predict_target=target_classes,
    input_size=[2, 400, 88],
    input_channel_first=True,
    model_channel_first=True,
)

# load the pretrained explainer
Exp = ICE.Explainer.Explainer(title=title)
Exp.load()

# target piece to explain
piece_path = list(
    Path(
        "/share/cp/projects/concept_composers/ConceptDataset/midi/alberti_bass"
    ).iterdir()
)[-1]
generator = Generator()
x = partitura.load_performance_midi(piece_path)
x = generator.generate_segment(x)[:, :400, :]


Exp.local_explanations(x, wm, name="Local_alberti_6")

