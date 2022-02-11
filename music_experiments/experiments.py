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

concepts_path = os.path.join(concepts_path, "npy")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda:0"  ## TODO: import or configure (also used in trainer)


seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True


model_loc = (
    "/share/cp/projects/concept_composers/experiments/kim2020/training/2112120930/model"
)
model_name = "resnet50_valloss_1.6795074939727783_acc_0.870350090595109.pt"

model = resnet50(in_channels=2, num_classes=13)
checkpoint = torch.load(os.path.join(model_loc, model_name), map_location=device)
state_dict = {
    k.replace("module.", ""): checkpoint["model.state_dict"][k]
    for k in checkpoint["model.state_dict"].keys()
}

model.load_state_dict(state_dict)
model.to(device)
model.eval()

batch_size = 10


train_dataset = MIDIDataset(
    train=True,  # newly added
    txt_file=os.path.join(splits_root, "train.txt"),
    classes=13,
    omit=None,  # str
    seg_num=3,
    age=False,
    transform=None,
    transpose_rng=None,
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

import ICE.ModelWrapper
import ICE.Explainer
import ICE.utils
import shutil


# a solution to have datasets only for a specific class and to return only X instead of the dictionary with X and Y
class XSubset(torch.utils.data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.dataset[[self.indices[i] for i in idx]]["X"]]
        return [self.dataset[self.indices[idx]]["X"]]

    def __len__(self):
        return len(self.indices)


target_classes = [6, 12]
classes_names = [str(i) for i in target_classes]
n_components = 7

Y = np.array(train_dataset.y)
loaders = []
datasets = []
for target in target_classes:
    tdataset = XSubset(train_dataset, np.nonzero(Y == target)[0])
    datasets.append(tdataset)
    loaders.append(
        torch.utils.data.DataLoader(
            tdataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
    )

print(
    "-------------------------------------------------------------------------------------------"
)

wm = ICE.ModelWrapper.PytorchModelWrapper(
    model,
    batch_size=batch_size,
    predict_target=target_classes,
    input_size=[2, 400, 128],
    input_channel_first=True,
    model_channel_first=True,
)

# for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
for layer_name in ["layer4"]:

    title = (
        "ConcComp_{}_{}_[".format(layer_name, n_components)
        + "_".join(classes_names)
        + "]"
    )

    print("title:{}".format(title))
    print("target_classes:{}".format(target_classes))
    print("classes_names:{}".format(classes_names))
    print("n_components:{}".format(n_components))
    print("layer_name:{}".format(layer_name))

    try:  # delete old files
        shutil.rmtree("Explainers/" + title)
    except:
        pass
    # create an Explainer
    Exp = ICE.Explainer.Explainer(
        title=title,
        layer_name=layer_name,
        class_names=classes_names,
        utils=ICE.utils.img_utils(
            img_size=(400, 128), nchannels=2, img_format="channels_first"
        ),
        n_components=n_components,
        reducer_type="NMF",
    )
    # train reducer based on target classes
    Exp.train_model(wm, loaders)
    # generate features
    Exp.generate_features(wm, loaders)
    # generate global explanations
    Exp.global_explanations()
    # generate midi of global explanation
    Exp._sonify_features()
    # save the explainer, use load() to load it with the same title
    Exp.save()
