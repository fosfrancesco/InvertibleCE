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
import click
import json

import os
from pathlib import Path

# from torchsummary import summary

import numpy as np

from DeepComposerClassification.resnet import resnet50
from tcav.tcav_utils import assemble_concept, get_tensor_from_filename
from DeepComposerClassification.generator import Generator
from tools.data_loader import MIDIDataset

from config import concepts_path, splits_root

import ICE.ModelWrapper
import ICE.Explainer
import ICE.utils
import shutil

concepts_path = os.path.join(concepts_path, "npy")


def prepare_data(gpu_number, target_classes, batch_size):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    device = "cuda:0"  ## TODO: import or configure (also used in trainer)

    seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    model_loc = "/share/cp/projects/concept_composers/experiments/kim2020/training/2202180920/model"
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
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

    # target_classes = [5, 6]
    classes_names = [target_composers[i] for i in target_classes]
    # n_components = (20, 20, 100)
    # layer_name = "layer4"
    # iter_max = 100

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
    return (loaders, classes_names, model)


def train_explainer(
    model,
    batch_size,
    target_classes,
    classes_names,
    layer_name,
    n_components,
    loaders,
    iter_max,
    dimension,
    reducer
):

    wm = ICE.ModelWrapper.PytorchModelWrapper(
        model,
        batch_size=batch_size,
        predict_target=target_classes,
        input_size=[2, 400, 88],
        input_channel_first=True,
        model_channel_first=True,
    )

    title = (
        "ConcComp_{}_{}[".format(layer_name, n_components)
        + "_".join(classes_names)
        + "]"
    )

    print("title:{}".format(title))
    print("target_classes:{}".format(target_classes))
    print("classes_names:{}".format(classes_names))
    print("n_components:{}".format(n_components))
    print("layer_name:{}".format(layer_name))

    # delete old files
    if Path("Explainers/" + title).exists():
        shutil.rmtree("Explainers/" + title)
    # create an Explainer
    exp = ICE.Explainer.Explainer(
        title=title,
        layer_name=layer_name,
        class_names=classes_names,
        utils=ICE.utils.img_utils(
            img_size=(400, 88), nchannels=2, img_format="channels_first"
        ),
        n_components=n_components,
        reducer_type=reducer,
        nmf_initialization=None,
        dimension=dimension,
        iter_max=iter_max,
    )
    # train reducer based on target classes
    exp.train_model(wm, loaders)
    return exp


def build_explanation(exp, wm, loaders):
    # generate features
    exp.generate_features(wm, loaders)
    # generate global explanations
    exp.global_explanations()
    # generate midi of global explanation
    exp._sonify_features()
    exp._sonify_features(contrast=True, unfiltered_midi=True)
    # save the explainer, use load() to load it with the same title
    exp.save()


@click.command()
@click.option("--reducer", help="Either NMF or NTD", default="NTD", type=str)
@click.option("--max-iter", default=10, type=int)
@click.option("--gpu-number", default=0, type=int)
@click.option(
    "--targets",
    help="A list of integers (target classes) as string",
    default="[5,6]",
    type=str,
)
@click.option("--dimension", default=4, type=int)
@click.option(
    "--rank",
    help="An integer, or list of integers as string",
    default="[20, 5,5, 100]",
    type=str,
)
@click.option(
    "--layer", help="The name of the target layer", default="layer4", type=str
)
@click.option("--batch-size", default=10, type=int)
def start_experiment(
    reducer, max_iter, gpu_number, targets, dimension, rank, layer, batch_size
):
    # convert targets string to list
    target_classes = json.loads(targets)
    rank = json.loads(rank)
    loaders, classes_names, model = prepare_data(
        str(gpu_number), target_classes, batch_size
    )
    train_explainer(
        model,
        batch_size,
        target_classes,
        classes_names,
        layer,
        rank,
        loaders,
        max_iter,
        dimension,
        reducer
    )


if __name__ == "__main__":
    start_experiment()
