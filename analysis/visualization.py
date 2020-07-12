import time
import os
import numpy as np
import torch
import warnings
from config.args_analysis_visualize import *
from utils.helper_evaluation import Visualizer
import models

from datasets.s3dis_loader import S3DISDataset
from datasets.scannet_loader import ScanNetDataset

warnings.filterwarnings('ignore')
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda") if args.cuda else torch.device("cpu")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

log_path = args.log_path

if not os.path.exists(log_path):
    raise ValueError('The given log dose not exists:' + log_path)


chosen_snapshot = args.snapshot
chosen_relu = args.relu  # Which feature is visualized
chosen_deformation = args.deformation  # which feature is visualized(index of the deform operation in the network )
compute_activations = args.compute_activations
# Because of the time needed to compute feature activations for the test set, if you already computed them, they
# are saved and used again. Set this parameter to True if you want to compute new activations and erase the old
# ones. N.B. if chosen_relu = None, the code always recompute activations. Chose a relu idx to avoid it.

batch_size = args.batch_size
in_radius = args.in_radius
dl0 = args.first_subsampling_dl
#dataset.load_subsampled_clouds(dl0)




print("Dataset Preparation")

if args.dataset_name.startswith('S3DIS'):
    dataset = S3DISDataset()
elif args.dataset_name.startwith("ScanNet"):
    dataset = ScanNetDataset()
else:
    raise ValueError('Unsupported dataset:' + args.dataset_name)

print("Creating Model")

if args.dataset.startswith('S3DIS'):
    model = models.RandLA_Net_Scannet()
elif args.dataset.startswith('ScanNet'):
    model = models.RandLA_Net_Scannet()
else:
    raise ValueError('Unsupported dataset: ' + args.dataset_name)

print("Start Visualization")


class ModelVisualizer:
    def __init__(self, model, restore_snap=None):

    def top_relu_activations(self, model, dataset, relu_idx=0, top_num=5):
        """
        Test the model on test dataset to see which points active the most each neurons in a relu layer
        :param model: model used at training
        :param dataset: dataset used at training
        :param relu_idx: which features are to be visualized
        :param top_num: how many top candidates are kept per features
        """

