
import sys
sys.path.append('/work/soghigian_lab/abdullah.zubair/novel-species-detection')

import os, sys
os.chdir(os.path.dirname(os.getcwd()))

# --------------------------------------------------

import shutil
import glob
import time
import argparse
import json
from easydict import EasyDict
import copy
import pprint
from collections import namedtuple

import math
import numpy as np
import pandas as pd
pd.set_option('max_colwidth', None)
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchcontrib
import torch.backends.cudnn as cudnn

import fastai
from fastai.basic_data import DataBunch
# from fastai.vision import Learner
from modules.blend_data_augmentation import Learner
from fastai.distributed import setup_distrib, num_distrib

from tqdm import tqdm
from functools import partial

import models.model_list as model_list
from modules.adabound import AdaBound
from modules.ranger913A import Ranger
from modules.train_annealing import fit_with_annealing
import modules.swa as swa
from utils.dataloader import get_data_loaders, get_fastai_data_bunch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from utils.losses import MosLoss
from utils.metrics import accuracy, macro_f1, genus_accuracy, species_accuracy,\
                            genus_f1_score, species_f1_score
from utils.misc import log_metrics, cosine_annealing_lr
from utils.callbacks import SaveBestModel, WandbCallback
from utils.vis_utils import plttensor

# from utils.logger import Logger as TfLogger
# from tensorboardX import SummaryWriter
import wandb

from configs.config import config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    cudnn.benchmark = True

if not os.path.exists('./configs/old_configs/'+config.exp_name):
    os.makedirs('./configs/old_configs/'+config.exp_name)
shutil.copy2('./configs/config.py', './configs/old_configs/{}/config.py'
                    .format(config.exp_name))

if not os.path.exists('./model_weights/'+config.exp_name):
    os.makedirs('./model_weights/'+config.exp_name)
if not os.path.exists('./subm/'+config.exp_name):
    os.makedirs('./subm/'+config.exp_name) 
    
args = EasyDict()
args.dev_mode = False
args.resume = False
args.latest = False
args.realtime = False

pprint.pprint(config)

# --------------------------------------------------

print(os.getcwd())

# --------------------------------------------------

def get_class_map(df):
    class_map = {}
    for i in range(len(df)):
        class_map[df.loc[i, 'Species']] = df.loc[i, 'Species_Name']
    return class_map

# --------------------------------------------------

cm = get_class_map(pd.read_csv(config.DATA_CSV_PATH))
for i in range(len(cm)):
    print(i, ": ", cm[i])
print(len(cm))

# --------------------------------------------------

model_params = [config.exp_name, config.model_name]
MODEL_CKPT = os.path.abspath('./model_weights/{}/best_{}.pth'.format(*model_params))

Net = getattr(model_list, config.model_name)

net = Net(config=config)

gpu = setup_distrib(config.gpu)
opt = config.optimizer
mom = config.mom
alpha = config.alpha
eps = config.eps

if   opt=='adam': opt_func = partial(optim.Adam, betas=(mom,alpha), eps=eps)
elif opt=='adamw': opt_func = partial(optim.AdamW, betas=(mom,alpha), eps=eps)
elif opt=='radam': opt_func = partial(RAdam, betas=(mom,alpha), eps=eps)
elif opt=='novograd': opt_func = partial(Novograd, betas=(mom,alpha), eps=eps)
elif opt=='rms': opt_func = partial(optim.RMSprop, alpha=alpha, eps=eps)
elif opt=='sgd': opt_func = partial(optim.SGD, momentum=mom)
elif opt=='rangervar': opt_func = partial(RangerVar,  betas=(mom,alpha), eps=eps)
elif opt=='ranger': opt_func = partial(Ranger,  betas=(mom,alpha), eps=eps)
elif opt=='ralamb': opt_func = partial(Ralamb,  betas=(mom,alpha), eps=eps)
elif opt=='over9000': opt_func = partial(Over9000,  k=12, betas=(mom,alpha), eps=eps)
elif opt=='lookahead': opt_func = partial(LookaheadAdam, betas=(mom,alpha), eps=eps)
elif opt=='Adams': opt_func=partial(Adams)
elif opt=='rangernovo': opt_func=partial(RangerNovo)
elif opt=='rangerlars': opt_func=partial(RangerLars)

else:
    raise ValueError("Optimizer not recognized")
        
train_ds, valid_ds = get_data_loaders(config, get_dataset=True, one_hot_labels=config.one_hot_labels)
print("Train dataset size = {}".format(len(train_ds)))
data = DataBunch.create(train_ds, valid_ds, bs=config.batch_size,
                         num_workers=config.num_workers)
loss = MosLoss(config=config)

freeze_bn = False
save_imgs = False
train_losses = []
valid_losses = []
valid_f1s = []
lr_hist = []

# callback_fns=[WandbCallback] if (config.wandb and not get_learn) else []

print('Training ...')
print('Saving to ', MODEL_CKPT)
metrics = [partial(accuracy, one_hot_labels=config.one_hot_labels), partial(macro_f1, one_hot_labels=config.one_hot_labels)]

# --------------------------------------------------

learn = (Learner(data, net, wd=config.weight_decay, opt_func=opt_func,
         metrics=metrics,
         bn_wd=False, true_wd=True,
         loss_func = loss,
         # loss_func = LabelSmoothingCrossEntropy(),
#          callback_fns=callback_fns,
         model_dir=MODEL_CKPT)
        )

if gpu is None: learn.to_parallel()
elif num_distrib()>1: learn.to_distributed(gpu)

best_save_cb = SaveBestModel(learn, config=config)    

# --------------------------------------------------

learn.fit_one_cycle(config.epochs, config.lr, div_factor=10, pct_start=0.3, callbacks=[best_save_cb])        

# --------------------------------------------------

# ## Test

# --------------------------------------------------


# --------------------------------------------------


# --------------------------------------------------

# ## Plots

# --------------------------------------------------


# --------------------------------------------------

from configs.old_configs.paper_redo.fold4.config import config
import pandas as pd
import math
import numpy as np
import pandas as pd
pd.set_option('max_colwidth', None)
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
subm = pd.read_csv('./subm/{}/best_acc.csv'.format(config.exp_name))
(subm['Species'] == subm['SpeciesPred']).sum()/len(subm)

# --------------------------------------------------

def plot_pretty_blue_confusion_matrix(y_true, y_pred, classes,
                                      normalize=False,
                                      title=None,
                                      cmap=plt.cm.Blues,
                                      savepath=None,
                                      figsize=(24,24)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    precisions = np.diag(cm) / np.sum(cm, axis = 0)
    recalls = np.diag(cm) / np.sum(cm, axis = 1)
    print("Precision: ", np.round(precisions, 4))
    print("Recall: ", np.round(recalls, 4))

    # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix")

    # print(cm)
    save_df = pd.DataFrame(data=cm, columns=classes)
    save_df.index = classes
    if savepath is not None:
        save_df.to_csv(savepath.replace(savepath.split('/')[-1],'confusion.csv'), index=True)
    else:
        save_df.to_csv('./subm/confusion.csv', index=True)

    fig, ax = plt.subplots(figsize=figsize)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.xticks(fontsize=round(figsize[0]*.5))
    plt.yticks(fontsize=round(figsize[0]*.5))
    ax.set_xlabel('True label',fontsize=round(figsize[0]*.75))
    ax.set_ylabel('Predicted label',fontsize=round(figsize[0]*.75))
    plt.title(label=title,fontsize=round(figsize[0]))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # ax.tick_params(direction='out', length=6, width=2, colors='r',
    #            grid_color='r', grid_alpha=0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.25)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if math.isnan(val):
                val = 0.0
            ax.text(j, i, format(val, fmt),
                    ha="center", va="center", size=round(figsize[0]*.5),
                    color="white" if cm[i, j] > thresh else "black")
    if savepath is not None:
        fig.savefig(savepath)
    return ax

# --------------------------------------------------

# from post_plots import plot_pretty_blue_confusion_matrix 
plot_pretty_blue_confusion_matrix(subm['Species'], subm['SpeciesPred'],
                      np.array([i[1] for i in config.class_map[:config.num_species.sum()]]), normalize=True,
                      savepath='./subm/{}/confusion_matrix.png'.format(config.exp_name))

# --------------------------------------------------

#!python post_plots.py --infile ./subm/run68/best_acc.csv

# --------------------------------------------------

# ## Generate Features & Probabilities for full dataset

# --------------------------------------------------


# --------------------------------------------------


# --------------------------------------------------

