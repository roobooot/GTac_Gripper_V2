import logging

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

import utils.plotconfusionmatrix

import dataset.biotacsp

import torch
import torch.nn.functional as F
import torch.utils.data.dataloader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import network.utils

import argparse

log = logging.getLogger(__name__)

def eval(model, device, loader, plot=True):

    ## Launch predictions on test and calculate metrics
    acc_ = 0.0
    y_ = []
    preds_ = []

    model.eval()

    for batch in loader:

        batch = batch.to(device)
        pred_ = model(batch).max(1)[1]
        acc_ += pred_.eq(batch.y).sum().item()

        batch.y = batch.y.int()

        y_.append(batch.y.int())
        preds_.append(pred_.int())
        print(f'y: {batch.y}, pred_:{pred_}')
        # print(f'preds_: {preds_}')

    # TODO: OJO A QUE ESTO NO COJA LA LONGITUD EN BATCHES
    log.info("CHECK CHECK CHECK: {0}".format(len(loader)))
    acc_ /= len(loader)

    prec_, rec_, fscore_, _ = precision_recall_fscore_support(y_, preds_, average='weighted')

    log.info("Metrics")
    log.info("Accuracy: {0}".format(acc_))
    log.info("Precision: {0}".format(prec_))
    log.info("Recall: {0}".format(rec_))
    log.info("F-score: {0}".format(fscore_))

    if (plot):
        conf_matrix_ = confusion_matrix(y_, preds_)
        print(f'y_:{y_}, pred_:{preds_}')
        ## Plot non-normalized confusion matrix
        utils.plotconfusionmatrix.plot_confusion_matrix(conf_matrix_, classes=np.unique(y_),
                            title='Confusion matrix, without normalization')


'''
parser_ = argparse.ArgumentParser(description="Parameters")
parser_.add_argument("--log_path", nargs="?", default="logs", help="Logging path")
parser_.add_argument("--split", nargs="?", default="test", help="Dataset split to evaluate")#test
#parser_.add_argument("--checkpoint", nargs="?", required=True,  help="Path to save checkpoints")#default="C:\\Users\\Lenovo\\Desktop\\tactile-gcn-12features\\ckpts\\train-GCN_test-False-0-Aug22_23-22-36_fold4_val.pkl",
parser_.add_argument("--normalize", nargs="?", type=bool, default=True, help="Normalize dataset using feature scaling")
parser_.add_argument("--graph_k", nargs="?", type=int, default=0, help="K-Neighbours for graph connections, use 0 for manual connections")
parser_.add_argument("--batch_size", nargs="?", type=int, default=1, help="Batch Size")
parser_.add_argument("--network", nargs="?", default="GCN_test", help="The network model to train")
args = parser_.parse_args()
'''

'''
biotacsp_dataset = dataset.biotacsp.BioTacSp(root='../data/biotacsp/', k=0, split="test", normalize=False)#root='../data/biotacsp/'
#biotacsp_dataset_ = dataset.biotacsp.BioTacSp(root='data/biotacsp', k=args.graph_k, split=args.split, normalize=args.normalize)
eval_loader_ = DataLoader(biotacsp_dataset, batch_size=1, shuffle=False, num_workers=1)

## Select CUDA device
device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
log.info(device_)
log.info(torch.cuda.get_device_name(0))

## Build model
model_ = network.utils.get_network("GCN_test", 12, 9).to(device_)#model,num_features,num_class
log.info(model_)

checkpoint_ = torch.load("C:\\Users\\Lenovo\\Desktop\\tactile-gcn-12features\\ckpts\\train-GCN_test-False-0-Aug22_23-22-36_fold4_val.pkl")
model_.load_state_dict(checkpoint_['model_state'])
eval(model_,device_,eval_loader_,plot=True)
'''


