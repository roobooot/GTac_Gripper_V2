import logging
import random

import numpy as np
import pandas as pd

import torch
from torch_geometric.data import InMemoryDataset

import transforms.tograph

log = logging.getLogger(__name__)

class BioTacSp(InMemoryDataset):

  def __init__(self, root, k, split="train", normalize=True, csvs=None, transform=None, pre_transform=None):

    self.split = split
    self.csvs = csvs
    self.k = k
    self.normalize = normalize
    self.mins = []
    self.maxs = []

    super(BioTacSp, self).__init__(root, transform, pre_transform)

    self.data, self.slices = torch.load(self.processed_paths[0])

    # Compute class weights for sampling
    '''
      #change to the number of the classes
    '''
    '''#change to the number of the classes'''
    self.class_weights = np.zeros(15)
    for i in range(len(self.data['y'])):
      self.class_weights[self.data['y'][i]] += 1
    self.class_weights /= len(self.data['y'])

  @property
  def raw_file_names(self):
    if (self.split == "train"):
      return ['train_dataset.csv']
    elif (self.split == "test"):
      return ['test_dataset.csv']
    elif (self.split == "eval"):
      return ['val_dataset.csv']
    elif (self.split == None):
      return self.csvs

  @property
  def processed_file_names(self):
    if (self.split == "train"):
      return ["biotacsp_{0}.pt".format(self.k)]
    elif (self.split == "test"):
      return ["biotacsp_test_{0}.pt".format(self.k)]
    elif (self.split == None):
      return ['biotacsp_' + ''.join(self.csvs) + '.pt']

  def download(self):

    url_ = "https://github.com/yayaneath/biotac-sp-images"

    raise RuntimeError(
      "Dataset not found. Please download {} from {} and move it to {}".format(
        str(self.raw_file_names),
        url_,
        self.raw_dir))

#this function here is important
  def process(self):

    transform_tograph_ = transforms.tograph.ToGraph(self.k) #pass the assert k

    data_list_ = []

    for f in range(len(self.raw_paths)):

      log.info("Reading CSV file {0}".format(self.raw_paths[f]))

      grasps_ = pd.read_csv(self.raw_paths[f])       #read the raw data

      for i in range(len(grasps_)):

        sample_ = self._sample_from_csv(grasps_, i)
        sample_ = transform_tograph_(sample_)      # generate graphical data

        if self.pre_transform is not None:
          sample_ = self.pre_transform(sample_)

        data_list_.append(sample_)

    if self.normalize: # Feature scaling
      raw_dataset_np_ = np.array([sample.x.numpy() for sample in data_list_])

      self.mins = np.min(raw_dataset_np_, axis=(0, 1))
      self.maxs = np.max(raw_dataset_np_, axis=(0, 1))

      for i in range(len(data_list_)):
        data_list_[i].x = torch.from_numpy((data_list_[i].x.numpy() - self.mins) / (self.maxs - self.mins))

    data_ = self.collate(data_list_)

    torch.save(data_, self.processed_paths[0])

  def _sample_from_csv(self, grasps, idx):

    sample_ = grasps.iloc[idx]

    object_ = sample_.iloc[576]
    #slipped_ = sample_.iloc[1]

    finger1_x_data = np.copy(sample_.iloc[0:48]).astype(np.int, copy=False)
    finger1_y_data = np.copy(sample_.iloc[48:96]).astype(np.int, copy=False)
    finger1_z_data = np.copy(sample_.iloc[96:144]).astype(np.int, copy=False)
    finger2_x_data = np.copy(sample_.iloc[144:192]).astype(np.int, copy=False)
    finger2_y_data = np.copy(sample_.iloc[192:240]).astype(np.int, copy=False)
    finger2_z_data = np.copy(sample_.iloc[240:288]).astype(np.int, copy=False)
    finger3_x_data = np.copy(sample_.iloc[288:336]).astype(np.int, copy=False)
    finger3_y_data = np.copy(sample_.iloc[336:384]).astype(np.int, copy=False)
    finger3_z_data = np.copy(sample_.iloc[384:432]).astype(np.int, copy=False)
    finger4_x_data = np.copy(sample_.iloc[432:480]).astype(np.int, copy=False)
    finger4_y_data = np.copy(sample_.iloc[480:528]).astype(np.int, copy=False)
    finger4_z_data = np.copy(sample_.iloc[528:576]).astype(np.int, copy=False)



    '''
    data_index_ = np.copy(sample_.iloc[2:26]).astype(np.int, copy=False)
    data_middle_ = np.copy(sample_.iloc[26:50]).astype(np.int, copy=False)
    data_thumb_ = np.copy(sample_.iloc[50:75]).astype(np.int, copy=False)
    '''


    sample_ = {'object': object_,
               'F1_x': finger1_x_data,
               'F1_y': finger1_y_data,
               'F1_z': finger1_z_data,
               'F2_x': finger2_x_data,
               'F2_y': finger2_y_data,
               'F2_z': finger2_z_data,
               'F3_x': finger3_x_data,
               'F3_y': finger3_y_data,
               'F3_z': finger3_z_data,
               'F4_x': finger4_x_data,
               'F4_y': finger4_y_data,
               'F4_z': finger4_z_data,
              }
    '''
    'data_index': data_index_,
    'data_middle': data_middle_,
    'data_thumb': data_thumb_
    '''

    return sample_

  def get_train_test(self, dataset, shuffle=True, TRAINRATIO=0.8, VALRATIO=0.1):
    if shuffle:
       random.shuffle(dataset)#dataset =dataset.shuffle()

    train_dataset = dataset[:int(TRAINRATIO * len(dataset))]
    val_dataset = dataset[int(TRAINRATIO * len(dataset)):int((TRAINRATIO + VALRATIO) * len(dataset))]
    test_dataset = dataset[int((TRAINRATIO + VALRATIO) * len(dataset)):]
    return train_dataset, val_dataset, test_dataset
