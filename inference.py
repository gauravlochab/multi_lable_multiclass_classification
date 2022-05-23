import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from dataset import ImageDataset
from torch.utils.data import DataLoader

from torchvision import models as models
import torch.nn as nn

import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset


from PIL import Image
import numpy as np
import math
import random
from tqdm import tqdm
import shutil
import os
from pathlib import Path
import pandas as pd

# PyTorch libraries
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import argparse



parser = argparse.ArgumentParser(description='inference')
#parser.add_argument('--image_dir',help='input images directory',required=True)
parser.add_argument('--csv_file', help='csv path',required=True)
parser.add_argument('--ckpts', help='model-checkpoints',required=True)

args = vars(parser.parse_args())


#base_dir = os.path.abspath(args['image_dir'])
csv_path = args['csv_file']
ckpt_path = args['ckpts']
print('lolololol',csv_path,ckpt_path)

class ImageDataset(Dataset):
    def __init__(self, csv, train, test):
        self.csv = csv
        self.train = train
        self.test = test
        self.all_image_names = self.csv[:]['filename']
        self.all_labels = np.array(self.csv.drop(['filename','neck', 'sleeve_length','pattern'], axis=1))
        self.train_ratio = int(0.85 * len(self.csv))
        self.valid_ratio = len(self.csv) - self.train_ratio

        # set the training data images and labels
        if self.train == True:
            print(f"Number of training images: {self.train_ratio}")
            self.image_names = list(self.all_image_names[:self.train_ratio])
            print(self.image_names)
            self.labels = list(self.all_labels[:self.train_ratio])

            # define the training transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((400, 400)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
            ])

        # set the validation data images and labels
        elif self.train == False and self.test == False:
            print(f"Number of validation images: {self.valid_ratio}")
            self.image_names = list(self.all_image_names[-self.valid_ratio:-10])
            self.labels = list(self.all_labels[-self.valid_ratio:])

            # define the validation transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((400, 400)),
                transforms.ToTensor(),
            ])

        # set the test data images and labels, only last 10 images
        # this, we will use in a separate inference script
        elif self.test == True and self.train == False:
            self.image_names = list(self.all_image_names)
            self.labels = list(self.all_labels)

             # define the test transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])

    def __len__(self):
        #print('rebrebre')
        return len(self.image_names)
    
    def __getitem__(self, index):
        #print('vgrejoidbvre',self.image_names[index])
        try:
          image = cv2.imread(f"./classification-assignment/images/{self.image_names[index]}")
        except Exception as e:
            print('Path of the image is', self.image_names[index])
            print('Unable to read the image')
        
        #print(image.shape)
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.transform(image)
        targets = self.labels[index]
        #print(targets)
        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32),
            'file_name':self.image_names[index]
        }

    def get_test_Data(self):
      return self.samples



def model_load(pretrained, requires_grad):
    model = models.resnet50(progress=True, pretrained=pretrained)
    # to freeze the hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # to train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layer learnable
    # we have 21 classes in total
    model.fc = nn.Linear(2048, 21)
    return model

def main(model_path=ckpt_path,test_csv_path =csv_path):
  #read test csv
  print(model_path,test_csv_path)
  test_csv = pd.read_csv(csv_path)
  # initialize the computation device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #intialize the model
  model = model_load(pretrained=False, requires_grad=False).to(device)
  # load the model checkpoint
  checkpoint = torch.load(model_path)
  # load model weights state_dict
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()
  one_hot_neck = pd.get_dummies(test_csv.neck, prefix='neck')
  # one hot encode the sleeve_length attribute
  one_hot_sleeve_length = pd.get_dummies(test_csv.sleeve_length, prefix='sleeve_length')
  # one hot encode the pattern attribute
  one_hot_pattern = pd.get_dummies(test_csv.pattern, prefix='pattern')
  # concatenate the one hot encoded attributes to dataframe
  test_csv = pd.concat([test_csv, one_hot_neck, one_hot_sleeve_length, one_hot_pattern], axis=1)
  test_data = ImageDataset(
      test_csv, train=False, test=True
  )
  test_loader = DataLoader(
      test_data, 
      batch_size=1,
      shuffle=False
  )
  cols = ['filename', 'neck', 'sleeve','pattern']
  lst = []
  for counter, data in enumerate(test_loader):
      #print(data['file_name'])
      image, target = data['image'].to(device), data['label']
      #print(image.shape,target.shape)
      #get all the index positions where value == 1
      target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
      # get the predictions by passing the image through the model
      outputs = model(image)
      outputs = torch.sigmoid(outputs)
      outputs = outputs.detach().cpu()
      sorted_indices = np.argsort(outputs[0])
      best = sorted_indices[-3:]
      #print(data['file_name'],target_indices,best.detach().cpu().numpy()[0])
      lst.append([data['file_name'], best.detach().cpu().numpy()[0],best.detach().cpu().numpy()[1],best.detach().cpu().numpy()[2]])
  df1 = pd.DataFrame(lst, columns=cols)
  df1.to_csv('Output.csv')


if __name__ == "__main__":
    main()

