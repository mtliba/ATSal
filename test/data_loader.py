import torch
import os
import datetime
import numpy as np
import cv2
from torch.utils import data
from torchvision import utils
from tqdm import tqdm
class Test_dataset(data.Dataset):

  def __init__(self, root_path):

        self.root_path = root_path

        self.video_list = []
        self.video_name_list = []
        for i in tqdm(os.listdir(root_path)):
          
            if i=='.ipynb_checkpoints':
              continue
            frame_files = [os.path.join(self.root_path, str(i), file) for file in os.listdir(os.path.join(self.root_path, str(i))) if file.endswith('.png')]
            frame_files_sorted = sorted(frame_files)
            self.video_list.append(frame_files_sorted)
            self.video_name_list.append(i)


  def __len__(self):
 
        return len(self.video_list)
  def video_names(self):
        return self.video_name_list
  def __getitem__(self, video_index):

        'Generates one sample of data'
        frames = self.video_list[video_index]
        data = []
        final = []
        list_of_frame = []
        for i, path_to_frame in enumerate(frames):
          if path_to_frame.split('.')[-1]=='ipynb_checkpoints':
            continue
          X = cv2.imread(path_to_frame)
          print(path_to_frame)
          X = cv2.resize(X, (640, 320))
          X = X.astype(np.float32)
          X -= [103.939, 116.779, 123.68]
          X = torch.FloatTensor(X)
          X = X.permute(2,0,1)
          data.append(X.unsqueeze(0))
          list_of_frame.append(path_to_frame)
          
          if (i+1)%20 == 0 or i == (len(frames)-1):
            data_tensor = torch.cat(data,0)
            data = []
            final.append((data_tensor,list_of_frame))
            list_of_frame = []
        return final