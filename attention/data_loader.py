import torch
import os
import datetime
import numpy as np
import cv2
from torch.utils import data
from torchvision import utils

# The DataLoader for our specific datataset with extracted frames
class Static_dataset(data.Dataset):

  def __init__(self, split, number_of_frames, root_path, load_gt = True, resolution=None, val_perc = 0.2):
        # augmented frames
        self.frames_path = os.path.join(root_path,'image') 
        
        self.load_gt = load_gt

        if load_gt:
          #ground truth        
          self.gt_path = os.path.join(root_path, "saliency")
          #self.fx_pah = os.path.join(root_path, "fixation") 

        self.resolution = resolution
        self.frames_list = []
        self.gt_list = []
        #self.fx_list = []
        # Gives accurate human readable time, rounded down not to include too many decimals
        print('start load data')
        frame_files = os.listdir(os.path.join(self.frames_path))
        self.frames_list = sorted(frame_files, key = lambda x: int(x.split(".")[0]) )
        self.frames_list = self.frames_list[:number_of_frames]
        print(' load images data')
        if load_gt:
           gt_files = os.listdir(os.path.join(self.gt_path))
           #fx_files = os.listdir(os.path.join(self.fx_pah ))


           self.gt_list = sorted(gt_files, key = lambda x: int(x.split(".")[0]) )
           #self.fx_list = sorted(fx_files, key = lambda x: int(x.split(".")[0]) )

           print(' load groundtruth data')
           self.gt_list = self.gt_list[:number_of_frames]
           #self.fx_list = self.fx_list[:number_of_frames]

        print('data loaded')
        '''
        limit = int(round(val_perc*len(self.frames_list)))
        limit = len(self.frames_list) - limit 
        print(self.frames_list[limit])
        if split == "train":
          self.frames_list = self.frames_list[:limit]
          self.gt_list = self.gt_list[:limit]
          self.fx_list = self.fx_list[:limit]
          self.gt_module_list = self.gt_module_list[:limit]

        elif split == "validation":
          self.frames_list = self.frames_list[limit:]
          self.gt_list = self.gt_list[limit:]
          self.fx_list = self.fx_list[limit:]         
          self.gt_module_list = self.gt_module_list[limit:]
        '''
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.frames_list)

  def __getitem__(self, frame_index):

        'Generates one sample of data'
        
        frame = self.frames_list[frame_index]
        
        if self.load_gt:
          gt = self.gt_list[frame_index]
          

        path_to_frame = os.path.join(self.frames_path, frame)
      
        X = cv2.imread(path_to_frame)


             
        X = X.astype(np.float32)
        
        X = X - [0.485, 0.456, 0.406]
        X = torch.cuda.FloatTensor(X)
        X = X.permute(2,0,1)
        # add batch dim
        data = X.unsqueeze(0)
        data = X

        # Load and preprocess ground truth (saliency maps)
        if self.load_gt:

            path_to_gt = os.path.join(self.gt_path , gt)

            # Load as grayscale
            
            y = np.load(path_to_gt) 
                
                
        

        if self.load_gt:
            y = y.astype(np.float32)
            y = (y-np.min(y))/(np.max(y)-np.min(y))
            y = torch.cuda.FloatTensor(y)
            #y = y.permute(2,0,1)
            gt = y.unsqueeze(0)


            
            packed = (data,gt) # pack data with the corresponding  ground truths
        else:
            packed = (data, "_")


        return packed
