import os 
import numpy as np
import sys
from numpy import random
import matplotlib.pyplot as plt
import cv2

# put parent desired fixation folder path
your_stored_path = './'

#change path to your drive
your_path = './'
# download dataset
if not os.path.exists(os.path.join(your_path,'left_fixation')):
  os.mkdir(os.path.join(your_path,'left_fixation'))
os.chdir(os.path.join(your_path,'left_fixation'))
for i in range(1,98):
  # download dataset of left fixation 
  get_ipython().system('wget ftp://gdchallenge18:1piN4nte5@ftp.ivc.polytech.univ-nantes.fr/Images/HE/Scanpaths/L/HEscanpath_'+str(i)+'.txt')
if not os.path.exists(os.path.join(your_path,'right_fixation')):
  os.mkdir(os.path.join(your_path,'right_fixation'))
os.chdir(os.path.join(your_path,'right_fixation'))
for i in range(1,98):
  # download dataset of left fixation 
  get_ipython().system('wget ftp://gdchallenge18:1piN4nte5@ftp.ivc.polytech.univ-nantes.fr/Images/HE/Scanpaths/R/HEscanpath_'+str(i)+'.txt')

files = [f for f in os.listdir(os.path.join(your_path,'left_fixation'))]
files.sort()

if not os.path.exists(your_stored_path+'/fixations'):
    os.mkdir(your_stored_path+'/fixations')   
if not os.path.exists(your_stored_path+'/fixations/left'):
    os.mkdir(your_stored_path+'/fixations/left')
if not os.path.exists(your_stored_path+'/fixations/right'):
    os.mkdir(your_stored_path+'/fixations/right')
for i,f in enumerate(files):
  scanpath1_path = os.path.join(your_path,'left_fixation',f)
  scanpath2_path = os.path.join(your_path,'right_fixation',f)
  # Load fixation lists
  fixations1 = np.loadtxt(scanpath1_path, delimiter=",", skiprows=1, usecols=(1,2))
  fixations2 = np.loadtxt(scanpath2_path, delimiter=",", skiprows=1, usecols=(1,2))
  fixations1 = fixations1 * [2048, 1024] - [1,1]; fixations1 = fixations1.astype(int)
  fixations2 = fixations2 * [2048, 1024] - [1,1]; fixations2 = fixations2.astype(int)
  fixmap1 = np.zeros((1024,2048), dtype=int)
  for iFix in range(fixations1.shape[0]):
    fixmap1[ fixations1[iFix, 1], fixations1[iFix, 0] ] += 1
  fixmap2 = np.zeros((1024,2048), dtype=int)
  for iFix in range(fixations2.shape[0]):
    fixmap2[ fixations2[iFix, 1], fixations2[iFix, 0] ] += 1
  # show images fixation 
  print(f"### image {i}")
  print('show left fixation ')
  cv2.imwrite(your_stored_path+'/fixations/left/'+str(i)+'.png',(fixmap1)*255)
  #plt.imshow(np.float32(fixmap1));plt.show();exit()
  print('show right fixation ')
  cv2.imwrite(your_stored_path+'/fixations/right/'+str(i)+'.png',(fixmap2)*255)
  #plt.imshow(np.float32(fixmap2));plt.show();exit() 
