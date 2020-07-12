import matplotlib.image as mpimg
import numpy as np
import os ,sys
from numpy import random
import matplotlib.pyplot as plt
import cv2
# put parent desired fixation folder path
your_stored_path = './'
# path of stored scan_path csv files
saved_scanpath ='./Scan_Path/'
# fill dictionary and list with names and frames number of your videos
my_dict={'12_TeatroRegioTorino':600,'14_Warship':500,'15_Cockpit':500,'18_Bar':501,'1_PortoRiverside':500,'3_PlanEnergyBioLab':500,'4_Ocean':600}
listvid=['12_TeatroRegioTorino','14_Warship','15_Cockpit','18_Bar','1_PortoRiverside','3_PlanEnergyBioLab','4_Ocean']
i= 0 ;j= 0
# download dataset
if not os.path.exists(os.path.join(saved_scanpath,'left_fixation')):
  os.mkdir(os.path.join(saved_scanpath,'left_fixation'))
os.chdir(os.path.join(saved_scanpath,'left_fixation'))
for vid in listvid:
  # download dataset of left fixation 
  get_ipython().system('wget ftp://gdchallenge18:1piN4nte5@ftp.ivc.polytech.univ-nantes.fr/Videos/HE/Scanpaths/L/'+vid+'_fixations.csv')
if not os.path.exists(os.path.join(saved_scanpath,'right_fixation')):
  os.mkdir(os.path.join(saved_scanpath,'right_fixation'))
os.chdir(os.path.join(saved_scanpath,'right_fixation'))
for vid in listvid:
  # download dataset of left fixation 
  get_ipython().system('wget ftp://gdchallenge18:1piN4nte5@ftp.ivc.polytech.univ-nantes.fr/Videos/HE/Scanpaths/R/'+vid+'_fixations.csv')

# to devide the total number of frame to pooled chunk of frames 
def getFramePoolingIdx(tempWindowSize, FrameCount):
	tempWindowSize = int( np.round(FrameCount / 20 * (tempWindowSize/1000)))
	frameLost = FrameCount % tempWindowSize
	framePooling = np.arange(0, FrameCount+1, tempWindowSize) + frameLost//2
	framePooling = np.concatenate([ framePooling[:-1, None],
									framePooling[ 1:, None]-1 ], axis=1).astype(int)
	return framePooling
#in order to render resulted fixation map throughout pooled frame
def getPooledFramesFM(fixations, range_, shape, path = None ):
  iStart, iEnd = range_
  print(range(iStart, iEnd+1))
  global  i,j
  ii= 00 
  fixationmap = np.zeros(shape, dtype=int)
  for iFrame in range(iStart, iEnd+1):
    FIX = np.where(np.logical_and( fixations[:, 2] <= iFrame, fixations[:, 3] >= iFrame ) )[0]
    for iFix in FIX:
      fixationmap[ int(round(fixations[iFix, 1])), int(round(fixations[iFix, 0])) ] += 1
      ii+=1
    if path == 'left' :
        a = 4-len(str(i));a = '0'*a + str(i);im_name = a+'.png';i+=1
        cv2.imwrite(your_stored_path+'/fixations/'+vid+'/left/'+im_name,(np.fliplr(fixationmap))*255)
    if path == 'right':
        a = 4-len(str(j));a = '0'*a + str(j);im_name1 = a+'.png';j+=1
        cv2.imwrite(your_stored_path+'/fixations/'+vid+'/left/'+im_name1,(np.fliplr(fixationmap))*255)
  return np.fliplr(fixationmap)

# go through csv files and extract and save fixations
for vid in listvid:
    scanpath2_path = saved_scanpath +"/left_fixation/"+vid+"_fixations.csv"
    scanpath1_path = saved_scanpath +"/right_fixation/"+vid+"_fixations.csv"
    if not os.path.exists(your_stored_path+'/fixations'):
        os.mkdir(your_stored_path+'/fixations')   
    if not os.path.exists(your_stored_path+'/fixations/'+vid):
        os.mkdir(your_stored_path+'/fixations/'+vid)
    if not os.path.exists(your_stored_path+'/fixations/'+vid+'/left'):
        os.mkdir(your_stored_path+'/fixations/'+vid+'/left')
    if not os.path.exists(your_stored_path+'/fixations/'+vid+'/right'):
        os.mkdir(your_stored_path+'/fixations/'+vid+'/right')
    fixations1 = np.loadtxt(scanpath1_path, delimiter=",", skiprows=1, usecols=(1,2, 5,6))
    fixations2 = np.loadtxt(scanpath2_path, delimiter=",", skiprows=1, usecols=(1,2, 5,6))
    fixations1 = fixations1 * [2048, 1024, 1,1] - [1,1, 0,0]
    fixations2 = fixations2 * [2048, 1024, 1,1] - [1,1, 0,0]
    i=0;j=0
    fPool = getFramePoolingIdx(200, 500)
    for iFrame in range(fPool.shape[0]):
        # Retrieve fixations map for frame range
      print('visualize left fixation') 
      fixmap1_frame = getPooledFramesFM(fixations1, fPool[iFrame, :], [1024, 2048],'left')
      print('visualize right fixation') 
      fixmap2_frame = getPooledFramesFM(fixations2, fPool[iFrame, :], [1024, 2048],'right')
      #plt.imshow(fixmap1_frame);plt.show();exit() 
      #plt.imshow(fixmap2_frame);plt.show();exit() 
