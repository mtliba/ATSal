from functools import partial
import numpy as np
from numpy import random
import time
from skimage import exposure
from skimage.transform import resize
import cv2
from model import Sal_based_Attention_module
import re, os, glob
import cv2
import torch
import scipy.misc
from torchvision import utils

EPSILON = np.finfo('float').eps


def normalize(x, method='standard', axis=None):
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res


def AUC_Judd(saliency_map, fixation_map, jitter=False):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        saliency_map += random.rand(*saliency_map.shape) * 1e-7
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F]  # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds) + 2)
    fp = np.zeros(len(thresholds) + 2)
    tp[0] = 0;
    tp[-1] = 1
    fp[0] = 0;
    fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh)  # Total number of saliency map values above threshold
        tp[k + 1] = (k + 1) / float(n_fix)  # Ratio saliency map values at fixation locations above threshold
        fp[k + 1] = (above_th - k - 1) / float(n_pixels - n_fix)  # Ratio other saliency map values above threshold
    return np.trapz(tp, fp)  # y, x


def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F]  # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # For each fixation, sample n_rep values from anywhere on the saliency map
    if rand_sampler is None:
        r = random.randint(0, n_pixels, [n_fix, n_rep])
        S_rand = S[r]  # Saliency map values at random locations (including fixated locations!? underestimated)
    else:
        S_rand = rand_sampler(S, F, n_rep, n_fix)
    # Calculate AUC per random split (set of random locations)
    auc = np.zeros(n_rep) * np.nan
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:, rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds) + 2)
        fp = np.zeros(len(thresholds) + 2)
        tp[0] = 0;
        tp[-1] = 1
        fp[0] = 0;
        fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k + 1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k + 1] = np.sum(S_rand[:, rep] >= thresh) / float(n_fix)
        auc[rep] = np.trapz(tp, fp)
    return np.mean(auc)  # Average across random splits


def KLD(p, q):
    p = normalize(p, method='sum')
    q = normalize(q, method='sum')
    return np.sum(np.where(p != 0, p * np.log((p + EPSILON) / (q + EPSILON)), 0))


def NSS(saliency_map, fixation_map):
    s_map = np.array(saliency_map)
    f_map = np.array(fixation_map)
    f_map = f_map > 0
    if s_map.shape != f_map.shape:
        s_map = resize(s_map, f_map.shape)
    # Normalize saliency map to have zero mean and unit std
    s_map = normalize(s_map, method='standard')
    # Mean saliency value at fixation locations
    return np.mean(s_map[f_map])


def CC(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3,
                      mode='constant')  # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have zero mean and unit std
    map1 = normalize(map1, method='standard')
    map2 = normalize(map2, method='standard')
    # Compute correlation coefficient
    return np.corrcoef(map1.ravel(), map2.ravel())[0, 1]


def SIM(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3,
                      mode='constant')  # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have values between [0,1] and sum up to 1
    map1 = normalize(map1, method='range')
    map2 = normalize(map2, method='range')
    map1 = normalize(map1, method='sum')
    map2 = normalize(map2, method='sum')
    # Compute histogram intersection
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)


metrics = {
    "AUC_Judd": [AUC_Judd, False, 'fix'],  # Binary fixation map
    "AUC_Borji": [AUC_Borji, False, 'fix'],  #  Binary fixation map
    "NSS": [NSS, False, 'fix'],  #  Binary fixation map
    "CC": [CC, False, 'sal'],  #  Saliency map
    "SIM": [SIM, False, 'sal'],  #  Saliency map
    "KLD": [KLD, False, 'sal']}  #  Saliency map

#  Possible float precision of bin files
dtypes = {16: np.float16,
          32: np.float32,
          64: np.float64}


# get_binsalmap_infoRE = re.compile("(\w+_\d{1,2})_(\d+)x(\d+)_(\d+)b")
def get_binsalmap_info(filename):
    name, width, height, dtype = get_binsalmap_infoRE.findall(filename.split(os.sep)[-1])[0]
    width, height, dtype = int(width), int(height), int(dtype)
    return name, width, height


def getSimVal(salmap1, salmap2, fixmap1=None, fixmap2=None):
    values = []

    for metric in keys_order:

        func = metrics[metric][0]
        sim = metrics[metric][1]
        compType = metrics[metric][2]

        if not sim:
            if compType == "fix" and not "NoneType" in [type(fixmap1), type(fixmap2)]:
                m = (func(salmap1, fixmap1)
                     + func(salmap2, fixmap1)) / 2
            else:
                m = (func(salmap1, salmap2)
                     + func(salmap2, salmap1)) / 2
        else:
            m = func(salmap1, salmap2)

        values.append(m)

    return values


model = Sal_based_Attention_module()
wrihgt = torch.load('weight/train1_wheight_epoch_10.pt')
checlpoint = wrihgt['state_dict']
model.load_state_dict(checlpoint)
model.cuda()

if __name__ == "__main__":
    from time import time

    t2 = time()
    #  Similarité metrics to compute and output to file
    keys_order = ['AUC_Judd', 'AUC_Borji', 'NSS', 'CC', 'SIM', 'KLD']

    # Head-only data
    # SM_PATH = "../H/SalMaps/"
    # SP_PATH = "../H/Scanpaths/"
    # Head-and-Eye data
    IMG_Path = "./validation_set/image/"
    SM_PATH = "./validation_set/saliency/"
    SP_PATH = "./validation_set/fixation/"

    """
    One issue when comparing saliency maps in equirectangular format, is that poles of the sphere are over-represented because of the latitudinal distortions
    One possible correction is to take N points uniformely sampled on a sphere
        see blog.wolfram.com/2011/07/28/how-i-made-wine-glasses-from-sunflowers/
        A bigger N means a better approximation, the solution given above shows irregularities at sin = 0
    We know an equirectangular map's vetical distortion is a function of sin(y/np.pi). We propose as a second solution to multiply all rows of the saliency matrix with a weight vector modeled:
        sin(linspace(0, pi, height)) - O to pi with as many steps as vertical pixels
    """
    SAMPLING_TYPE = [  #  Different sampling method to apply to saliency maps
        "Sphere_9999999",  # Too many points
        "Sphere_1256637",  # 100,000 points per steradian
        "Sphere_10000",  # 10,000
        "Sin",  # Sin(height)
        "Equi"  # None
    ]
    SAMPLING_TYPE = SAMPLING_TYPE[-2]  #  Sin weighting by default
    print("SAMPLING_TYPE:", SAMPLING_TYPE)

    #  Path to vieo saliency maps we wish to compare
    image_list = os.listdir(IMG_Path)
    image_list.sort()
    final = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    i = 0
    for img in image_list:
        t1 = time()
        ground_truth_path = SM_PATH + img[:-4] + '.npy'
        print(ground_truth_path)
        scanpath1_path = SP_PATH + img[:-4] + '.npy'
        print(scanpath1_path)
        image_path = IMG_Path + img
        print(image_path)
        # fixmap1 = cv2.imread(scanpath1_path,0)
        # fixmap1 = cv2.resize(fixmap1/255,(640,320))
        # fixmap1 = np.array([fixmap1])
        # fixmap1.shape)
        # salmap1 = cv2.imread(ground_truth_path,0)
        # salmap1 = cv2.resize(salmap1/255, (640, 320))
        fixmap1 = np.load(scanpath1_path)
        salmap1 = np.load(ground_truth_path)
        salmap1 = np.array([salmap1])
        inpt = cv2.imread(image_path)
        inpt = cv2.resize(inpt, (640, 320))
        inpt = np.float32(inpt)
        inpt -= [0.485, 0.456, 0.406]
        inpt = torch.cuda.FloatTensor(inpt)
        inpt = inpt.permute(2, 0, 1)
        inpt = inpt.unsqueeze(0)
        fixmap1 = np.array([fixmap1])
        with torch.no_grad():
            saliency_map, _ = model(inpt)
        saliency_map = saliency_map.squeeze(0)
        saliency_map = saliency_map.permute(0, 1, 2)
        saliency_map = (saliency_map.cpu()).detach().numpy()

        salmap1 = normalize(salmap1, method='sum')
        saliency_map = normalize(saliency_map, method='sum')
        
        fixmap1[fixmap1>0]=1
        s = NSS(salmap1,fixmap1)
        print(s)
        # Compute similarity metrics
        #a = AUC_Judd(saliency_map, fixmap1)
        #print(a)
        values = getSimVal(saliency_map,salmap1,fixmap1)
        # Outputs results

        #print("Name, metric, value")
        final = final +np.array(values)
        for iVal, val in enumerate(values):
            print("{}, {}, {}".format(img, keys_order[iVal], val))

        #print("T_delta = {}".format(time() - t1))
    print("******* final *******")
    for iVal, val in enumerate(final/len(image_list)):
        print("{}, {}, {}".format('final', keys_order[iVal], val))

        #print("Total_delta = {}".format(time() - t2))
