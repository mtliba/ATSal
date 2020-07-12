# ATSal : An Attention Based Architecture for Saliency Prediction in 360◦ Videos
![](https://img.shields.io/badge/python-v3.6.8-orange.svg?style=flat-square)
![](https://img.shields.io/badge/pytorch-v1.2.0-orange.svg?style=flat-square)
![](https://img.shields.io/badge/torchvision-v0.4.0-orange.svg?style=flat-square)
![](https://img.shields.io/badge/opencv-v4.0.0.21-orange.svg?style=flat-square)
![](https://img.shields.io/badge/numpy-v1.16.2-orange.svg?style=flat-square)


## Abstract :
The spherical domain representation of 360◦
video/image presents many challenges related to the storage, processing, transmission and rendering of omnidirectional videos (ODV). Models of human visual attention can be used so that only a single viewport is rendered at a time, which is important when developing systems that allow users to explore ODV with head mounted displays (HMD). Accordingly, researchers have proposed various saliency models for 360◦ video/images. This paper proposes ATSal, a novel attention based (head-eye) saliency
model for 360◦ videos. The attention mechanism explicitly encodes global static visual attention allowing expert models to focus on learning the saliency on local patches throughout consecutive frames. We compare the proposed approach to other state-ofthe-art saliency models on two datasets: Salient360! and VREyeTracking. Experimental results on over 80 ODV videos (75K+ frames) show that the proposed method outperforms the existing state-of-the-art.

## Publication :
Find the extended pre-print version of our work on [arXiv](https://) .

- Reference 

if this project has been helpful to your work ,Please cite with the following Bibtex code:

```
@article{Mtliba-2020-ATSal,
author = {Yasser, Dahou and Marouane, Tliba and McGuinness, Kevin and O'Connor, Noel},
title = {ATSal : An Attention Based Architecture for Saliency Prediction in 360◦ Videos},
journal = {arXiv},
month = {July},
year = {2020}
}
```
Cite with human friendly style :

*Yasser Dahou, Marouane Tliba, Kevin McGuinnessand Noel E. O’Connor
"ATSal : An Attention Based Architecture for Saliency Prediction in 360◦ Videos"arXiv. 2020.*

## Model Architecture  :

![architecture-fig]

[architecture-fig]: https://raw.githubusercontent.com/mtliba/ATSal/asset/image/model.PNG
"ATSal architecture"


## Model Parameters  :

ATSal attention model initialization :

* [[intitial (100 MB)]]()

ATSal attention model trained on Salient360! and Sitzman image dataset:

* [[attention-model (100 MB)]]()

ATSal attention model trained on Salient360! and VR-EyeTracking video dataset:

* [[ATSal-attention (100 MB)]]()

ATSal expert models trained on Salient360! and VR-EyeTracking video dataset:

* [[ATSal-experts-Poles (100 MB)]]()
* [[ATSal-experts-Equator (100 MB)]]()

## DATASETS:

*link to augmented Salient360! and Sitzman Image DATASETS:
[augmented-static-dataset](http://)

*link to reproduced VR-EYETRACKING DATASETS:
[VR-EYETRACKING](http://)

## COMPARATIVE PERFORMANCE STUDY ON: SALIENT360! , VR-EYETRACKING DATASETS:
![result-fig]

[result-fig]: https://github.com/mtliba/ATSal/blob/asset/image/result.PNG?raw=true


