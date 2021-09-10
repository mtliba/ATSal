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



Find the extended pre-print version of our work on [arXiv](https://) .



## Head end Eye movment prediction in omnidirectonal video :
Is the task that aims to model the gaze fixation distribution patterns of humans on static and dynamic omnidirectional scenes, due to the predicted saliency map which defined as a heatmap of probabilities, where every probability corresponds to how likely it is that the corresponding pixel will attract human attention, so it could be used to prioritize the information across space and time for videos, this task is quite beneficial in a variety of computer vision applications including image and video compression, image segmentation, object recognition, etc.
## Model Architecture  :

![architecture-fig]

[architecture-fig]: https://raw.githubusercontent.com/mtliba/ATSal/asset/image/model.PNG
"ATSal architecture"


## Model Parameters  :

ATSal attention model initialization :

* [[intitial (374 MB)]](https://drive.google.com/file/d/1qT4tALLSGmsRfqf_dJ-1nhS_3iT4fFMg/view?usp=sharing)

ATSal attention model trained on Salient360! and Sitzman image dataset:

* [[attention-model (374 MB)]](https://drive.google.com/file/d/1dGnufIVaqmKxKm1jvMn65fWa-E4sxld9/view?usp=sharing)

ATSal attention model trained on Salient360! and VR-EyeTracking video dataset:

* [[ATSal-attention (374 MB)]](https://drive.google.com/file/d/18tqbKvKgm7jsWUul0F0QsGhKYTg9XOrA/view?usp=sharing)

ATSal expert models trained on Salient360! and VR-EyeTracking video dataset:

* [[ATSal-experts-Poles (364 MB)]](https://drive.google.com/file/d/1N0u-jZEU6042F2YJDaTvXyCLcv5m-3Tj/view?usp=sharing)
* [[ATSal-experts-Equator (364 MB)]](https://drive.google.com/file/d/1l1DuvpTeqkS5wJhdHPEdGwxBmi4EFaAc/view?usp=sharing)

## DATASETS:
saliency prediction studies in 360◦images are still limited.  The  absence  of  common  head and eye-gaze datasets for 360◦content and difficulties of their reproducibility compared with publicly provided 2D stimuli dataset could be one of the reasons that have hindered progress in the development of computational saliency models on this front so that here we are providing a reproduced version of VR- EyeTracking Dataset with 215 videos,  and  an augmented version of Sitzmann_TVCG_VR  dataset with 440 images. 
* link to augmented Salient360! and Sitzman Image DATASETS:
[augmented-static-dataset](https://drive.google.com/drive/folders/1EJgxC6SzjehWi3bu8PRVHWJrkeZbAiqD?usp=sharing)

* link to reproduced VR-EYETRACKING DATASETS:
[VR-EYETRACKING](https://mtliba.github.io/Reproduced-VR-EyeTracking/)

## COMPARATIVE PERFORMANCE STUDY ON: SALIENT360! , VR-EYETRACKING DATASETS:
![result-fig]

[result-fig]: https://github.com/mtliba/ATSal/blob/asset/image/result.PNG?raw=true


## Test model: 
To test a pre-trained model on video data and produce saliency maps, execute the following command:
```
cd test/weight
weight.sh

cd ..
python test -'path to your video dataset' -'output path'

```

## Demo:
Here we are providing a comparison between attention stream, expert stream, and our final model ATSal, as shown bellow the attention stream overestimate salient area where it predicts the static global attention, the expert models predict dynamic saliency on each viewport independently based on its content and location but still introduce artifact on viewports boundaries and ignore the global attention statistic, unlike the fusion of both streams 'ATSal model' that is better at attending salient information distribution over space and time. 

![](https://github.com/mtliba/ATSal/blob/asset/image/output62.gif)
![](https://github.com/mtliba/ATSal/blob/asset/image/output%20(4).gif)
## Contact:
For questions, bug reports, and suggestions about this work, please create an [issue](https://github.com/mtliba/ATSal/issues) in this repository or send an email to mtliba@inttic.dz .

