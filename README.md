# Hierarchical Temporal Transformer for 3D Hand Pose Estimation and Action Recognition from Egocentric RGB Videos


Original implementation of the paper Yilin Wen, Hao Pan, Lei Yang, Jia Pan, Taku Komura and Wenping Wang, "Hierarchical Temporal Transformer for 3D Hand Pose Estimation and Action Recognition from Egocentric RGB Videos", CVPR, 2023. 
[[Paper]](https://arxiv.org/pdf/2209.09484.pdf)[[Supplementary Video]](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/ylwen_connect_hku_hk/EqSS-_AKROVDsKSzb6FMuZYBrsrmAZ7yuwvVXX4pT6c3ug?e=yE8gFK)

A version of extended abstract was accpeted by the _Human Body, Hands, and Activities from Egocentric and Multi-view Cameras Workshop_, ECCV, 2022. [[Extended Abstract]](https://fylwen.github.io/misc/HTT_eccvw_extended_abstract.pdf)

## Requirements
### Environment

The code is tested with the following environment:
```  
Ubuntu 20.04
python 3.9
pytorch 1.10.0
torchvision 0.11.0
```

Other dependent packages as included in ```requirements.txt``` can be installed by pip. Note that we also refer to the utility functions in [```libyana```](https://github.com/hassony2/libyana). To install this ```libyana``` library, we follow [LPC, CVPR 2020](https://github.com/hassony2/handobjectconsist/blob/master/environment.yml/#L35) to run:
```
pip install git+https://github.com/hassony2/libyana@v0.2.0
```

### Data Preprocessing

To facilitate computation, for downloaded [FPHA](https://guiggh.github.io/publications/first-person-hands/) and [H2O](https://taeinkwon.com/projects/h2o/) datasets: We resize all images into the 480x270 pixels, and use lmdb to manage the training images. One may refer to the ```preprocess_utils.py``` for related functions.


### Pretrained Model
Our pretrained weights for FPHA and H2O, and other related data for running the demo code of the inference stage can be downloaded via the following link:
[[Inference Data]](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/ylwen_connect_hku_hk/EoE_gh8D2dZNkdDWvaM3ZmwBYpQd7c2WjqDshr5qy_Zklg?e=mhXyKS)

which includes:
1) ```./ckpts/```: The pretrained ckpt files for the FPHA and H2O datasets.
2) ```./demo_fpha/```: Demo test sequence from FPHA dataset. Here we have already resized the demo images into the size of 480x270 pixels.
3) ```./curves/```: .npz files for visualizing the 3D PCK(-RA) at different error thresholds on FPHA and H2O. 


You may keep the downloaded ```ws``` folder under the root directory of this git repository.


## Quick Start
### Demo 
For the demo _drink mug_ clip from the FPHA dataset, run
```
CUDA_VISIBLE_DEVICES=0 python eval.py  --val_split test \
--train_dataset fhbhands --val_dataset fhbhands \
--dataset_folder ./ws/demo_fpha/ \
--resume_path ./ws/ckpts/htt_fpha/checkpoint_45.pth  --is_demo
```

and check the ```./ws/vis/out.avi``` [(Demo)](https://connecthkuhk-my.sharepoint.com/:v:/g/personal/ylwen_connect_hku_hk/EThs-9gNWURJuvmHWybBVLQBqSKw4BEjvhucwadaxXOZkg?e=YXkI1k) for the qualitative result of our estimated 3D hand pose in the camera space and its 2D projection in the image plane. We also label our output action category.



### Plot 3D PCK(-RA) Curves for Hand Pose Estimation

Run
```
python plot_pck_curves.py
```
to plot the curves the 3D PCK(-RA) at different error thresholds on FPHA and H2O.

### Evaluation for Hand Pose Estimation and Action Recognition

Run
```
CUDA_VISIBLE_DEVICES=0 python eval.py --batch_size <batch_size> \
--val_split <val_split> --train_dataset <dataset> --val_dataset <dataset> \
--dataset_folder <path_to_dataset_root> \
--resume_path <path_to_pth>
```
for evaluation on the dataset and split given by ```<dataset>``` and ```<val_split>```. 

Note that for the test split of H2O, we report the hand MEPE and action recall rate by referring to our submitted results in the [H2O challenge codalab](https://taeinkwon.com/projects/h2o/).

## Training

Run ```python train.py``` with parsed arguments to train a network with regard to your training data. 

## Acknowledgement
 
For the transformer architecture, we rely on the code of [DETR, ECCV 2020](https://github.com/facebookresearch/detr/blob/main/models/transformer.py) and [Attention is All You Need, NeurIPS 2017](https://nlp.seas.harvard.edu/annotated-transformer/#positional-encoding).

For evaluation of 3D hand pose estimation, we follow the code of [```libyana```](https://github.com/hassony2/libyana/blob/master/libyana/evalutils/zimeval.py) and original [ColorHandPose3DNetwork, ICCV 2017](https://github.com/lmb-freiburg/hand3d/blob/master/utils/general.py).

For data processing and augmentation, resnet architecture, and other utility functions, our code is heavily relied on the code of [LPC, CVPR 2020](https://github.com/hassony2/handobjectconsist) and [```libyana```](https://github.com/hassony2/libyana). 



## Citiation
If you find this work helpful, please consider citing
```
@article{wen2023hierarchical,
  title={Hierarchical Temporal Transformer for 3D Hand Pose Estimation and Action Recognition from Egocentric RGB Videos},
  author={Wen, Yilin and Pan, Hao and Yang, Lei and Pan, Jia and Komura, Taku and Wang, Wenping},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
