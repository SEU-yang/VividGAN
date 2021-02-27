# VividGAN-TIP2021

Code for Face Hallucination With Finishing Touches (TIP 2021). 

!https://ieeexplore.ieee.org/document/9318504

## Prerequisites:

Requirements: Python=3.6 and Pytorch>=1.0.0

### 

1. Install Pytorch

2. Prepare Dataset  (I am using Multi-PIE, CelebA and MMI as the training set.)

## Train: 

python train.py

Change the option in train.py to set the dataset's directory. 

## Test

python test.py

## Citation

If you find VividGAN useful in your research, please consider citing:
```
ARTICLE{9318504,
  author={Y. {Zhang} and I. W. {Tsang} and J. {Li} and P. {Liu} and X. {Lu} and X. {Yu}},
  journal={IEEE Transactions on Image Processing}, 
  title={Face Hallucination With Finishing Touches}, 
  year={2021},
  volume={30},
  number={},
  pages={1728-1743},
  doi={10.1109/TIP.2020.3046918}}
```
