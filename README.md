# Robust Real-World Image Super-Resolution against Adversarial Attacks

This repository is an official PyTorch implementation of the paper **"Robust Real-World Image Super-Resolution against Adversarial Attacks"** from **ACM MM 2021**.

We provide the training and testing codes, pre-trained models. You can train your model from scratch, or use the pre-trained model.

## Code
### Dependencies
* Python 3.6
* PyTorch >= 1.1.0
* numpy
* cv2
* skimage
* tqdm
* torch-dct


### Quick Start
Clone this github repo.
```bash
git clone https://github.com/yuejiutao/Robust-Real-World-Image-Super-Resolution-against-Adversarial-Attacks.git
cd Robust-Real-World-Image-Super-Resolution-against-Adversarial-Attacks
```

#### Folder
We recommend that you use the following directory structure：<br>
```
yourfolder
└─Code
│   └─Robust-Real-World-Image-Super-Resolution-against-Adversarial-Attacks
│   └─Other project...
└─Data
│   └─RealSR
│       └─x4
│         └─test_LR
│         └─test_HR
│         └─train_LR
│         └─train_HR
│         └─adv
└─Other folder...         
```

#### Training
1. Download the RealSR dataset([Version3](https://github.com/csjcai/RealSR)) and unpack them like above. Then, change the ```dataroot``` and ```test_dataroot``` argument in ```./options/realSR_HGSR_MSHR.py``` to the place where images are located.
2. Run the adversarial training with ```train.py``` using script file ```train.sh```.
```bash
sh train.sh
```
3. You can change the ```exp_name``` in ```./options/realSR_HGSR_MSHR.py``` and find the results in  ```./experiments/exp_name```.


#### Testing
1. Download our pre-trained models to ```Robust-Real-World-Image-Super-Resolution-against-Adversarial-Attacks/pre/``` folder or use your pre-trained models
2. Change the ```test_dataroot``` argument in ```test.sh``` to the place where images are located
3. Run ```test.sh```.
```bash
sh test.sh
```
4. You can find the enlarged images under different adversarial intensities in ```/yourfolder/Data/RealSR/x4/adv/CDC_MC/``` folder

#### Pretrained models


## Citation
If you find our work useful in your research or publication, please cite:
```
@inproceedings{yue2021robust,
  title={Robust Real-World Image Super-Resolution against Adversarial Attacks},
  author={Yue, Jiutao and Li, Haofeng and Wei, Pengxu and Li, Guanbin and Lin, Liang},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={5148--5157},
  year={2021}
}
```
