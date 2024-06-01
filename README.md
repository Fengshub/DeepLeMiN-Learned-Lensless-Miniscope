# DeepLeMiN_private
Feng Tian, Ben Mattison, Weijian Yang "DeepLeMiN: Deep-learning-powered Physics-aware Lensless Miniscope"
### Clone this repository:
```
git clone https://github.com/Fengshub/DeepLeMiN-Learned-Lensless-Miniscope
```

## [preprint paper](https://www.biorxiv.org/content/10.1101/2024.05.03.592471v1)

## 2D sample reconstructions
Dataset for 2D reconstruction test of green stained [**lens tissue**](https://github.com/Fengshub/DeepLeMiN_private/tree/main/2D%20reconstructions_Lenstissue) <br /><br />
Input: measured [**image**](https://drive.google.com/drive/folders/1nHIXtpC-AYwwulnrEYHi7ey7W1iOFivf?usp=drive_link) of green fluorescent stained lens tissue, dissembled into sub-FOV patches.<br />
Output: the [**reconstructed**](https://drive.google.com/drive/folders/1nHIXtpC-AYwwulnrEYHi7ey7W1iOFivf?usp=drive_link) slide containing green lens tissue features.<br />
[**Code**](https://github.com/Fengshub/DeepLeMiN_private/blob/main/2D%20reconstructions_Lenstissue/2D_lenstissue.py) for Multi-FOV ADMM-Net model to generate reconstruction results. The function of each script section is described at the beginning of each section.<br />
[**Code**](https://github.com/Fengshub/DeepLeMiN_private/blob/main/2D%20reconstructions_Lenstissue/2D_lenstissue.m) to display the generated image and reassemble sub-FOV patches.<br />

## 3D sample reconstructions
[**Dataset**](https://drive.google.com/drive/folders/16JSduy1YqkJh47kPMQdfikxNKHB7Yi_T?usp=drive_link) for 3D reconstruction test of in-vivo mouse brain video recording.<br /><br />
Input: Time-series standard-derivation of difference-to-local-mean weighted raw video.<br />
Output: reconstructed 4-D volumetric video containing 3-dimensional distribution of neural activities.<br />
[**Code**](https://github.com/Fengshub/DeepLeMiN_private/blob/main/3D%20reconstructions_mouse/3D%20mouse.py) for Multi-FOV ADMM-Net model to generate reconstruction results. The function of each script section is described at the beginning of each section.<br />
[**Code**](https://github.com/Fengshub/DeepLeMiN_private/blob/main/3D%20reconstructions_mouse/3D%20mouse.m) to display the generated image and calculate temporal correlation.<br />

## schematic of imager
![schematicimager](https://github.com/Fengshub/3D-Microscope/blob/main/imgs/schematicimager.PNG)
## assembled imager
![assembleimager](https://github.com/Fengshub/3D-Microscope/blob/main/imgs/assembleimager.png)

