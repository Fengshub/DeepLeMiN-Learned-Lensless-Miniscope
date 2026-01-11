# DeepInMiniscope: Learned Integrated Miniscope
Feng Tian, Ben Mattison, Weijian Yang "DeepInMiniscope: Deep learning–powered physics-informed integrated miniscope"
### Clone this repository:
```
git clone https://github.com/Fengshub/DeepLeMiN-Learned-Lensless-Miniscope
```

## [Paper link](https://www.science.org/doi/10.1126/sciadv.adr6687)
### [preprint paper](https://www.biorxiv.org/content/10.1101/2024.05.03.592471v1)

## 2D sample reconstructions
Dataset for 2D reconstruction test of green stained [**lens tissue**](https://drive.google.com/drive/folders/1lsAjVdHU8wLL1I7Y60G6kH1KkP-dSooJ?usp=drive_link) <br /><br />
Input: measured [**image**](https://drive.google.com/file/d/1RM8N0R-_M4KtLYLxMbP1nVtHrzkRl8qo/view?usp=drive_link) of green fluorescent stained lens tissue, dissembled into sub-FOV patches.<br />
Output: the [**reconstructed**](https://drive.google.com/file/d/1rXcDQbROTnVweovR2DKKLYlkijlpa-Bk/view?usp=drive_link) slide containing green lens tissue features.<br />
[**Code**](https://github.com/Fengshub/DeepInMiniscope-Learned-Lensless-Miniscope/blob/main/2D_lenstissue/2D_lenstissue.py) for Multi-FOV ADMM-Net model to generate reconstruction results. The function of each script section is described at the beginning of each section.<br />
[**Code**](https://github.com/Fengshub/DeepInMiniscope-Learned-Lensless-Miniscope/blob/main/2D_lenstissue/lenstissue_2D.m) to display the generated image and reassemble sub-FOV patches.<br />

## 3D sample reconstructions
[**Dataset**](https://drive.google.com/drive/folders/1Zejm5FODAm7GRUgAYJpx1vZBarTAVnNT?usp=drive_link) for 3D reconstruction test of in-vivo mouse brain video recording.<br /><br />
Input: Time-series standard-deviation of difference-to-local-mean weighted raw video.<br />
Output: reconstructed 4-D volumetric video containing 3-dimensional distribution of neural activities.<br />
[**Code**](https://github.com/Fengshub/DeepInMiniscope-Learned-Lensless-Miniscope/blob/main/3D_mouse/3D%20mouse.py) for Multi-FOV ADMM-Net model to generate reconstruction results. The function of each script section is described at the beginning of each section.<br />
[**Code**](https://github.com/Fengshub/DeepInMiniscope-Learned-Lensless-Miniscope/blob/main/3D_mouse/mouse_3D.m) to display the generated image and calculate temporal correlation.<br />

## Photolithography masks for microlens array fabrication and resolution characterization
Navigate to `./Photomasks` for mask design files, including [**lithography_mask**](https://github.com/Yang-Research-Laboratory/DeepInMiniscope-Learned-Integrated-Miniscope/blob/main/Photomasks/lithography_mask_v12.GDS) for metal deposition, [**calibration target**](https://github.com/Yang-Research-Laboratory/DeepInMiniscope-Learned-Integrated-Miniscope/blob/main/Photomasks/calibrationtarget.dxf) for resolution characterization, [**sample target**](https://github.com/Yang-Research-Laboratory/DeepInMiniscope-Learned-Integrated-Miniscope/blob/main/Photomasks/target_sample.stl) for training dataset acquisition and [**macro files**](https://github.com/Yang-Research-Laboratory/DeepInMiniscope-Learned-Integrated-Miniscope/blob/main/Photomasks/target_sample_script_v2.FCMacro) for automated batch mask generation in FreeCAD. The complete .zip file for training target can be downloaded [**here**](https://drive.google.com/file/d/1QnZeh7LNLx590gcw4X_ggBbsCfexrvBK/view?usp=sharing).

## Zemax optical design files and doublet microlens unit CAD
Navigate to `./Photomasks` to view doublet lens CAD, load the main [**Zemax file**](https://github.com/Yang-Research-Laboratory/DeepInMiniscope-Learned-Integrated-Miniscope/blob/main/Zemax/doublet_unit.zmx) for optical layout and view doublet lens parameters.

## Running List-RL Algorithm  
Navigate to `./list_rl `and follow the instructions in list_RL.m to run sample reconstructions, or provide your input datta (measured images from mask-based imager) to perform reconstructions.

### Example workflow that combines all provided modules:
*→ Simulate optics using Zemax <br />
→ Fabricate mask using photomask files <br />
→ Assemble miniscope with CAD housing <br />
→ Acquire raw measurement images <br />
→ Reconstruct volume using List-RL or MultiFOVADMM-Net model <br />
→ Post-process / visualize output<br />*

## schematic of imager
![schematicimager](https://github.com/Fengshub/3D-Microscope/blob/main/imgs/schematicimager.PNG)
## assembled imager
![assembleimager](https://github.com/Fengshub/DeepLeMiN-Learned-Lensless-Miniscope/blob/main/imgs/assembleimager.jpg)


