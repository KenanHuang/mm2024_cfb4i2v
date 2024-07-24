The main code of our approach is stored in **cfb_vp**, the code of generating source frames is stored in **depthstillation**.

## Training Steps:

### 1. Prepare the pretrained model of I3D

The *rgb_imagenet.pt* and *flow_imagenet.pt* can be found in https://github.com/piergiaj/pytorch-i3d/tree/master/models.

please download them and store into **codes/cfb_vp/models**

### 2. Prepare the data file

For training and evaluation,we need to prepare two datafiles for source and target domain respectively.

Source data are recorded as followed:

```
/data/datasets/lab/EADs/EAD_image_dataset/clap/applauding_136.jpg 0
/data/datasets/lab/EADs/EAD_image_dataset/clap/applauding_152.jpg 0
/data/datasets/lab/EADs/EAD_image_dataset/clap/applauding_114.jpg 0
/data/datasets/lab/EADs/EAD_image_dataset/clap/applauding_006.jpg 0
```

Target data are recorded as followed:

```
/data/datasets/lab/hmdb51/avi/clap/#20_Rhythm_clap_u_nm_np1_le_goo_4.avi 0
/data/datasets/lab/hmdb51/avi/clap/Hand_Clapping_Game__Bim_Bum__clap_f_nm_np2_fr_med_1.avi 0
/data/datasets/lab/hmdb51/avi/clap/Kurt_Kr_mer_-_Klatschen_im_Flugzeug_clap_u_cm_np1_fr_med_1.avi 0
/data/datasets/lab/hmdb51/avi/clap/Wendy_playing_wii_fit_with_clapping_hands_clap_f_cm_np1_le_med_1.avi 0
```

Both of them are formated in "{data path} {ground-truth label}"

### 3. Generate source frames

The code stored in **depthstillation** contains the depthmap estimation network *MiDas*. And the main code for generating realistic source frames.

For users,just need to replace the the value of *--input_sourcelist* and *--output_dir_path* and run the command as followed:

```bash
bash ./start.sh
```

### 4. Train model

**train_with_rgbflow_rpflow_random_videopace.py** is our main code of implement.

We put the demo scripts for training model in the root of **cfb_vp**.

For trainning the model for S2U benchmark,just run the following command as followed:

```bash
bash ./train_with_rgbflow_rpflow_random_videopace-s2u.sh
```

For trainning the model for E2h benchmark,just run the following command as followed:

```bash
bash ./train_with_rgbflow_rpflow_random_videopace-e2h.sh
```

We also put our training logs in training_logs, we contain two types of value for Accuracy including  Acc and AccSf.

Acc means we sum the logits of RGB and flow branches.

AccSf means we sum the logits after softmax of RGB and flow branches, which is  reported in our manuscript.
