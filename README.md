# [NeurIPS 2025] PPMStereo: Pick-and-Play Memory Construction for Consistent Dynamic Stereo Matching

**[City University of Hong Kong](https://www.cityu.edu.hk/en)**

Yun Wang, Junjie Hu*, Qiaole Dong, Yongjian Zhang, Yanwei Fu, Tin Lun Lam, Dapeng Wu

[[`Paper`](https://research.facebook.com/publications/dynamicstereo-consistent-dynamic-depth-from-stereo-videos/)] [[`Project`](https://dynamic-stereo.github.io/)] [[`BibTeX`](#citing-dynamicstereo)]

![nikita-reading](https://user-images.githubusercontent.com/37815420/236242052-e72d5605-1ab2-426c-ae8d-5c8a86d5252c.gif)

**PPMStereo** is a transformer-based architecture for temporally consistent depth estimation from stereo videos. It has been trained on a combination of two datasets: [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) and **Dynamic Replica** dataset.

### Download the Dynamic Replica dataset
Download `links.json` from the *data* tab on the [project website](https://dynamic-stereo.github.io/) after accepting the license agreement.
```
git clone https://github.com/facebookresearch/dynamic_stereo
cd dynamic_stereo
export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH
```
Add the downloaded `links.json` file to the project folder. Use flag `download_splits` to choose dataset splits that you want to download: 
```
python /scripts/download_dynamic_replica.py --link_list_file links.json \
--download_folder /data/ywang/dataset/ --download_splits real valid test train
```

Memory requirements for dataset splits after unpacking (with all the annotations):
- train - 1.8T
- test - 328G
- valid - 106G
- real - 152M

You can use [this PyTorch dataset class](https://github.com/facebookresearch/dynamic_stereo/blob/dfe2907faf41b810e6bb0c146777d81cb48cb4f5/datasets/dynamic_stereo_datasets.py#L287) to iterate over the dataset.

## Installation

Describes installation of PPMStereo with the latest PyTorch3D, PyTorch 1.12.1 & cuda 11.3

### Setup the root for all source files:
```
git clone https://github.com/facebookresearch/dynamic_stereo
cd dynamic_stereo
export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH
```
### Create a conda env:
```
conda create -n ppmstereo python=3.10
conda activate ppmstereo
```
### Install requirements
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=12.1 -c pytorch
# It will require some time to install PyTorch3D. In the meantime, you may want to take a break and enjoy a cup of coffee.
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -r requirements.txt
```

### (Optional) Install RAFT-Stereo
```
mkdir third_party
cd third_party
git clone https://github.com/princeton-vl/RAFT-Stereo
cd RAFT-Stereo
bash download_models.sh
cd ../..
```



## Evaluation

You can  download the checkpoints manually by clicking the links below. Copy the checkpoints to `./ckpt/ppmstereo/`.

- [PPMStereo](https://drive.google.com/drive/folders/1mMPmpw0gGuwpmTvylkuSalLWzf3-LHFv?usp=drive_link) trained on SceneFlow
- [PPMStereo](https://drive.google.com/drive/folders/1mMPmpw0gGuwpmTvylkuSalLWzf3-LHFv?usp=drive_link) trained on SceneFlow and *Dynamic Replica*

To evaluate PPMStereo:
```
python ./evaluation/evaluate.py --config_name eval_dynamic_replica_40_frames \
 MODEL.model_name=PPMStereoModel exp_dir=./outputs/test_dynamic_replica_ds \
 MODEL.PPMStereoModel.model_weights=./ckpt/ppmstereo_stereo_sf.pth 
```
Due to the high image resolution, evaluation on *Dynamic Replica* requires a 48GB GPU. If you don't have enough GPU memory, you can decrease `kernel_size` from 20 to 10 by adding `MODEL.PPMStereoModel.kernel_size=10` to the above python command. Another option is to decrease the dataset resolution. Additionally, we recommend reducing `iters = 20` to `iters = 10` with only a slight drop in accuracy to facilitate the evaluation process.

As a result, you should see the numbers from *Table 3* in the [paper](https://arxiv.org/pdf/2305.02296.pdf). (for this, you need `kernel_size=20`)

Reconstructions of all the *Dynamic Replica* splits (including *real*) will be visualized and saved to `exp_dir`.

If you installed [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), you can run:
```
python ./evaluation/evaluate.py --config-name eval_dynamic_replica_40_frames \
  MODEL.model_name=RAFTStereoModel exp_dir=./outputs/test_dynamic_replica_raft
```

Other public datasets we use: 
 - [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
 - [Sintel](http://sintel.is.tue.mpg.de/stereo)
 - [Middlebury](https://vision.middlebury.edu/stereo/data/)
 - [ETH3D](https://www.eth3d.net/datasets#low-res-two-view-training-data)
 - [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_stereo.php) 

## Training
Training requires a100 GB GPUs. You can decrease `image_size` and / or `sample_len` if you don't have enough GPU memory.
```
python train.py --batch_size 2 \
 --spatial_scale -0.2 0.4 --image_size 320 512 --saturation_range 0 1.4 --num_steps 200000  \
 --ckpt_path dynamicstereo_sf_dr  \
  --sample_len 5 --lr 0.0003 --train_iters 10 --valid_iters 20    \
  --num_workers 28 --save_freq 100  --update_block_3d --different_update_blocks \
  --attention_type self_stereo_temporal_update_time_update_space --train_datasets dynamic_replica things monkaa driving
```
If you want to train on SceneFlow only, remove the flag `dynamic_replica` from `train_datasets`.

## License
The project of PPMStereo is licensed under the MIT license.


## Citing DynamicStereo
If you use our model in your research, please use the following BibTeX entry.
```
@article{wang2025ppm,
  title={PPMStereo: Pick-and-Play Memory Construction for Consistent Dynamic Stereo Matching},
  author={Yun Wang, Junjie Hu, Qiaole Dong, Yongjian Zhang, Yanwei Fu, Tin Lun Lam, Dapeng Wu},
  journal={NeurIPS},
  year={2025}
}
```
## Acknowledgement
In this project, we use parts of public codes and thank the authors for their contribution in: 

[DynamicStereo](https://github.com/facebookresearch/dynamic_stereo)

[BidaStereo](https://github.com/TomTomTommi/bidastereo)

[StereoAnyVideo](https://github.com/TomTomTommi/stereoanyvideo)

We thank the original authors for their excellent works.
