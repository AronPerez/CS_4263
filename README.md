# CS-4263-Project
CS-4263-001-Spring-2021-Deep Learning

## Directory structure

```text
input               --- Please locate data here
src
|-ensemble          --- For 4. Ensemble scripts
|-lib               --- Library codes
|-modeling          --- For 1. training, 2. prediction and 3. evaluation scripts
  |-results         --- Training, prediction and evaluation results will be stored here
README.md           --- This instruction file
requirements.txt    --- For python library versions
```
## Hardware (The following specs were used to create the original solution)

 - Ubuntu 18.04 LTS
 - 32 CPUs
 - 128GB RAM
 - 8 x NVIDIA Tesla V100 GPUs

 ## Software (python packages are detailed separately in `requirements.txt`):

Python 3.8.5
CUDA 10.1.243
cuddn 7.6.5
nvidia drivers v.55.23.0
-- Equivalent Dockerfile for the GPU installs: Use `nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04` as base image

Also, we installed OpenMPI==4.0.4 for running pytorch distributed training.

### Python Library

Deep learning framework, base library
 - torch==1.6.0+cu101
 - torchvision==0.7.0
 - l5kit==1.1.0
 - cupy-cuda101==7.0.0
 - pytorch-ignite==0.4.1
 - pytorch-pfn-extras==0.3.1

CNN models
 - [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch) ==0.7.4
 - [efficientnet_pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) ==0.7.0
 - [resnest](https://github.com/zhanghang1989/ResNeSt) ==0.0.6b20200912
 - [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) ==0.1.2
 - timm==0.3.1
 - Shapely==1.7.1

Data processing/augmentation
 - albumentations==0.4.3
 - scikit-learn==0.22.2.post1

We also installed `apex` https://github.com/nvidia/apex

Please refer `requirements.txt` for more details.


### Environment Variable
We recommend to set following environment variables for better performance.

```bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## Data setup

Please download competition data:
 - [lyft-motion-prediction-autonomous-vehicles](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/data)
 - [lyft-full-training-set](https://www.kaggle.com/philculliton/lyft-full-training-set)

For the `lyft-motion-prediction-autonomous-vehicles` dataset,
extract them under `input/lyft-motion-prediction-autonomous-vehicles` directory.

For the `lyft-full-training-set` data which only contains `train_full.zarr`,
please place it under `input/lyft-motion-prediction-autonomous-vehicles/scenes` as follows:
```text
input
|-lyft-motion-prediction-autonomous-vehicles
  |-scenes
    |-train_full.zarr (Place here!)
    |-train.zarr
    |-validate.zarr
    |-test.zarr
    |-... (other data)
  |-... (other data)

```


[Note] If you want to run training from scratch, please **remove `results` folder once**.
The training script tries to resume from `results` folder when `resume_if_possible=True` is set.

```bash
$ cd src/modeling

# Single GPU training (Please run this for first time, for input data cache creation)
$ python train_lyft.py --yaml_filepath ./flags/20201104_cosine_aug.yaml

# Multi GPU training (-n 8 for 8 GPU training)
$ mpiexec -x MASTER_ADDR=localhost -x MASTER_PORT=8899 -n 8 \
  python train_lyft.py --yaml_filepath ./flags/20201104_cosine_aug.yaml
```
We have trained 9 different models for final submission.
Each training configuration can be found in `src/modeling/flags`,
and the training results are located in `src/modeling/results`.

### Prediction for test dataset

predict_lyft.py` under `src/modeling` executes the prediction for test data.

Specify `out` as trained directory, the script uses trained model of this directory to inference.
Please set `--convert_world_from_agent true` after `l5kit==1.1.0`.

```bash
$ cd src/modeling
$ python predict_lyft.py --out results/20201104_cosine_aug --convert_world_from_agent true
```

Predicted results are stored under `out` directory.
For example, `results/20201104_cosine_aug/prediction_ema/submission.csv` is created with above setting.
