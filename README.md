# CS-4263-Project
CS-4263-001-Spring-2021-Deep Learning

# Synopsis
The L5Kit data is stored in zarr format which is basically a set of numpy structured arrays. Conceptually, it is similar to a set of CSV files with records and different columns. As for any zarr file, there must be a root folder. So, the L5Kit data consists of groups of zarr datasets, we can see it by using zl5.info

# Data Overview

Scenes : a collection of frames:

- Frames : a scene has a list of frames that start from scene.frame_index_interval[0] and ends at scene.frame_index_interval[1]
- Host : a scene has a host which is the AV that films the scene.
- Timestamps: a scene has a start_time and an end_time

Frames : a collection of agents (the host agents + other agents)
Agents : Any object in circulation with the automatic vehicle (AV)
agents_mask: a mask that (for train and validation) masks out objects that aren't useful for training. In test, the mask (provided in files as mask.npz) masks out any test object for which predictions are NOT required.
Traffic_light_faces : traffic lights and their faces (bulbs)

# L5Kit
- Load driving scenes from zarr files
- Read semantic maps
- Read aerial maps
- Create birds-eye-view (BEV) images which represent a scene around an AV or another vehicle
- Sample data
- Train neural networks
- Visualize results

# How to run
* To get access to the full dataset, please use the following [Google Drive link](https://drive.google.com/drive/folders/1cC6BD46FMK9-UryYAKEeHRXG7qYdNTBL?usp=sharing)
* Load the notebook into [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)
* Select master branch
* Set your runtime to GPU or TPU for best performance
* Be sure to connect the new drive you have access to with the following
```
from google.colab import drive
drive.mount('/content/drive')
```
# Config file
* The config file is located under ``/content/drive/MyDrive/lyft-motion-prediction-autonomous-vehicles/config.yaml`` 
This variable is set with the following
```python
# set env variable for data
DIR_INPUT = "/content/drive/MyDrive/lyft-motion-prediction-autonomous-vehicles/"
cfg = load_config_data("./config.yaml")
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)
```
The inputs to the dataset are already mapped in it
* Continue loading each dataset
The output should look like the following
```
+------------+------------+------------+---------------+-----------------+----------------------+----------------------+----------------------+---------------------+
| Num Scenes | Num Frames | Num Agents | Num TR lights | Total Time (hr) | Avg Frames per Scene | Avg Agents per Frame | Avg Scene Time (sec) | Avg Frame frequency |
+------------+------------+------------+---------------+-----------------+----------------------+----------------------+----------------------+---------------------+
|   16265    |  4039527   | 320124624  |    38735988   |      112.19     |        248.36        |        79.25         |        24.83         |        10.00        |
+------------+------------+------------+---------------+-----------------+----------------------+----------------------+----------------------+---------------------+
```
* Loading the train dataset can take anywhere from 15 minutes to 2 hours
* For best results, please login to Shamu and access a GPU node
# Continue loading cells
* You will begin training once you get past ==== INIT MODEL ====
