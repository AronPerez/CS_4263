# CS-4263-Project
CS-4263-001-Spring-2021-Deep Learning


# How to run
* To get access to the full dataset, please use the following [Google Drive link](https://drive.google.com/drive/folders/1cC6BD46FMK9-UryYAKEeHRXG7qYdNTBL?usp=sharing)
* Load the notebook into [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)
* Select visual branch
* Set your runtime to GPU or TPU for best performance
* Be sure to connect the new drive you have access to with the following
```
from google.colab import drive
drive.mount('/content/drive')
```
# Config file
* The config file is located under ``/content/drive/MyDrive/lyft-motion-prediction-autonomous-vehicles/config.yaml`` 
# Submission file
* Set your submission file to visualize
For testing, use `submission_B6_2.csv` as it has the best predictions
```python
# set env variable for data
df_sub = pd.read_csv('/content/drive/MyDrive/lyft-motion-prediction-autonomous-vehicles/results/batches/submission_B6_2.csv')
df_sub = df_sub.set_index(['timestamp', 'track_id'])
```
# Load test dataset
This is already set with the following, nothing neesd to be changed
```python
# set env variable for data
DIR_INPUT = "/content/drive/MyDrive/lyft-motion-prediction-autonomous-vehicles/"
cfg = load_config_data("/content/drive/MyDrive/lyft-motion-prediction-autonomous-vehicles/config.yaml")
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)
```
* Rasterize the train dataset
* Continue running cells until you are at the ``Here are predictions`` section
Here you can play with the data and see different views
# agent_semantic_dataset vs agent_satellite_dataset
* agent_semantic_dataset allows you to see the raw view

![semantic data](https://raw.githubusercontent.com/AronPerez/CS_4263/visual/README_images/Plot1.png)
* agent_satellite_dataset allows you to see satellite image overlay with the prediction points

![semantic data](https://raw.githubusercontent.com/AronPerez/CS_4263/visual/README_images/Plot4.png)
