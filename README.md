# CS-4263-Project
CS-4263-001-Spring-2021-Deep Learning


# How to run
* To get access to the full dataset, please use the following [Google Drive link](https://drive.google.com/drive/folders/1cC6BD46FMK9-UryYAKEeHRXG7qYdNTBL?usp=sharing)
* Load the notebook into [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)
* Select eval branch
* Set your runtime to GPU or TPU for best performance
* Be sure to connect the new drive you have access to with the following
```
from google.colab import drive
drive.mount('/content/drive')
```
# Select a model
* Multiple modes exist
For best usage, please use ``model_multi_update_lyft_public.pth``
* Load this by setting the following:
```python
cfg["model_params"]["weight_path"] = "/content/drive/MyDrive/lyft-motion-prediction-autonomous-vehicles/models/model_multi_update_lyft_public.pth"
```
after:
```python
# set env variable for data
# Aaron P DIR_INPUT = "/content/Dataset/lyft-motion-prediction-autonomous-vehicles/"
# Default "./content/drive/MyDrive/lyft-motion-prediction-autonomous-vehicles/"
# Config.yaml need to be manually uploaded to /content/
!pwd
DIR_INPUT = "/content/drive/MyDrive/lyft-motion-prediction-autonomous-vehicles/"
cfg = load_config_data("/content/drive/MyDrive/lyft-motion-prediction-autonomous-vehicles/config.yaml")
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)
```
# Load test dataset
* We will be making predictions against this with our model against the test dataset
# Predictions
* Once you hit ``Eval Loop`` you will begin making predictions with the model
* Once done, you will save the predictions CSV to your Google Colab insance
Please rename it here
```python
pred_path = './submission_r1_1000i.csv'
write_pred_csv(pred_path,
           timestamps=np.concatenate(timestamps),
           track_ids=np.concatenate(agent_ids),
           coords=np.concatenate(future_coords_offsets_pd),
           confs = np.concatenate(confidences_list)
          )
```

