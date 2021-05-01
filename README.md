# CS-4263-Project
CS-4263-001-Spring-2021-Deep Learning


# How to run
* To get access to the full dataset, please use the following [Google Drive link](https://drive.google.com/drive/folders/1cC6BD46FMK9-UryYAKEeHRXG7qYdNTBL?usp=sharing)
* Load the notebook into [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)
* Select train branch
* Set your runtime to GPU or TPU for best performance
* Be sure to connect the new drive you have access to with the following
```
from google.colab import drive
drive.mount('/content/drive')
```
* Continue running cells until you hit the train dataset section
# Load train dataset
* This is done via the following
```python
train_cfg = cfg["train_data_loader"]
rasterizer = build_rasterizer(cfg, dm)

#perturb_prob = cfg["train_data_loader"]["perturb_probability"]
#perturbation = AckermanPerturbation(random_offset_generator=GaussianRandomGenerator(mean=np.array([0.0, 0.0]), std=np.array([1.0, np.pi / 6])),
                                    #perturb_prob=train_cfg["perturb_probability"],)

train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()

train_dataset_2 = AgentDataset(cfg, train_zarr, rasterizer)
train_dataloader_2 = DataLoader(train_dataset_2, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
                                num_workers=train_cfg["num_workers"])

print("==================================TRAIN DATA==================================")
print(train_dataset_2)
```
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
# Training
* This takes place in ``== Train Model ==``
* Your model will be saved every 200 steps
* The default step count is 1000 due to limitations set by the environment
