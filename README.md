# Reward Learning

This project two Models to learn rewards from real demonstrations, In this project the models receives images as an environment state and classify these images into Success or Non-success completetion of the task in hand. The models were tested on five different robotic manipulation tasks. Furthermore, the models were trained with and without data augmentation. The models are:

### FFN
This is a FeedForward Network.

### FCN
This is a Fully Convolutional Network

# Dataset
The dataset of the five different tasks are avaliable in the following links:

[Task1](https://universityoflincoln-my.sharepoint.com/:u:/g/personal/17682200_students_lincoln_ac_uk/ERCK459EU55EmITx6YCFUjQBQx-CNTQfu_B_HN72vNVIig?e=kFHW3V)

[Task2](https://universityoflincoln-my.sharepoint.com/:u:/g/personal/17682200_students_lincoln_ac_uk/EeoT032lyp5Eovv-08TddmQB5G_nxR97CCt3dGfxNfe0PQ?e=agSXAQ)

[Task3](https://universityoflincoln-my.sharepoint.com/:u:/g/personal/17682200_students_lincoln_ac_uk/EZuHO_TvI9tCopMHT7m_LIQBdeckREhz_V5wNK9-ySHGcQ?e=LnwPv4)

[Task4](https://universityoflincoln-my.sharepoint.com/:u:/g/personal/17682200_students_lincoln_ac_uk/EbPoLtXOfxNAsPdJwPsgersBpvVgYPCtKDGggCj_YJAVwQ?e=yMWhWT)

[Task5](https://universityoflincoln-my.sharepoint.com/:u:/g/personal/17682200_students_lincoln_ac_uk/EcLU4RJS7nFOkIl1CsBUHXABkhYqqenPcb8xeG7TiLC-mQ?e=pf0JC9)

The following link contains the data of the unseen test conditions:

[Unseen Conditions](https://universityoflincoln-my.sharepoint.com/:u:/g/personal/17682200_students_lincoln_ac_uk/ERzPa0b2jBBJv1Gk7LGtXPEBprnPA4v7X1uNGjanjJvJiA?e=b7jntD)

Note: The _data_ directory structure should look like this:
```bash
.
+-- data
|     +-- Task1
|     |    +-- 5Demos 
|     |    |     +-- Demo1
|     |    |     +-- Demo2
|     |    |     +-- Demo3
|     |    |     +-- Demo4
|     |    |     +-- Demo5
|     |    +-- Test 
|     |          +-- Demo6
|     |          +-- Demo7
|     |          +-- Demo8
|     |          +-- Demo9
|     |          +-- Demo10
|     +-- Task2
|     |    +-- 5Demos 
|     |    |     +-- Demo1
|     |    |     +-- Demo2
|     |    |     +-- Demo3
|     |    |     +-- Demo4
|     |    |     +-- Demo5
|     |    +-- Test 
|     |          +-- Demo6
|     |          +-- Demo7
|     |          +-- Demo8
|     |          +-- Demo9
|     |          +-- Demo10
|     +-- ...
+-- ...
```

# Installation

```bash
cd RewardLearning
pip install -e .
```

# Usage
After selecting the correct config file:
#### 1. Import Libraries:
```bash
from models.FCN_Classifier import FCN_Classifier
from models.FFN_Classifier import FFN_Classifier
from dataset_creator.DatasetCreator import DatasetCreator
from data_utils.DataAugmentor import DataAugmentor
from data_utils.ImgEmbed import ImageEmbedding
from config import cfg
```

#### 2. Generate the Augmented Data:
```bash
MyDataAugmentor = DataAugmentor(cfg=cfg)
MyDataAugmentor.Generate()
```

#### 3. Create the Dataset:
```bash
MyDataCreator = DatasetCreator(cfg=cfg)
MyDataCreator.CreateTrainDATA()
MyDataCreator.CreateTestDATA()
```

#### 4. Create the model and train it:
```bash
MyClassifier = FFN_Classifier(mode="Train", cfg=cfg)
MyClassifier.epochs = 2
history = MyClassifier.Train()
MyClassifier.Plot_Model_History(history)
MyClassifier.Confusion_Report()
```

#### 5. Use the model to predict the label of an image:
```bash
Img = MyClassifier.LoadImg("./sample_data/test.png")
prediction = MyClassifier.Predict(Img)
print("Prediction = ", prediction)
```

#### 6. Use the model to predict the labels of some images:
```bash
Img = MyClassifier.LoadImages(DATA_PATH="./sample_data/")
prediction = MyClassifier.Predict(Img)
print("Prediction = ", prediction)
```



#### NOTE: More details will be available soon!!!
