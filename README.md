# Reward Learning

This project contains the implementation of eight different classifier used to learn the rewards of five different robotic manipulation tasks:

| Model | Description |
|:---:|:---:|
| `NASNet` | Fully Connected Neural Net that extracts features using NASNet model and feed them to Feed-Forward Network |
| `FCN` | Fully Convolutional Neural Net |
| `T-FCN` | Time-Based Fully Convolutional Neural Net |
| `Attention-RNN` | Attention-Based Encoder-Decoder |
| `Transformer` | Transformer Network |
| `DANN` | Domain-Adversarial Neural Network |
| `ADDA` | Adversarial-Discriminative Domain Adaptation |
| `T-FCN-ADDA` | Timing-Based and Domain-Based Fully Convolutional Net |

# Dataset
The dataset of the five different tasks are avaliable in the following links:

[Task1](https://universityoflincoln-my.sharepoint.com/:u:/g/personal/17682200_students_lincoln_ac_uk/ERCK459EU55EmITx6YCFUjQB-Pucw-77d2MN4Hw5TS1wkw?e=BfLHUw)

[Task2](https://universityoflincoln-my.sharepoint.com/:u:/g/personal/17682200_students_lincoln_ac_uk/EeoT032lyp5Eovv-08TddmQBQ-a2HQZTDHbhovTPBnYpeQ?e=QoBvBf)

[Task3](https://universityoflincoln-my.sharepoint.com/:u:/g/personal/17682200_students_lincoln_ac_uk/EZuHO_TvI9tCopMHT7m_LIQBthC1KUZ1uTLKpxx7PCTGAQ?e=pqTVMT)

[Task4](https://universityoflincoln-my.sharepoint.com/:u:/g/personal/17682200_students_lincoln_ac_uk/EbPoLtXOfxNAsPdJwPsgersBEZs9Mpb96FW2DUuRgXan3Q?e=HQjjxN)

[Task5](https://universityoflincoln-my.sharepoint.com/:u:/g/personal/17682200_students_lincoln_ac_uk/EcLU4RJS7nFOkIl1CsBUHXABZN8GGEDks4AMfq6S64qo9g?e=5ZtYKC)

The following link contains the data of the unseen test conditions:

[Unseen Conditions](https://universityoflincoln-my.sharepoint.com/:u:/g/personal/17682200_students_lincoln_ac_uk/ERzPa0b2jBBJv1Gk7LGtXPEBcEFCZVtLQEQ6ZpLf6kuJsw?e=KJadfb)

# Installation

```bash
cd RewardLearning
pip install -e .
```

# Usage
#### 1. Import Libraries:
```bash
import numpy as np
from models.NASNet import NASNet
from models.FCN import FCN
from models.T_FCN import T_FCN
from models.Attention_RNN import Attention_RNN
from models.Transformer import Transformer
from models.DANN import DANN
from models.ADDA import ADDA
from models.T_FCN_ADDA import T_FCN_ADDA
```

#### 2. Define the directories:
```bash
My_Model_Logs_DIR = ...
My_Model_Weights = ...
Train_Data_Dir = ...
Test_Data_Dir = ...
Embed_Model_Weights = ... # This is used in 'Attention_RNN' and 'Transformer' only.
```

#### 3. Select the model:
```bash
if model is 'NASNet': My_Model = NASNet(Logs_DIR=My_Model_Logs_DIR)
if model is 'FCN' : My_Model = FCN(Logs_DIR=My_Model_Logs_DIR)
if model is 'T_FCN' : My_Model = T_FCN(Logs_DIR=My_Model_Logs_DIR)
if model is 'Attention_RNN' : My_Model = Attention_RNN(Logs_DIR=My_Model_Logs_DIR, embed_model_weights=Embed_Model_Weights)
if model is 'Transformer' : My_Model = Transformer(Logs_DIR=My_Model_Logs_DIR, embed_model_weights=Embed_Model_Weights)
if model is 'DANN' : My_Model = DANN(Logs_DIR=My_Model_Logs_DIR)
if model is 'ADDA' : My_Model = ADDA(Logs_DIR=My_Model_Logs_DIR)
if model is 'T_FCN_ADDA' : My_Model = T_FCN_ADDA(Logs_DIR=My_Model_Logs_DIR)
```

#### 4. Create and load the train and test data:
```bash
My_Model.Create_Train_Data(DATADIR=Train_Data_Dir, SAVEDIR=My_Model_Logs_DIR)
My_Model.Create_Test_Data(DATADIR=Test_Data_Dir, SAVEDIR=My_Model_Logs_DIR)

My_Model.x_train = np.load(My_Model_Logs_DIR + model + '_x_train.npy')
My_Model.y_train = np.load(My_Model_Logs_DIR + model + '_y_train.npy')
My_Model.x_test = np.load(My_Model_Logs_DIR + model + '_x_test.npy')
My_Model.y_test = np.load(My_Model_Logs_DIR + model + '_y_test.npy')
```

#### 5. Train the model:
```bash
My_Model.Fit()
```

#### 6. Evaluate the model:
```bash
My_Model.Evaluate()
```



#### NOTE: More details will be available soon!!!
