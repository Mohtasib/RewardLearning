import numpy as np
from models.NASNet import NASNet
from models.FCN import FCN
from models.T_FCN import T_FCN
from models.Attention_RNN import Attention_RNN
from models.Transformer import Transformer
from models.DANN import DANN
from models.ADDA import ADDA
from models.T_FCN_ADDA import T_FCN_ADDA


"""
######################### NASNet #########################

NASNet_Logs_DIR = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/NASNet/'
Train_Data_Dir = 'C:/Users/computing/Desktop/test_data/Task5/Train/'
Test_Data_Dir = 'C:/Users/computing/Desktop/test_data/Task5/unseen/'
Test_Images = 'C:/Users/computing/Desktop/test_data/Task5/unseen/Data/Success/'
Test_Image = 'C:/Users/computing/Desktop/test_data/Task5/unseen/Data/Success/Im0100.png'
NASNet_Weights = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/NASNet/NASNet_Weights.h5'

MyNASNet = NASNet(Logs_DIR=NASNet_Logs_DIR)
print('MyNASNet.version = ', MyNASNet.version())
MyNASNet.Create_Train_Data(DATADIR=Train_Data_Dir, SAVEDIR=NASNet_Logs_DIR)
MyNASNet.Create_Test_Data(DATADIR=Test_Data_Dir, SAVEDIR=NASNet_Logs_DIR)

MyNASNet.x_train = np.load(NASNet_Logs_DIR + 'NASNet_x_train.npy')
MyNASNet.y_train = np.load(NASNet_Logs_DIR + 'NASNet_y_train.npy')
MyNASNet.x_test = np.load(NASNet_Logs_DIR + 'NASNet_x_test.npy')
MyNASNet.y_test = np.load(NASNet_Logs_DIR + 'NASNet_y_test.npy')

MyNASNet.Fit(epochs=2)
MyNASNet.Load_Model(NASNet_Weights)
MyNASNet.Evaluate()

data = MyNASNet.Load_Img(Test_Image)
print('Test Image = ', MyNASNet.Predict(data))

data = MyNASNet.Load_Images(Test_Images)
print('Test Images = ', MyNASNet.Predict(data))


########################### FCN ##########################

FCN_Logs_DIR = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/FCN/'
Train_Data_Dir = 'C:/Users/computing/Desktop/test_data/Task5/Train/'
Test_Data_Dir = 'C:/Users/computing/Desktop/test_data/Task5/unseen/'
Test_Images = 'C:/Users/computing/Desktop/test_data/Task5/unseen/Data/Success/'
Test_Image = 'C:/Users/computing/Desktop/test_data/Task5/unseen/Data/Success/Im0100.png'
FCN_Weights = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/FCN/FCN_Weights.h5'

MyFCN = FCN(Logs_DIR=FCN_Logs_DIR)
print('MyFCN.version = ', MyFCN.version())
MyFCN.Create_Train_Data(DATADIR=Train_Data_Dir, SAVEDIR=FCN_Logs_DIR)
MyFCN.Create_Test_Data(DATADIR=Test_Data_Dir, SAVEDIR=FCN_Logs_DIR)

MyFCN.x_train = np.load(FCN_Logs_DIR + 'FCN_x_train.npy')
MyFCN.y_train = np.load(FCN_Logs_DIR + 'FCN_y_train.npy')
MyFCN.x_test = np.load(FCN_Logs_DIR + 'FCN_x_test.npy')
MyFCN.y_test = np.load(FCN_Logs_DIR + 'FCN_y_test.npy')

MyFCN.Fit(epochs=2)
MyFCN.Load_Model(FCN_Weights)
MyFCN.Evaluate()

data = MyFCN.Load_Img(Test_Image)
print('Test Image = ', MyFCN.Predict(data))

data = MyFCN.Load_Images(Test_Images)
print('Test Images = ', MyFCN.Predict(data))


########################### T_FCN ##########################

T_FCN_Logs_DIR = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/T_FCN/'
Train_Data_Dir = 'C:/Users/computing/Desktop/test_data/Task5/Train/'
Test_Data_Dir = 'C:/Users/computing/Desktop/test_data/Task5/unseen/'
Test_Images = 'C:/Users/computing/Desktop/test_data/Task5/unseen/Data/Success/'
Test_Image = 'C:/Users/computing/Desktop/test_data/Task5/unseen/Data/Success/Im0100.png'
T_FCN_Weights = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/T_FCN/T_FCN_Weights.h5'

MyT_FCN = T_FCN(Logs_DIR=T_FCN_Logs_DIR)
#print('MyT_FCN.version = ', MyT_FCN.version())
#MyT_FCN.Create_Train_Data(DATADIR=Train_Data_Dir, SAVEDIR=T_FCN_Logs_DIR)
#MyT_FCN.Create_Test_Data(DATADIR=Test_Data_Dir, SAVEDIR=T_FCN_Logs_DIR)

MyT_FCN.x_train = np.load(T_FCN_Logs_DIR + 'T_FCN_x_train.npy')
MyT_FCN.y_train = np.load(T_FCN_Logs_DIR + 'T_FCN_y_train.npy')
MyT_FCN.x_test = np.load(T_FCN_Logs_DIR + 'T_FCN_x_test.npy')
MyT_FCN.y_test = np.load(T_FCN_Logs_DIR + 'T_FCN_y_test.npy')

#MyT_FCN.Fit(epochs=2)
MyT_FCN.Load_Model(T_FCN_Weights)
MyT_FCN.Evaluate()

data = MyT_FCN.Load_Img(Test_Image)
print('Test Image = ', MyT_FCN.Predict(data))

data = MyT_FCN.Load_Images(Test_Images)
print('Test Images = ', MyT_FCN.Predict(data))


######################### Attention_RNN #########################

Attention_RNN_Logs_DIR = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/Attention_RNN/'
Attention_RNN_embed_model_weights = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/Attention_RNN/Attention_RNN_Embed_Model_Weights.h5'
Train_Data_Dir = 'C:/Users/computing/Desktop/test_data/Task5/Train/'
Test_Data_Dir = 'C:/Users/computing/Desktop/test_data/Task5/unseen/'
Test_Images = 'C:/Users/computing/Desktop/test_data/Task5/unseen/Data/Success/'
Test_Image = 'C:/Users/computing/Desktop/test_data/Task5/unseen/Data/Success/Im0100.png'
Attention_RNN_Weights = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/Attention_RNN/Attention_RNN_Weights.h5'

MyAttention_RNN = Attention_RNN(    Logs_DIR=Attention_RNN_Logs_DIR,
                                    embed_model_weights=Attention_RNN_embed_model_weights)
print('MyAttention_RNN.version = ', MyAttention_RNN.version())
#MyAttention_RNN.Create_Train_Data(DATADIR=Train_Data_Dir, SAVEDIR=Attention_RNN_Logs_DIR)
#MyAttention_RNN.Create_Test_Data(DATADIR=Test_Data_Dir, SAVEDIR=Attention_RNN_Logs_DIR)

MyAttention_RNN.x_train = np.load(Attention_RNN_Logs_DIR + 'Attention_RNN_x_train.npy')
MyAttention_RNN.y_train = np.load(Attention_RNN_Logs_DIR + 'Attention_RNN_y_train.npy')
MyAttention_RNN.x_test = np.load(Attention_RNN_Logs_DIR + 'Attention_RNN_x_test.npy')
MyAttention_RNN.y_test = np.load(Attention_RNN_Logs_DIR + 'Attention_RNN_y_test.npy')

#MyAttention_RNN.Fit(epochs=2)
MyAttention_RNN.Load_Model(Attention_RNN_Weights)
MyAttention_RNN.Evaluate()

data = MyAttention_RNN.Load_Img(Test_Image)
print('Test Image = ', MyAttention_RNN.Predict(data))


######################### Transformer #########################

Transformer_Logs_DIR = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/Transformer/'
Transformer_embed_model_weights = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/Transformer/Transformer_Embed_Model_Weights.h5'
Train_Data_Dir = 'C:/Users/computing/Desktop/test_data/Task5/Train/'
Test_Data_Dir = 'C:/Users/computing/Desktop/test_data/Task5/unseen/'
Test_Images = 'C:/Users/computing/Desktop/test_data/Task5/unseen/Data/Success/'
Test_Image = 'C:/Users/computing/Desktop/test_data/Task5/unseen/Data/Success/Im0100.png'
Transformer_Weights = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/Transformer/Transformer_Weights.h5'

MyTransformer = Transformer(    Logs_DIR=Transformer_Logs_DIR,
                                embed_model_weights=Transformer_embed_model_weights)
print('MyTransformer.version = ', MyTransformer.version())
#MyTransformer.Create_Train_Data(DATADIR=Train_Data_Dir, SAVEDIR=Transformer_Logs_DIR)
#MyTransformer.Create_Test_Data(DATADIR=Test_Data_Dir, SAVEDIR=Transformer_Logs_DIR)

MyTransformer.x_train = np.load(Transformer_Logs_DIR + 'Transformer_x_train.npy')
MyTransformer.y_train = np.load(Transformer_Logs_DIR + 'Transformer_y_train.npy')
MyTransformer.x_test = np.load(Transformer_Logs_DIR + 'Transformer_x_test.npy')
MyTransformer.y_test = np.load(Transformer_Logs_DIR + 'Transformer_y_test.npy')

#MyTransformer.Fit(epochs=100)
MyTransformer.Load_Model(Transformer_Weights)
MyTransformer.Evaluate()

data = MyTransformer.Load_Img(Test_Image)
print('Test Image = ', MyTransformer.Predict(data))


########################### DANN ##########################

DANN_Logs_DIR = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/DANN/'
Train_Data_Dir = 'C:/Users/computing/Desktop/test_data/Task5/Train/'
Test_Data_Dir = 'C:/Users/computing/Desktop/test_data/Task5/unseen/'
Test_Images = 'C:/Users/computing/Desktop/test_data/Task5/unseen/Data/Success/'
Test_Image = 'C:/Users/computing/Desktop/test_data/Task5/unseen/Data/Success/Im0100.png'
DANN_Weights = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/DANN/DANN_Weights.h5'

MyDANN = DANN(Logs_DIR=DANN_Logs_DIR)
#print('MyDANN.version = ', MyDANN.version())
#MyDANN.Create_Train_Data(DATADIR=Train_Data_Dir, SAVEDIR=DANN_Logs_DIR)
#MyDANN.Create_Test_Data(DATADIR=Test_Data_Dir, SAVEDIR=DANN_Logs_DIR)

MyDANN.x_train = np.load(DANN_Logs_DIR + 'DANN_x_train.npy')
MyDANN.y_train = np.load(DANN_Logs_DIR + 'DANN_y_train.npy')
MyDANN.x_test = np.load(DANN_Logs_DIR + 'DANN_x_test.npy')
MyDANN.y_test = np.load(DANN_Logs_DIR + 'DANN_y_test.npy')

#MyDANN.Fit(epochs=2)
MyDANN.Load_Model(DANN_Weights)
MyDANN.Evaluate()

data = MyDANN.Load_Img(Test_Image)
print('Test Image = ', MyDANN.Predict(data))

data = MyDANN.Load_Images(Test_Images)
print('Test Images = ', MyDANN.Predict(data))


########################### ADDA ##########################

ADDA_Logs_DIR = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/ADDA/'
Train_Data_Dir = 'C:/Users/computing/Desktop/test_data/Task5/Train/'
Test_Data_Dir = 'C:/Users/computing/Desktop/test_data/Task5/unseen/'
Test_Images = 'C:/Users/computing/Desktop/test_data/Task5/unseen/Data/Success/'
Test_Image = 'C:/Users/computing/Desktop/test_data/Task5/unseen/Data/Success/Im0100.png'
ADDA_Weights = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/ADDA/ADDA_Weights.h5'

MyADDA = ADDA(Logs_DIR=ADDA_Logs_DIR)
#print('MyADDA.version = ', MyADDA.version())
#MyADDA.Create_Train_Data(DATADIR=Train_Data_Dir, SAVEDIR=ADDA_Logs_DIR)
#MyADDA.Create_Test_Data(DATADIR=Test_Data_Dir, SAVEDIR=ADDA_Logs_DIR)

MyADDA.x_train = np.load(ADDA_Logs_DIR + 'ADDA_x_train.npy')
MyADDA.y_train = np.load(ADDA_Logs_DIR + 'ADDA_y_train.npy')
MyADDA.x_test = np.load(ADDA_Logs_DIR + 'ADDA_x_test.npy')
MyADDA.y_test = np.load(ADDA_Logs_DIR + 'ADDA_y_test.npy')

#MyADDA.Fit(epochs=2)
MyADDA.Load_Model(ADDA_Weights)
MyADDA.Evaluate()

data = MyADDA.Load_Img(Test_Image)
print('Test Image = ', MyADDA.Predict(data))

data = MyADDA.Load_Images(Test_Images)
print('Test Images = ', MyADDA.Predict(data))

"""
########################### T_FCN_ADDA ##########################

T_FCN_ADDA_Logs_DIR = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/T_FCN_ADDA/'
Train_Data_Dir = 'C:/Users/computing/Desktop/test_data/Task5/Train/'
Test_Data_Dir = 'C:/Users/computing/Desktop/test_data/Task5/unseen/'
Test_Images = 'C:/Users/computing/Desktop/test_data/Task5/unseen/Data/Success/'
Test_Image = 'C:/Users/computing/Desktop/test_data/Task5/unseen/Data/Success/Im0100.png'
T_FCN_ADDA_Weights = 'C:/Users/computing/OneDrive - University of Lincoln/MyProjects/RewardLearning/RewardLearning/Test/T_FCN_ADDA/T_FCN_ADDA_Weights.h5'

MyT_FCN_ADDA = T_FCN_ADDA(Logs_DIR=T_FCN_ADDA_Logs_DIR)
print('MyT_FCN_ADDA.version = ', MyT_FCN_ADDA.version())
#MyT_FCN_ADDA.Create_Train_Data(DATADIR=Train_Data_Dir, SAVEDIR=T_FCN_ADDA_Logs_DIR)
#MyT_FCN_ADDA.Create_Test_Data(DATADIR=Test_Data_Dir, SAVEDIR=T_FCN_ADDA_Logs_DIR)

MyT_FCN_ADDA.x_train = np.load(T_FCN_ADDA_Logs_DIR + 'T_FCN_ADDA_x_train.npy')
MyT_FCN_ADDA.y_train = np.load(T_FCN_ADDA_Logs_DIR + 'T_FCN_ADDA_y_train.npy')
MyT_FCN_ADDA.x_test = np.load(T_FCN_ADDA_Logs_DIR + 'T_FCN_ADDA_x_test.npy')
MyT_FCN_ADDA.y_test = np.load(T_FCN_ADDA_Logs_DIR + 'T_FCN_ADDA_y_test.npy')

#MyT_FCN_ADDA.Fit(epochs=2)
MyT_FCN_ADDA.Load_Model(T_FCN_ADDA_Weights)
MyT_FCN_ADDA.Evaluate()

data = MyT_FCN_ADDA.Load_Img(Test_Image)
print('Test Image = ', MyT_FCN_ADDA.Predict(data))

data = MyT_FCN_ADDA.Load_Images(Test_Images)
print('Test Images = ', MyT_FCN_ADDA.Predict(data))
