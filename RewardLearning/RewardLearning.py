import numpy as np
from models.NASNet import NASNet
from models.FCN import FCN
from models.T_FCN import T_FCN
from models.Attention_RNN import Attention_RNN
from models.Transformer import Transformer
from models.DANN import DANN
from models.ADDA import ADDA
from models.T_FCN_ADDA import T_FCN_ADDA


def LearnRewards(Task):
    models = ['NASNet', 'FCN', 'T_FCN', 'Attention_RNN', 'Transformer', 'DANN', 'ADDA', 'T_FCN_ADDA']

    for model in models:
        My_Model_Logs_DIR = './data/' + Task + '/' + model + '/'
        My_Model_Weights = './data/' + Task + '/' + model + '/' + model + '_Weights.h5'
        Train_Data_Dir = './data/TasksData/' + Task + '/Train/'
        Test_Data_Dir = './data/TasksData/' + Task + '/unseen/'
        Embed_Model_Weights = './data/' + Task + '/FCN/FCN_Weights.h5'

        if model is 'NASNet': My_Model = NASNet(Logs_DIR=My_Model_Logs_DIR)
        if model is 'FCN' : My_Model = FCN(Logs_DIR=My_Model_Logs_DIR)
        if model is 'T_FCN' : My_Model = T_FCN(Logs_DIR=My_Model_Logs_DIR)
        if model is 'Attention_RNN' : My_Model = Attention_RNN(Logs_DIR=My_Model_Logs_DIR, embed_model_weights=Embed_Model_Weights)
        if model is 'Transformer' : My_Model = Transformer(Logs_DIR=My_Model_Logs_DIR, embed_model_weights=Embed_Model_Weights)
        if model is 'DANN' : My_Model = DANN(Logs_DIR=My_Model_Logs_DIR)
        if model is 'ADDA' : My_Model = ADDA(Logs_DIR=My_Model_Logs_DIR)
        if model is 'T_FCN_ADDA' : My_Model = T_FCN_ADDA(Logs_DIR=My_Model_Logs_DIR)

        My_Model.Create_Train_Data(DATADIR=Train_Data_Dir, SAVEDIR=My_Model_Logs_DIR)
        My_Model.Create_Test_Data(DATADIR=Test_Data_Dir, SAVEDIR=My_Model_Logs_DIR)

        My_Model.x_train = np.load(My_Model_Logs_DIR + model + '_x_train.npy')
        My_Model.y_train = np.load(My_Model_Logs_DIR + model + '_y_train.npy')
        My_Model.x_test = np.load(My_Model_Logs_DIR + model + '_x_test.npy')
        My_Model.y_test = np.load(My_Model_Logs_DIR + model + '_y_test.npy')

        My_Model.Fit()
        My_Model.Load_Model(My_Model_Weights)
        My_Model.Evaluate()

LearnRewards('Task1')
LearnRewards('Task2')
LearnRewards('Task3')
LearnRewards('Task4')
LearnRewards('Task5')
