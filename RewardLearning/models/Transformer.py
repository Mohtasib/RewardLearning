
"""
The original transformer implmentation was taken from:
https://github.com/CyberZHG/keras-transformer
"""
import os
import numpy as np
import random
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from models.BaseTransformer import get_model

from keras.models import Model
from keras.models import load_model
from keras.layers import Flatten

from PIL import Image

class ImageEmbedding():
    def __init__(self, embed_model_weights=None, embed_model_output_layer="conv2d_5"):
        self.embed_model_weights = embed_model_weights
        self.embed_model_output_layer = embed_model_output_layer
        self.model = self.CreateEmbedModel(MODEL_PATH=self.embed_model_weights, Intermediate_Layer_Name=self.embed_model_output_layer)

    def CreateEmbedModel(self, MODEL_PATH, Intermediate_Layer_Name):
        model = load_model(MODEL_PATH)  # create the original model

        intermediate_layer_model = Model(inputs=model.input,
                                        outputs=model.get_layer(Intermediate_Layer_Name).output)
        embeds = Flatten()(intermediate_layer_model.output)
        return Model(inputs = intermediate_layer_model.input, outputs = embeds)
    
    def LoadImg(self, 
                IMG_PATH, 
                IMG_SHAPE=[160, 160, 3]):
        img_width = IMG_SHAPE[0]
        img_height = IMG_SHAPE[1]
        num_channels = IMG_SHAPE[2]
        img = Image.open(IMG_PATH)  # this is a PIL image
        resized = img.resize((img_width, img_height)) # Resize the image
        reshaped = np.reshape(resized, (-1, img_width, img_height, num_channels))   # this is a Numpy array with shape (3, 150, 150)
        return reshaped/255

    def Embed_An_Img(self, Img):
        return self.model.predict(Img)[0]
                
    def LoadAndEmbed(self, ImgPath, IMG_SHAPE=[160, 160, 3]):
        Img = self.LoadImg(ImgPath, IMG_SHAPE)
        return self.Embed_An_Img(Img)
      
class Transformer():
    def __init__(   self, 
                    Logs_DIR, 
                    embed_model_weights, 
                    tokens=4, # tokens = {0:Non-success, 1:Success, 2:START, 3:STOP}
                    seq_n_timesteps=10,
                    seq_n_features_in=200,
                    encoder_num=3,
                    decoder_num=3,
                    head_num=4,
                    hidden_dim=100,
                    dropout_rate=0.05,
                    Print_Model_Summary=False):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.tokens = tokens
        self.seq_n_timesteps = seq_n_timesteps
        self.img_embedding_len = seq_n_features_in
        self.seq_n_features_in = seq_n_features_in
        self.encoder_num = encoder_num
        self.decoder_num = decoder_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.model = None
        self.ImgEmbed = None
        self.Logs_DIR = Logs_DIR
        os.makedirs(self.Logs_DIR, exist_ok=True)
        self.embed_model_weights = embed_model_weights
        self.Print_Model_Summary = Print_Model_Summary
        self.Weights = self.Logs_DIR + 'Transformer_Weights.h5'
        self.Model_Summary = self.Logs_DIR + 'Transformer_Summary.txt'
        self.model = self.Create_Model(Print_Model_Summary=self.Print_Model_Summary)   

    def Create_Data(self, DATADIR, SAVEDIR_x_data, SAVEDIR_y_data, IMG_SHAPE, SeqLen, ImgEmbedLen):
        # Extract tha data from the demos files:
        CATEGORIES = ["Non-success", "Success"]
        EncoderData = []
        DecoderData = []
        EncoderDataset = []
        DecoderDataset = []
        demos = os.listdir(DATADIR) #  get the list of folders in this directory
        for d in range(0, len(demos)):
            print("Working on", demos[d])
            TempEncoderData = []
            TempDecoderData = []
            for category in CATEGORIES:
                DATA_PATH = os.path.join(DATADIR + demos[d], category)
                class_num = CATEGORIES.index(category)
                for i in os.listdir(DATA_PATH):
                    if self.ImgEmbed is None: 
                        self.ImgEmbed = ImageEmbedding(self.embed_model_weights)
                    TempEncoderData.append(self.ImgEmbed.LoadAndEmbed(os.path.join(DATA_PATH, i), IMG_SHAPE))
                    TempDecoderData.append(class_num)
            EncoderData.append(TempEncoderData)
            DecoderData.append(TempDecoderData)

        # Padding and preparing the data in the right shape:
        EncoderData = np.array(EncoderData)
        DecoderData = np.array(DecoderData)

        NumOfEncoderSeqs = sum(len(line) for line in EncoderData[:])
        NumOfDecoderSeqs = sum(len(line) for line in DecoderData[:])

        EncoderInput = np.zeros((NumOfEncoderSeqs, SeqLen, ImgEmbedLen), dtype='float32')
        DecoderInput = np.zeros((NumOfDecoderSeqs, SeqLen+1, 1), dtype='float32')
        DecoderOutput = np.zeros((NumOfDecoderSeqs, SeqLen+1, 1), dtype='float32')

        count = 0
        for i in range(0, len(EncoderData)):
            for j in range(0,len(EncoderData[i])):
                if j+1 < SeqLen :
                    # Seq:
                    for x in range(0,j+1):
                        for k in range(0,len(EncoderData[i][j])):
                            EncoderInput[count][SeqLen+x-j-1][k] = float(EncoderData[i][x][k]) # padding zeros at the beginning
                if j+1 >= SeqLen:
                    # Seq:
                    for x in range(0,SeqLen):
                        for k in range(0,len(EncoderData[i][j])):
                            EncoderInput[count][x][k] = float(EncoderData[i][j+x+1-SeqLen][k])
                count += 1
        
        START = 2
        END = 3

        count = 0
        for i in range(0, len(DecoderData)):
            for j in range(0,len(DecoderData[i])):
                if j+1 < SeqLen :
                    # Start of Seq:
                    DecoderInput[count][0] = float(START)
                    # Seq:
                    for x in range(0,j+1):
                        DecoderInput[count][SeqLen+x-j] = float(DecoderData[i][x]) # padding zeros at the beginning
                if j+1 >= SeqLen:
                    # Start of Seq:
                    DecoderInput[count][0] = float(START)
                    # Seq:
                    for x in range(0,SeqLen):
                        DecoderInput[count][x+1] = float(DecoderData[i][j+x+1-SeqLen])
                count += 1

        count = 0
        for i in range(0, len(DecoderData)):
            for j in range(0,len(DecoderData[i])):
                if j+1 < SeqLen :
                    # Seq:
                    for x in range(0,j+1):
                        DecoderOutput[count][SeqLen+x-j-1] = float(DecoderData[i][x]) # padding zeros at the beginning
                    # End of Seq:
                    DecoderOutput[count][-1] = float(END)
                if j+1 >= SeqLen:
                    # Seq:
                    for x in range(0,SeqLen):
                        DecoderOutput[count][x] = float(DecoderData[i][j+x+1-SeqLen])
                    # End of Seq:
                    DecoderOutput[count][-1] = float(END)
                count += 1

        EncoderDataset = EncoderInput
        DecoderDataset = (DecoderInput, DecoderOutput)

        np.save(SAVEDIR_x_data, EncoderDataset) 
        np.save(SAVEDIR_y_data, DecoderDataset) 

    def Create_Train_Data(self, DATADIR, SAVEDIR, IMG_SHAPE=[160, 160, 3]):
        print("Creating the train data !!!")
        SAVEDIR_x_data = SAVEDIR + "Transformer_x_train.npy"
        SAVEDIR_y_data = SAVEDIR + "Transformer_y_train.npy"
        # if the directory is not exists then create it:
        os.makedirs(SAVEDIR, exist_ok=True) 
        self.Create_Data(   DATADIR, 
                            SAVEDIR_x_data, 
                            SAVEDIR_y_data, 
                            IMG_SHAPE, 
                            self.seq_n_timesteps, 
                            self.seq_n_features_in)
        print("The train data is ready !!!")
    
    def Create_Test_Data(self, DATADIR, SAVEDIR, IMG_SHAPE=[160, 160, 3]):
        print("Creating the test data !!!")
        SAVEDIR_x_data = SAVEDIR + "Transformer_x_test.npy"
        SAVEDIR_y_data = SAVEDIR + "Transformer_y_test.npy"
        # if the directory is not exists then create it:
        os.makedirs(SAVEDIR, exist_ok=True) 
        self.Create_Data(   DATADIR, 
                            SAVEDIR_x_data, 
                            SAVEDIR_y_data, 
                            IMG_SHAPE, 
                            self.seq_n_timesteps, 
                            self.seq_n_features_in)
        print("The test data is ready !!!")

    def Create_Model(self, Print_Model_Summary=False):
        model = get_model(
                            token_num=self.tokens, # tokens = {0:Non-success, 1:Success, 2:START, 3:STOP}
                            embed_dim=self.seq_n_features_in,
                            encoder_num=self.encoder_num,
                            decoder_num=self.decoder_num,
                            head_num=self.head_num,
                            hidden_dim=self.hidden_dim,
                            attention_activation='relu',
                            feed_forward_activation='relu',
                            dropout_rate=self.dropout_rate,
                            embed_weights=np.random.random((self.tokens, self.seq_n_features_in)),
                        )
        model.compile( optimizer = "adam",
                            loss = "sparse_categorical_crossentropy")

        if Print_Model_Summary:
            print(model.summary())
        # Open the file
        with open(self.Model_Summary,'w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        self.model = model
        return model

    def Fit(self, x_train=None, y_train=None,
            patience=15,
            batch_size=32,
            epochs=100,
            validation_split=0.2):
        if x_train is None:
            x_train = self.x_train
        if y_train is None:
            y_train = self.y_train

        EncoderInput = x_train
        DecoderInput = np.reshape(y_train[0],(y_train[0].shape[0],y_train[0].shape[1]))
        DecoderOutput = y_train[1]

        # simple early stopping
        es = EarlyStopping( monitor='val_loss',
                            mode='min',
                            verbose=1,
                            patience=patience)

        mc = ModelCheckpoint(   filepath=self.Logs_DIR + "/Best_Transformer_ckpt.h5",
                                save_best_only=True,
                                monitor='val_loss',
                                mode='min',
                                verbose=1)

        history = self.model.fit(   [EncoderInput, DecoderInput], DecoderOutput, 
                                    batch_size = batch_size, 
                                    epochs = epochs, 
                                    validation_split = validation_split,
                                    verbose = 1,
                                    callbacks=[es, mc])
        self.model.save(self.Weights)
        return history

    def Load_Model(self, MODEL_PATH):
        self.model.load_weights(MODEL_PATH)

    def Predict(self, data):
        DecoderOutput = np.full((data.shape[0], self.seq_n_timesteps+1), 2) # START
        return self.model.predict([data, DecoderOutput])[:, 0, 1]

    def Evaluate(self, x_test=None, y_test=None):
        if x_test is None:
            x_test = self.x_test
        if y_test is None:
            y_test = self.y_test

        y_test = y_test[0]
        DecoderInput = np.full((y_test.shape[0], y_test.shape[1]), 2) # START
        y_probs = self.model.predict([x_test, DecoderInput])[:, 0, 1]
        y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))[:,-1]
        y_hat = y_probs.round(0)

        model_acc = accuracy_score(y_test, y_hat)
        model_f1 = f1_score(y_test, y_hat)
        lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_probs)
        model_auc = auc(lr_recall, lr_precision)
        model_precision = precision_score(y_test, y_hat), 
        model_recall = recall_score(y_test, y_hat)
        model_cm = confusion_matrix(y_test, y_hat)
        
        print('=================================')
        print("Transformer: ")
        print("Accuracy = %0.3f" %(model_acc))
        print("Precision = %0.3f" %(model_precision))
        print("Recall = %0.3f" %(model_recall))
        print("F1 = %0.3f" %(model_f1))
        print("AUC = %0.3f" %(model_auc))
        print("CM = ", model_cm)
        print('=================================')

    def Load_Img(self, IMG_PATH):
        if self.ImgEmbed is None: 
            self.ImgEmbed = ImageEmbedding(self.embed_model_weights)
        seq = np.array([np.zeros((self.seq_n_timesteps, self.seq_n_features_in), dtype='float32')])
        seq[-1] = self.ImgEmbed.LoadAndEmbed(IMG_PATH)
        return seq

    #TODO: Need to implement this method:
    def Load_Images(self, DATA_PATH):
        pass

    def version(self):
        return "0.1.0"