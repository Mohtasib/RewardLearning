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

from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image

class ImageEmbedding():
    def __init__(self):
        self.model = applications.nasnet.NASNetLarge(   weights='imagenet',
                                                        include_top=False,
                                                        pooling='avg')

    def read_image(self, img_path):
        img_path = os.path.join(img_path)
        return image.load_img(img_path, target_size=(331,331))

    def format_image_array(self, img):
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def get_feature(self, path):
        img = self.read_image(path)
        arr = self.format_image_array(img)
        # extract the features
        return self.model.predict(arr)[0]
    
    def Embed_An_Img(self, Img):
        arr = self.format_image_array(Img)
        return self.model.predict(arr)[0, None]

class NASNet():
    def __init__(self, Logs_DIR, Print_Model_Summary=False):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.embed_len = 4032
        self.ImgEmbed = None
        self.Logs_DIR = Logs_DIR
        os.makedirs(self.Logs_DIR, exist_ok=True)
        self.Print_Model_Summary = Print_Model_Summary
        self.Weights = self.Logs_DIR + 'NASNet_Weights.h5'
        self.Model_Summary = self.Logs_DIR + 'NASNET_Summary.txt'
        self.model = self.Create_Model(Print_Model_Summary=self.Print_Model_Summary)

    def _Create_Data(self, DATADIR, SAVEDIR, shuffling=False):
        CATEGORIES = ["Non-success", "Success"]
        dataset = []
        for category in CATEGORIES :
            DATA_PATH = os.path.join(DATADIR, category)
            class_num = CATEGORIES.index(category)
            for i in os.listdir(DATA_PATH):
                if self.ImgEmbed is None: 
                    self.ImgEmbed = ImageEmbedding()
                arr = self.ImgEmbed.get_feature(os.path.join(DATA_PATH, i))
                dataset.append([arr, class_num])
    
        if shuffling:
            random.shuffle(dataset)
        data = np.array(dataset)
        x_data = list(data[:,0])
        y_data = list(data[:,1])

        x_data = np.reshape(np.array(x_data), (len(x_data), self.embed_len))
        y_data = np.reshape(np.array(y_data), (len(y_data), 1))

        np.save(SAVEDIR + 'x_data.npy', x_data) 
        np.save(SAVEDIR + 'y_data.npy', y_data) 

    def Create_Data(self, DATADIR, SAVEDIR_x_data, SAVEDIR_y_data, shuffling=False):
        demos = os.listdir(DATADIR) #  get the list of folders in this directory
        for i in range(0, len(demos)):
            print("Working on", demos[i])
            self._Create_Data(  DATADIR = os.path.join(DATADIR + demos[i] + "/"),
                                SAVEDIR = os.path.join(DATADIR + demos[i] + "/"),
                                shuffling = shuffling)

        x_dataset = []
        y_dataset = []

        demos = os.listdir(DATADIR) #  get the list of folders in this directory
        for i in range(0, len(demos)):
            data_dir = os.path.join(DATADIR + demos[i] + "/")
            x_train = np.load(data_dir + "x_data.npy")
            y_train = np.load(data_dir + "y_data.npy")
            x_dataset.append(x_train)
            y_dataset.append(y_train)

        x_data = np.vstack(x_dataset)
        y_data = np.vstack(y_dataset)

        np.save(SAVEDIR_x_data, x_data) 
        np.save(SAVEDIR_y_data, y_data) 

    def Create_Train_Data(self, DATADIR, SAVEDIR, shuffling=False):
        print("Creating the train data !!!")
        SAVEDIR_x_data = SAVEDIR + "NASNet_x_train.npy"
        SAVEDIR_y_data = SAVEDIR + "NASNet_y_train.npy"
        # if the directory is not exists then create it:
        os.makedirs(SAVEDIR, exist_ok=True) 
        self.Create_Data(DATADIR, SAVEDIR_x_data, SAVEDIR_y_data, shuffling)
        print("The train data is ready !!!")
    
    def Create_Test_Data(self, DATADIR, SAVEDIR, shuffling=False):
        print("Creating the test data !!!")
        SAVEDIR_x_data = SAVEDIR + "NASNet_x_test.npy"
        SAVEDIR_y_data = SAVEDIR + "NASNet_y_test.npy"
        # if the directory is not exists then create it:
        os.makedirs(SAVEDIR, exist_ok=True) 
        self.Create_Data(DATADIR, SAVEDIR_x_data, SAVEDIR_y_data, shuffling) 
        print("The test data is ready !!!")

    def Create_Model(self, Print_Model_Summary=False):
        model = Sequential()
        model.add(Dense(2048, input_shape=(self.embed_len,)))
        model.add(Activation('relu'))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(8))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile( optimizer = "adam",
                            loss = "binary_crossentropy", 
                            metrics = ['accuracy'])

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
        # simple early stopping
        es = EarlyStopping( monitor='val_loss',
                            mode='min',
                            verbose=1,
                            patience=patience)

        mc = ModelCheckpoint(   filepath=self.Logs_DIR + "/Best_NASNet_ckpt.h5",
                                save_best_only=True,
                                monitor='val_loss',
                                mode='min',
                                verbose=1)

        history = self.model.fit(   x_train,  y_train, 
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
        return self.model.predict(data, verbose=0)

    def Evaluate(self, x_test=None, y_test=None):
        if x_test is None:
            x_test = self.x_test
        if y_test is None:
            y_test = self.y_test

        y_probs = self.model.predict(x_test)
        y_hat = y_probs.round(0)

        model_acc = accuracy_score(y_test, y_hat)
        model_f1 = f1_score(y_test, y_hat)
        lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_probs)
        model_auc = auc(lr_recall, lr_precision)
        model_precision = precision_score(y_test, y_hat), 
        model_recall = recall_score(y_test, y_hat)
        model_cm = confusion_matrix(y_test, y_hat)

        print('=================================')
        print("NASNet: ")
        print("Accuracy = %0.3f" %(model_acc))
        print("Precision = %0.3f" %(model_precision))
        print("Recall = %0.3f" %(model_recall))
        print("F1 = %0.3f" %(model_f1))
        print("AUC = %0.3f" %(model_auc))
        print("CM = ", model_cm)
        print('=================================')

    def Load_Img(self, IMG_PATH):
        if self.ImgEmbed is None: 
            self.ImgEmbed = ImageEmbedding()
        img = self.ImgEmbed.read_image(IMG_PATH)
        return self.ImgEmbed.Embed_An_Img(img)

    def Load_Images(self, DATA_PATH):
        if self.ImgEmbed is None: 
            self.ImgEmbed = ImageEmbedding()
        dataset = []
        for i in os.listdir(DATA_PATH):
            arr = self.ImgEmbed.get_feature(os.path.join(DATA_PATH, i))
            dataset.append(arr)

        return np.reshape(np.array(dataset), (len(dataset), self.embed_len))

    def version(self):
        return "0.1.0"