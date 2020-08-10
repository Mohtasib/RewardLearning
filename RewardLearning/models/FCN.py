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
from keras.layers import Lambda, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from PIL import Image

class FCN():
    def __init__(self, Logs_DIR, Print_Model_Summary=False):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.Logs_DIR = Logs_DIR
        os.makedirs(self.Logs_DIR, exist_ok=True)
        self.Print_Model_Summary = Print_Model_Summary
        self.Weights = self.Logs_DIR + 'FCN_Weights.h5'
        self.Model_Summary = self.Logs_DIR + 'FCN_Summary.txt'
        self.model = self.Create_Model(Print_Model_Summary=self.Print_Model_Summary)   

    def _Create_Data(self, DATADIR, SAVEDIR, IMG_SHAPE, shuffling=False):
        CATEGORIES = ["Non-success", "Success"]
        img_width = IMG_SHAPE[0]
        img_height = IMG_SHAPE[1]
        num_channels = IMG_SHAPE[2]
        dataset = []
        for category in CATEGORIES :
            DATA_PATH = os.path.join(DATADIR, category)
            class_num = CATEGORIES.index(category)
            for i in os.listdir(DATA_PATH):
                img = Image.open(os.path.join(DATA_PATH, i))  # this is a PIL image
                resized = img.resize((img_width, img_height)) # Resize the image
                reshaped = np.reshape(resized, (-1, img_width, img_height, num_channels))
                arr = reshaped/255
                class_label = [0, 0]
                class_label[class_num] = 1
                dataset.append([arr, class_label])
    
        if shuffling:
            random.shuffle(dataset)
        data = np.array(dataset)
        x_data = list(data[:,0])
        y_data = list(data[:,1])

        x_data = np.reshape(np.array(x_data), (len(x_data), img_width, img_height, num_channels))
        y_data = np.reshape(np.array(y_data), (len(y_data), 2))

        np.save(SAVEDIR + 'x_data.npy', x_data) 
        np.save(SAVEDIR + 'y_data.npy', y_data) 

    def Create_Data(self, DATADIR, SAVEDIR_x_data, SAVEDIR_y_data, IMG_SHAPE, shuffling=False):
        demos = os.listdir(DATADIR) #  get the list of folders in this directory
        for i in range(0, len(demos)):
            print("Working on", demos[i])
            self._Create_Data(  DATADIR = os.path.join(DATADIR + demos[i] + "/"),
                                SAVEDIR = os.path.join(DATADIR + demos[i] + "/"),
                                IMG_SHAPE = IMG_SHAPE,
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

    def Create_Train_Data(self, DATADIR, SAVEDIR, IMG_SHAPE=[160, 160, 3], shuffling=False):
        print("Creating the train data !!!")
        SAVEDIR_x_data = SAVEDIR + "FCN_x_train.npy"
        SAVEDIR_y_data = SAVEDIR + "FCN_y_train.npy"
        # if the directory is not exists then create it:
        os.makedirs(SAVEDIR, exist_ok=True) 
        self.Create_Data(DATADIR, SAVEDIR_x_data, SAVEDIR_y_data, IMG_SHAPE, shuffling)
        print("The train data is ready !!!")
    
    def Create_Test_Data(self, DATADIR, SAVEDIR, IMG_SHAPE=[160, 160, 3], shuffling=False):
        print("Creating the test data !!!")
        SAVEDIR_x_data = SAVEDIR + "FCN_x_test.npy"
        SAVEDIR_y_data = SAVEDIR + "FCN_y_test.npy"
        # if the directory is not exists then create it:
        os.makedirs(SAVEDIR, exist_ok=True) 
        self.Create_Data(DATADIR, SAVEDIR_x_data, SAVEDIR_y_data, IMG_SHAPE, shuffling) 
        print("The test data is ready !!!")

    def ConvBlock(self, model, layers, filters):
        '''
        Create [layers] layers consisting of zero padding, 
        a convolution with [filters] 3x3 filters and batch normalization. 
        Perform max pooling after the last layer.
        '''
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Conv2D(filters, (3, 3), activation='relu'))
            model.add(BatchNormalization(axis=3))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def Create_Model(self, input_shape=(160, 160, 3), Print_Model_Summary=False):
        model = Sequential()
        # Input image: 160x160x3
        model.add(Lambda(lambda x: x, input_shape=input_shape))
        self.ConvBlock(model, 1, 32)
        # 80x80x32
        self.ConvBlock(model, 1, 64)
        # 40x40x64
        self.ConvBlock(model, 1, 128)
        # 20x20x128
        self.ConvBlock(model, 1, 128)
        # 10x10x128
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(2, (3, 3), activation='relu'))
        model.add(GlobalAveragePooling2D())
        # 10x10x2
        model.add(Activation('softmax'))

        model.compile( optimizer = "adam",
                            loss = "categorical_crossentropy", 
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

        mc = ModelCheckpoint(   filepath=self.Logs_DIR + "/Best_FCN_ckpt.h5",
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
        return self.model.predict(data, verbose=0)[:,1]

    def Evaluate(self, x_test=None, y_test=None):
        if x_test is None:
            x_test = self.x_test
        if y_test is None:
            y_test = self.y_test

        y_probs = self.model.predict(x_test)
        y_hat = y_probs.argmax(1)

        y_test = y_test[:, 1]
        y_probs = y_probs[:, 1]

        model_acc = accuracy_score(y_test, y_hat)
        model_f1 = f1_score(y_test, y_hat)
        lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_probs)
        model_auc = auc(lr_recall, lr_precision)
        model_precision = precision_score(y_test, y_hat), 
        model_recall = recall_score(y_test, y_hat)
        model_cm = confusion_matrix(y_test, y_hat)
        
        print('=================================')
        print("FCN: ")
        print("Accuracy = %0.3f" %(model_acc))
        print("Precision = %0.3f" %(model_precision))
        print("Recall = %0.3f" %(model_recall))
        print("F1 = %0.3f" %(model_f1))
        print("AUC = %0.3f" %(model_auc))
        print("CM = ", model_cm)
        print('=================================')

    def Load_Img(self, IMG_PATH, IMG_SHAPE=[160, 160, 3]):
        img_width = IMG_SHAPE[0]
        img_height = IMG_SHAPE[1]
        num_channels = IMG_SHAPE[2]
        img = Image.open(IMG_PATH)  # this is a PIL image
        resized = img.resize((img_width, img_height)) # Resize the image
        reshaped = np.reshape(resized, (-1, img_width, img_height, num_channels))   # this is a Numpy array with shape (3, 150, 150)
        return reshaped/255

    def Load_Images(self, DATA_PATH, IMG_SHAPE=[160, 160, 3]):
        img_width = IMG_SHAPE[0]
        img_height = IMG_SHAPE[1]
        num_channels = IMG_SHAPE[2]
        dataset = []
        for i in os.listdir(DATA_PATH):
            img = self.Load_Img(os.path.join(DATA_PATH, i), IMG_SHAPE) 
            dataset.append(img)

        return np.reshape(np.array(dataset), (len(dataset), img_width, img_height, num_channels))

    def version(self):
        return "0.1.0"