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

import time

import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.utils import shuffle
import keras
from keras.layers import Input, Flatten, Dense
from keras.layers import BatchNormalization, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model

from PIL import Image

def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]
    
def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]

class GradientReversal(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.num_calls = 0

    def call(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls
        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]
        
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
            
        self.num_calls += 1
        return y

class DANN():
    def __init__(self, Logs_DIR, Print_Model_Summary=False):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.domain_model = None
        self.Logs_DIR = Logs_DIR
        os.makedirs(self.Logs_DIR, exist_ok=True)
        self.Print_Model_Summary = Print_Model_Summary
        self.Weights = self.Logs_DIR + 'DANN_Weights.h5'
        self.model_Summary = self.Logs_DIR + 'DANN_Label_Classifier_Summary.txt'
        self.domain_model_Summary = self.Logs_DIR + 'DANN_Domain_Classifier_Summary.txt'
        self.model, self.domain_model = self.Create_Model(Print_Model_Summary=self.Print_Model_Summary)   

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
        SAVEDIR_x_data = SAVEDIR + "DANN_x_train.npy"
        SAVEDIR_y_data = SAVEDIR + "DANN_y_train.npy"
        # if the directory is not exists then create it:
        os.makedirs(SAVEDIR, exist_ok=True) 
        self.Create_Data(DATADIR, SAVEDIR_x_data, SAVEDIR_y_data, IMG_SHAPE, shuffling)
        print("The train data is ready !!!")
    
    def Create_Test_Data(self, DATADIR, SAVEDIR, IMG_SHAPE=[160, 160, 3], shuffling=False):
        print("Creating the test data !!!")
        SAVEDIR_x_data = SAVEDIR + "DANN_x_test.npy"
        SAVEDIR_y_data = SAVEDIR + "DANN_y_test.npy"
        # if the directory is not exists then create it:
        os.makedirs(SAVEDIR, exist_ok=True) 
        self.Create_Data(DATADIR, SAVEDIR_x_data, SAVEDIR_y_data, IMG_SHAPE, shuffling) 
        print("The test data is ready !!!")

    def Create_Model(   self, 
                        input_shape=(160, 160, 3),
                        label_shape=2,
                        domain_shape=2, 
                        Print_Model_Summary=False):
        inputs = Input ( shape = input_shape )
        # Input image: 160x160x3
        x = ZeroPadding2D((1, 1))(inputs)
        x = Conv2D(64, (3, 3))(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=3)(x)
        x = Conv2D(128, (3, 3))(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=3)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(128, (3, 3))(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=3)(x)
        x = Conv2D(64, (3, 3))(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=3)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(64, (3, 3))(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=3)(x)
        x = Conv2D(32, (3, 3))(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=3)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(16, (3, 3))(x)
        x = Activation('relu')(x)

        x4 = Flatten()(x)

        label_classifier = Dense(100, name = 'mo1')(x4)
        label_classifier = BatchNormalization(name = 'mo2')(label_classifier)
        label_classifier = Activation('relu', name = 'mo3')(label_classifier)
        label_classifier = Dropout(0.25, name = 'mo4')(label_classifier)

        label_classifier = Dense(50, name = 'mo5')(label_classifier)
        label_classifier = BatchNormalization(name = 'mo6')(label_classifier)
        label_classifier = Activation('relu', name = 'mo7')(label_classifier)
        label_classifier = Dropout(0.25, name = 'mo8')(label_classifier)

        label_classifier = Dense(label_shape, name = 'mo9')(label_classifier)
        label_classifier = BatchNormalization(name = 'mo10')(label_classifier)
        label_classifier_outputs = Activation('softmax', name = 'mo')(label_classifier)

        domain_classifier = GradientReversal()(x4)
        domain_classifier = Dense(100, name = 'do1')(domain_classifier)
        domain_classifier = BatchNormalization(name='do2')(domain_classifier)
        domain_classifier = Activation('relu', name = 'do3')(domain_classifier)
        domain_classifier = Dropout(0.25, name = 'do4')(domain_classifier)

        domain_classifier = Dense(50, name = 'do5')(domain_classifier)
        domain_classifier = BatchNormalization(name='do6')(domain_classifier)
        domain_classifier = Activation('relu', name = 'do7')(domain_classifier)
        domain_classifier = Dropout(0.25, name = 'do8')(domain_classifier)

        domain_classifier = Dense(domain_shape, name = 'do9')(domain_classifier)
        domain_classifier = BatchNormalization(name = 'do10')(domain_classifier)
        domain_classifier_outputs = Activation('softmax', name = 'do')(domain_classifier)

        label_classifier = Model(   inputs = inputs,
                                    outputs = label_classifier_outputs)
        domain_classifier = Model(  inputs = inputs,
                                    outputs = domain_classifier_outputs)
        
        label_classifier.compile(   optimizer = "Adam",
                                    loss = 'categorical_crossentropy',
                                    metrics = ['accuracy'])
        domain_classifier.compile(  optimizer = "Adam",
                                    loss = 'categorical_crossentropy',
                                    metrics = ['accuracy'])

        if Print_Model_Summary:
            print(label_classifier.summary())
        # Open the file
        with open(self.model_Summary,'w') as fh:
            label_classifier.summary(print_fn=lambda x: fh.write(x + '\n'))

        if Print_Model_Summary:
            print(domain_classifier.summary())
        # Open the file
        with open(self.domain_model_Summary,'w') as fh:
            domain_classifier.summary(print_fn=lambda x: fh.write(x + '\n'))

        self.model = label_classifier
        self.domain_model = domain_classifier
        return label_classifier, domain_classifier

    def Fit(self, model_label=None, model_domain=None, 
            source_x_train=None, source_y_train=None,
            target_x_train=None, target_y_train=None, 
            batch_size=32,
            epochs=100):
        if model_label is None: model_label = self.model
        if model_domain is None: model_domain = self.domain_model
        if source_x_train is None: source_x_train = self.x_train
        if source_y_train is None: source_y_train = self.y_train
        if target_x_train is None: target_x_train = self.x_test
        if target_y_train is None: target_y_train = self.y_test
        
        print("Start Training!!!")
        
        gen_source_batch = batch_generator([source_x_train, source_y_train ], batch_size)
        gen_target_batch = batch_generator([target_x_train, target_y_train ], batch_size // 2)

        imgs_per_epoch = source_x_train.shape[0]
        e = 0
        img_nr = 0

        target_accuracy_best = 0.0

        start = time.time()

        while e < epochs:
            Xs, ys = next(gen_source_batch)
            Xt, _ = next(gen_target_batch)

            Xd = np.vstack([Xs[0:batch_size // 2], Xt])
            yd = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]), np.tile([0., 1.], [batch_size // 2, 1])])
            Xd, yd = shuffle(Xd, yd)

            label_loss = model_label.train_on_batch(Xs, ys)
            domain_loss = model_domain.train_on_batch(Xd, yd)

            img_nr += batch_size
            if img_nr > imgs_per_epoch:
                img_nr = 0
                e += 1
                ys_pred = model_label.predict(source_x_train)
                yt_pred = model_label.predict(target_x_train)

                source_accuracy = accuracy_score(source_y_train.argmax(1), ys_pred.argmax(1))              
                target_accuracy = accuracy_score(target_y_train.argmax(1), yt_pred.argmax(1))

                log_str1 = "Epoch: " + str(e) + ", Time/Epoch: " + str(time.time()-start) + ", Best Target Accuracy = " + str(target_accuracy_best)
                log_str2 = "LABEL CLASSIFICATION: source_accuracy: {:.5f}, target_accuracy: {:.5f} \n"\
                                                            .format(source_accuracy*100, target_accuracy*100)
                print(log_str1)
                print(log_str2)

                if target_accuracy_best < target_accuracy:
                    target_accuracy_best = target_accuracy
                    model_label.save(self.Logs_DIR + "/Best_DANN_ckpt.h5")
                    print("Best Target Accuracy = ", target_accuracy)

                start = time.time()
        model_label.save(self.Weights)
        self.model = model_label
        self.domain_model = model_domain

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
        print("DANN: ")
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