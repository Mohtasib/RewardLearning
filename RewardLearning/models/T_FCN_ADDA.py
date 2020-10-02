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

from keras.layers import Input, Flatten, Dense, concatenate, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model, load_model
from keras.utils import to_categorical

from PIL import Image

def batch_generator(data, batch_size):
    """Generate batches of data.

    Given a list of numpy data, it iterates over the list and returns batches of the same size
    This
    """
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(all_examples_indices, size=batch_size, replace=False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr

class T_FCN_ADDA():
    def __init__(self, Logs_DIR, Print_Model_Summary=False):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.comb_model = None
        self.model = None
        self.domain_classification_model = None
        self.embeddings_model = None
        self.Logs_DIR = Logs_DIR
        os.makedirs(self.Logs_DIR, exist_ok=True)
        self.Print_Model_Summary = Print_Model_Summary
        self.Weights = self.Logs_DIR + 'T_FCN_ADDA_Weights.h5'
        self.comb_model_Summary = self.Logs_DIR + 'T_FCN_ADDA_comb_model_Summary.txt'
        self.model_Summary = self.Logs_DIR + 'T_FCN_ADDA_source_classification_model_Summary.txt'
        self.domain_classification_model_Summary = self.Logs_DIR + 'T_FCN_ADDA_domain_classification_model_Summary.txt'
        self.embeddings_model_Summary = self.Logs_DIR + 'T_FCN_ADDA_embeddings_model_Summary.txt'
        self.comb_model, self.model, self.domain_classification_model, self.embeddings_model = self.Create_Model(Print_Model_Summary=self.Print_Model_Summary)   

    def _Create_Data(self, DATADIR, SAVEDIR, IMG_SHAPE, shuffling=False):
        img_width = IMG_SHAPE[0]
        img_height = IMG_SHAPE[1]
        num_channels = IMG_SHAPE[2]
        dataset = []
        
        NS_len = len(os.listdir(os.path.join(DATADIR, "Non-success")))
        S_len = len(os.listdir(os.path.join(DATADIR, "Success")))
        Demo_len = NS_len + S_len
        
        # Load Non-success data:
        DATA_PATH = os.path.join(DATADIR, "Non-success")
        class_label = [1, 0]
        img_num = 1
        for i in os.listdir(DATA_PATH):
            img = Image.open(os.path.join(DATA_PATH, i))  # this is a PIL image
            resized = img.resize((img_width, img_height)) # Resize the image
            reshaped = np.reshape(resized, (-1, img_width, img_height, num_channels))
            arr = reshaped/255
            timing_label = float(img_num) / float(Demo_len)
            img_num += 1
            dataset.append([arr, class_label, timing_label])

        # Load Success data:
        DATA_PATH = os.path.join(DATADIR, "Success")
        class_label = [0, 1]
        img_num = NS_len + 1
        for i in os.listdir(DATA_PATH):
            img = Image.open(os.path.join(DATA_PATH, i))  # this is a PIL image
            resized = img.resize((img_width, img_height)) # Resize the image
            reshaped = np.reshape(resized, (-1, img_width, img_height, num_channels))
            arr = reshaped/255
            timing_label = float(img_num) / float(Demo_len)
            img_num += 1
            dataset.append([arr, class_label, timing_label])
    
        if shuffling:
            random.shuffle(dataset)
        data = np.array(dataset)
        x_data = list(data[:,0])
        y_data = list(data[:,1:])

        x_data = np.reshape(np.array(x_data), (len(x_data), img_width, img_height, num_channels))

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
            y_train = np.load(data_dir + "y_data.npy", allow_pickle=True)
            x_dataset.append(x_train)
            y_dataset.append(y_train)

        x_data = np.vstack(x_dataset)
        y_data = np.vstack(y_dataset)

        np.save(SAVEDIR_x_data, x_data) 
        np.save(SAVEDIR_y_data, y_data) 

    def Create_Train_Data(self, DATADIR, SAVEDIR, IMG_SHAPE=[160, 160, 3], shuffling=False):
        print("Creating the train data !!!")
        SAVEDIR_x_data = SAVEDIR + "T_FCN_ADDA_x_train.npy"
        SAVEDIR_y_data = SAVEDIR + "T_FCN_ADDA_y_train.npy"
        # if the directory is not exists then create it:
        os.makedirs(SAVEDIR, exist_ok=True) 
        self.Create_Data(DATADIR, SAVEDIR_x_data, SAVEDIR_y_data, IMG_SHAPE, shuffling)
        print("The train data is ready !!!")
    
    def Create_Test_Data(self, DATADIR, SAVEDIR, IMG_SHAPE=[160, 160, 3], shuffling=False):
        print("Creating the test data !!!")
        SAVEDIR_x_data = SAVEDIR + "T_FCN_ADDA_x_test.npy"
        SAVEDIR_y_data = SAVEDIR + "T_FCN_ADDA_y_test.npy"
        # if the directory is not exists then create it:
        os.makedirs(SAVEDIR, exist_ok=True) 
        self.Create_Data(DATADIR, SAVEDIR_x_data, SAVEDIR_y_data, IMG_SHAPE, shuffling) 
        print("The test data is ready !!!")

    def Create_Model(   self, 
                        input_shape=(160, 160, 3),
                        label_shape=2,
                        domain_shape=2,
                        Print_Model_Summary=False):
        inputs = Input(shape=input_shape)

        # Layer 1 Common Path Network:
        Common_Path_layer1 = ZeroPadding2D((1, 1))(inputs)
        Common_Path_layer1 = Conv2D(64, (3, 3), activation='relu', name='Common_Path_Conv2d_Layer1')(Common_Path_layer1)
        Common_Path_layer1 = BatchNormalization(axis=3, name='Common_Path_BatchNorm_Layer1')(Common_Path_layer1)
        # Layer 2 Common Path Network:
        Common_Path_layer2 = MaxPooling2D((2, 2), strides=(2, 2))(Common_Path_layer1)
        # Layer 3 Common Path Network:
        Common_Path_layer3 = ZeroPadding2D((1, 1))(Common_Path_layer2)
        Common_Path_layer3 = Conv2D(128, (3, 3), activation='relu', name='Common_Path_Conv2d_Layer3')(Common_Path_layer3)
        Common_Path_layer3 = BatchNormalization(axis=3, name='Common_Path_BatchNorm_Layer3')(Common_Path_layer3)
        # Layer 4 Common Path Network:
        Common_Path_layer4 = MaxPooling2D((2, 2), strides=(2, 2))(Common_Path_layer3)
        # Layer 5 Common Path Network:
        Common_Path_layer5 = ZeroPadding2D((1, 1))(Common_Path_layer4)
        Common_Path_layer5 = Conv2D(256, (3, 3), activation='relu', name='Common_Path_Conv2d_Layer5')(Common_Path_layer5)
        Common_Path_layer5 = BatchNormalization(axis=3, name='Common_Path_BatchNorm_Layer5')(Common_Path_layer5)
        # Layer 6 Common Path Network:
        Common_Path_layer6 = MaxPooling2D((2, 2), strides=(2, 2))(Common_Path_layer5)
        # Layer 7 Common Path Network:
        Common_Path_layer7 = ZeroPadding2D((1, 1))(Common_Path_layer6)
        Common_Path_layer7 = Conv2D(256, (3, 3), activation='relu', name='Common_Path_Conv2d_Layer7')(Common_Path_layer7)
        Common_Path_layer7 = BatchNormalization(axis=3, name='Common_Path_BatchNorm_Layer7')(Common_Path_layer7)
        # Layer 8 Common Path Network:
        Common_Path_layer8 = MaxPooling2D((2, 2), strides=(2, 2))(Common_Path_layer7)


        # Layer 9 Timing Path Network:
        Timing_Path_layer9 = ZeroPadding2D((1, 1))(Common_Path_layer8)
        Timing_Path_layer9 = Conv2D(128, (3, 3), activation='relu', name='Timing_Path_Conv2d_Layer9')(Timing_Path_layer9)
        Timing_Path_layer9 = BatchNormalization(axis=3, name='Timing_Path_BatchNorm_Layer9')(Timing_Path_layer9)
        # Layer 10 Timing Path Network:
        Timing_Path_layer10 = MaxPooling2D((2, 2), strides=(2, 2))(Timing_Path_layer9)

        # Layer 9 Classification Path Network:
        Class_Path_layer9 = ZeroPadding2D((1, 1))(Common_Path_layer8)
        Class_Path_layer9 = Conv2D(128, (3, 3), activation='relu', name='Class_Path_Conv2d_Layer9')(Class_Path_layer9)
        Class_Path_layer9 = BatchNormalization(axis=3, name='Class_Path_BatchNorm_Layer9')(Class_Path_layer9)
        # Layer 10 Classification Path Network:
        Class_Path_layer10 = MaxPooling2D((2, 2), strides=(2, 2))(Class_Path_layer9)

        # Layer 11 Timing Path Network:
        Timing_Path_layer11 = concatenate([Timing_Path_layer10, Class_Path_layer10], axis=3)
        # Layer 12 Timing Path Network:
        Timing_Path_layer12 = ZeroPadding2D((1, 1))(Timing_Path_layer11)
        Timing_Path_layer12 = Conv2D(64, (3, 3), activation='relu', name='Timing_Path_Conv2d_Layer12')(Timing_Path_layer12)
        Timing_Path_layer12 = BatchNormalization(axis=3, name='Timing_Path_BatchNorm_Layer12')(Timing_Path_layer12)
        # Layer 13 Timing Path Network:
        Timing_Path_layer13 = MaxPooling2D((2, 2), strides=(2, 2))(Timing_Path_layer12)
        
        # Layer 11 Classification Path Network:
        Class_Path_layer11 = concatenate([Class_Path_layer10, Timing_Path_layer10], axis=3)
        # Layer 12 Classification Path Network:
        Class_Path_layer12 = ZeroPadding2D((1, 1))(Class_Path_layer11)
        Class_Path_layer12 = Conv2D(64, (3, 3), activation='relu', name='Class_Path_Conv2d_Layer12')(Class_Path_layer12)
        Class_Path_layer12 = BatchNormalization(axis=3, name='Class_Path_BatchNorm_Layer12')(Class_Path_layer12)
        # Layer 13 Classification Path Network:
        Class_Path_layer13 = MaxPooling2D((2, 2), strides=(2, 2))(Class_Path_layer12)

        # Layer 14 Timing Path Network:
        Timing_Path_layer14 = concatenate([Timing_Path_layer13, Class_Path_layer13], axis=3)
        # Layer 15 Timing Path Network:
        Timing_Path_layer15 = ZeroPadding2D((1, 1))(Timing_Path_layer14)
        Timing_Path_layer15 = Conv2D(2, (3, 3), activation='relu', name='Timing_Path_Conv2d_Layer15')(Timing_Path_layer15)
        Timing_Path_layer15 = GlobalAveragePooling2D()(Timing_Path_layer15)

        # Layer 14 Classification Path Network:
        Class_Path_layer14 = concatenate([Class_Path_layer13, Timing_Path_layer13], axis=3)
        # Layer 15 Classification Path Network:
        Class_Path_layer15 = ZeroPadding2D((1, 1))(Class_Path_layer14)
        Class_Path_layer15 = Conv2D(2, (3, 3), activation='relu', name='Class_Path_Conv2d_Layer15')(Class_Path_layer15)
        Class_Path_layer15 = GlobalAveragePooling2D()(Class_Path_layer15)

        # Layer 16 Timing Path Network:
        Timing_Path_layer16 = concatenate([Timing_Path_layer15, Class_Path_layer15])
        # Output Layer Timing Path Network:
        Timing_Path_output = Dense(1, name='Timing_Path_Output')(Timing_Path_layer16)
        
        # Layer 16 Classification Path Network:
        Class_Path_layer16 = concatenate([Class_Path_layer15, Timing_Path_layer15])
        # Output Layer Classification Path Network:
        Class_Path_output = Dense(2, activation='softmax', name='Class_Path_Output')(Class_Path_layer16)
        

        x4 = Flatten()(Common_Path_layer8) 

        domain_classifier = Dense(100, name = 'do1')(x4)
        domain_classifier = BatchNormalization(name='do2')(domain_classifier)
        domain_classifier = Activation('relu', name = 'do3')(domain_classifier)
        domain_classifier = Dropout(0.25, name = 'do4')(domain_classifier)

        domain_classifier = Dense(50, name = 'do5')(domain_classifier)
        domain_classifier = BatchNormalization(name='do6')(domain_classifier)
        domain_classifier = Activation('relu', name = 'do7')(domain_classifier)
        domain_classifier = Dropout(0.25, name = 'do8')(domain_classifier)

        domain_classifier = Dense(2, name = 'do9')(domain_classifier)
        domain_classifier = BatchNormalization(name = 'do10')(domain_classifier)
        domain_classifier = Activation('sigmoid', name = 'do')(domain_classifier)

        comb_model = Model(inputs=inputs, outputs=[Class_Path_output, Timing_Path_output, domain_classifier])
        comb_model.compile(optimizer="Adam",
                loss={'Class_Path_Output': 'categorical_crossentropy', 'Timing_Path_Output': 'mse', 'do': 'categorical_crossentropy'},
                loss_weights={'Class_Path_Output': 1, 'Timing_Path_Output': 1, 'do': 2})

        source_classification_model = Model(inputs=inputs, outputs=[Class_Path_output, Timing_Path_output])
        source_classification_model.compile(
            optimizer="adam",
            loss={'Class_Path_Output': 'categorical_crossentropy', 'Timing_Path_Output': 'mse'},
            loss_weights={'Class_Path_Output': 1, 'Timing_Path_Output': 1})

        domain_classification_model = Model(inputs=inputs, outputs=[domain_classifier])
        domain_classification_model.compile(optimizer="Adam",
                    loss={'do': 'categorical_crossentropy'}, metrics=['accuracy'])

        embeddings_model = Model(inputs=inputs, outputs=[x4])
        embeddings_model.compile(   optimizer="Adam",
                                    loss = 'categorical_crossentropy',
                                    metrics=['accuracy'])

        if Print_Model_Summary:
            print(comb_model.summary())
        # Open the file
        with open(self.comb_model_Summary,'w') as fh:
            comb_model.summary(print_fn=lambda x: fh.write(x + '\n'))

        if Print_Model_Summary:
            print(source_classification_model.summary())
        # Open the file
        with open(self.model_Summary,'w') as fh:
            source_classification_model.summary(print_fn=lambda x: fh.write(x + '\n'))

        if Print_Model_Summary:
            print(domain_classification_model.summary())
        # Open the file
        with open(self.domain_classification_model_Summary,'w') as fh:
            domain_classification_model.summary(print_fn=lambda x: fh.write(x + '\n'))

        if Print_Model_Summary:
            print(embeddings_model.summary())
        # Open the file
        with open(self.embeddings_model_Summary,'w') as fh:
            embeddings_model.summary(print_fn=lambda x: fh.write(x + '\n'))

        self.comb_model = comb_model
        self.model = source_classification_model
        self.domain_classification_model = domain_classification_model
        self.embeddings_model = embeddings_model
        return comb_model, source_classification_model, domain_classification_model, embeddings_model

    def Fit(self, model=None, source_classification_model=None,
            domain_classification_model=None, embeddings_model=None, 
            source_x_train=None, source_y_train=None,
            target_x_train=None, target_y_train=None, 
            batch_size=32,
            epochs=100):
        if model is None: model = self.comb_model
        if source_classification_model is None: source_classification_model = self.model
        if domain_classification_model is None: domain_classification_model = self.domain_classification_model
        if embeddings_model is None: embeddings_model = self.embeddings_model
        if source_x_train is None: source_x_train = self.x_train
        if source_y_train is None: source_y_train = self.y_train
        if target_x_train is None: target_x_train = self.x_test
        if target_y_train is None: target_y_train = self.y_test

        class_data = list(source_y_train[:,0])
        timing_data = list(source_y_train[:,1])

        class_data = np.reshape(np.array(class_data), (len(class_data), 2))
        timing_data = np.reshape(np.array(timing_data), (len(timing_data), 1))


        Xs = source_x_train
        ys1 = class_data
        ys2 = timing_data

        class_data = list(target_y_train[:,0])
        class_data = np.reshape(np.array(class_data), (len(class_data), 2))

        Xt = target_x_train
        yt = class_data
        
        print("Start Training!!!")
        
        imgs_per_epoch = Xs.shape[0]
        e = 0
        img_nr = 0

        y_adversarial_1 = to_categorical(np.array(([1] * batch_size + [0] * batch_size)))
    
        sample_weights_class = np.array(([1] * batch_size + [0] * batch_size))
        sample_weights_adversarial = np.ones((batch_size * 2,))

        S_batches = batch_generator([Xs, ys1, ys2], batch_size)
        T_batches = batch_generator([Xt, np.zeros(shape = (len(Xt),2)), np.zeros(shape = (len(Xt),1))], batch_size)
        
        target_accuracy_best = 0.0
        start = time.time()

        while e < epochs:
            y_adversarial_2 = to_categorical(np.array(([0] * batch_size + [1] * batch_size)))

            X0, y01, y02 = next(S_batches)
            X1, _, _ = next(T_batches)

            X_adv = np.concatenate([X0, X1])
            y_class1 = np.concatenate([y01, np.zeros_like(y01)])
            y_class2 = np.concatenate([y02, np.zeros_like(y02)])

            adv_weights = []
            for layer in model.layers:
                if (layer.name.startswith("do")):
                    adv_weights.append(layer.get_weights())

            # note - even though we save and append weights, the batchnorms moving means and variances
            # are not saved throught this mechanism 
            stats = model.train_on_batch(X_adv, [y_class1, y_class2, y_adversarial_1],
                                    sample_weight=[sample_weights_class, sample_weights_class, sample_weights_adversarial])

            k = 0
            for layer in model.layers:
                if (layer.name.startswith("do")):
                    layer.set_weights(adv_weights[k])
                    k += 1

            class_weights = []
                
            
            for layer in model.layers:
                if (not layer.name.startswith("do")):
                    class_weights.append(layer.get_weights())

            stats2 = domain_classification_model.train_on_batch(X_adv, [y_adversarial_2])

            k = 0
            for layer in model.layers:
                if (not layer.name.startswith("do")):
                    layer.set_weights(class_weights[k])
                    k += 1

            img_nr += batch_size
            if img_nr > imgs_per_epoch:
                img_nr = 0
                e += 1
                # print(i, stats)
                y_test_hat_t = source_classification_model.predict(Xt)[0].argmax(1)
                y_test_hat_s = source_classification_model.predict(Xs)[0].argmax(1)
                source_accuracy = accuracy_score(ys1[:,1], y_test_hat_s)
                target_accuracy = accuracy_score(yt[:,1], y_test_hat_t)
                log_str1 = "Epoch: " + str(e) + ", Time/Epoch: " + str(time.time()-start) + ", Best Target Accuracy = " + str(target_accuracy_best)
                log_str2 = "LABEL CLASSIFICATION: source_accuracy: {:.5f}, target_accuracy: {:.5f} \n"\
                                                            .format(source_accuracy*100, target_accuracy*100)
                print(log_str1)
                print(log_str2)

                if target_accuracy_best < target_accuracy:
                    target_accuracy_best = target_accuracy
                    source_classification_model.save(self.Logs_DIR + "/Best_T_FCN_ADDA_ckpt.h5")
                    print("Best Target Accuracy = ", target_accuracy)
                
                start = time.time()
        source_classification_model.save(self.Weights)
        self.comb_model = model
        self.model = source_classification_model
        self.domain_classification_model = domain_classification_model
        self.embeddings_model = embeddings_model   

    def Load_Model(self, MODEL_PATH):
        self.model.load_weights(MODEL_PATH)

    def Predict(self, data):
        return self.model.predict(data, verbose=0)[0]

    def Evaluate(self, x_test=None, y_test=None):
        if x_test is None:
            x_test = self.x_test
        if y_test is None:
            y_test = self.y_test

        y_data = list(y_test[:,0])
        y_data = np.reshape(np.array(y_data), (len(y_data), 2))

        y_probs = self.model.predict(x_test)[0]
        y_hat = y_probs.argmax(1)

        y_test = y_data[:, 1]
        y_probs = y_probs[:, 1]

        model_acc = accuracy_score(y_test, y_hat)
        model_f1 = f1_score(y_test, y_hat)
        lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_probs)
        model_auc = auc(lr_recall, lr_precision)
        model_precision = precision_score(y_test, y_hat), 
        model_recall = recall_score(y_test, y_hat)
        model_cm = confusion_matrix(y_test, y_hat)
        
        print('=================================')
        print("T_FCN_ADDA: ")
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
