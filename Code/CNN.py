import tensorflow as tf
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score
from load_data import LoadData_slice
from keras.utils import np_utils
# =============================================================================
# from keras.models import Sequential
# from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, GlobalMaxPooling1D
# =============================================================================


##################################################################
## model 구성하기 ##
## 사용자 한 명씩 인증을 실시 ##


# subject 수
subjects = 11 

total_accuracy = []
total_precision = []
total_recall = []


for subject in range(subjects):

    inputs = tf.keras.Input(shape = (170, 1))
    conv_1 = tf.keras.layers.Conv1D(filters=4, kernel_size=5, strides=1, activation='relu', kernel_initializer ='he_normal')(inputs)
    max_1 = tf.keras.layers.MaxPool1D(2)(conv_1) 
    conv_2 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, activation='relu',kernel_initializer ='he_normal')(max_1) 
    max_2 = tf.keras.layers.MaxPool1D(2)(conv_2)
    conv_3 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation='relu',kernel_initializer ='he_normal')(max_2) 
    max_3 = tf.keras.layers.MaxPool1D(2)(conv_3) 
    conv_4 = tf.keras.layers.Conv1D(filters=16, kernel_size=5, strides=1, activation='relu',kernel_initializer ='he_normal')(max_3) 

    G_max_1 = tf.keras.layers.GlobalMaxPooling1D(data_format='channels_last')(conv_4) 

    den_1 = tf.keras.layers.Dense(32, activation='tanh',kernel_initializer ='he_normal')(G_max_1) 
    den_2 = tf.keras.layers.Dense(2, activation='softmax',kernel_initializer ='he_normal')(den_1) 

    #####################################################################
    ## 학습에 대한 설정 ##

    model = tf.keras.models.Model(inputs = inputs, outputs = den_2)
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    ######################################################################
    
    
    # train model 
    train, test = LoadData_slice(subject)

    train_X = train[:,:-1]
    train_Y = train[:,-1]

    train_X = np.reshape(train_X,(train_X.shape[0],train_X.shape[1],1))
    train_Y = np_utils.to_categorical(train_Y,num_classes=2) # binary class로 설정

    history = model.fit(train_X, train_Y,
                        epochs=50, batch_size=32, verbose=0)
                        #validation_data=(X_validation, Y_validation),
                        #callbacks=[cb_checkpoint, cb_early_stopping])
                        # validation data가 있을 경우 사용
    model.summary()

    test_X = test[:,:-1]
    test_Y = test[:,-1]

    test_X = np.reshape(test_X,(test_X.shape[0],test_X.shape[1],1))

    y_predict = model.predict(test_X)
    y_pred = np.argmax(y_predict, axis=1)

    print("subject {}\nAccuray : {}\nRecall : {}\nPrecision : {}\n".format(subject,accuracy_score(test_Y, y_pred),recall_score(test_Y, y_pred , average="macro"),precision_score(test_Y, y_pred , average="macro")))
