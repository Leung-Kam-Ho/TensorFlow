from cmath import log
from pickletools import optimize
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers

data_url = 'https://raw.githubusercontent.com/Apress/artificial-neural-networks-with-tensorflow-2/main/ch02/Churn_Modelling.csv' 
data=pd.read_csv(data_url)
data = shuffle(data)
#print(data.isnull().sum())
drop_list =  ['CustomerId', 'Surname','RowNumber', 'Exited']
x = data.drop(labels = drop_list,axis = 1)
y = data['Exited']
label = LabelEncoder()

#tran male female to 0 and 1
x['Gender'] = label.fit_transform(x['Gender'])
x['Geography'] = label.fit_transform(x['Geography'])
x = pd.get_dummies(x, drop_first=True, columns=['Geography'])

#standardlize data
#z = (x - mu)/s 
scaler = StandardScaler()
x = scaler.fit_transform(x)

#print(x)

#Split data
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.3)
#print(x_train.shape[1])
print(data)
#print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
with tf.device("/CPU:0"):
    model = keras.models.Sequential()

    #first layer with 128 node
    model.add(keras.layers.Dense(128, activation = 'relu', input_dim = x_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(keras.layers.Dense(64, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(keras.layers.Dense(32, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(keras.layers.Dense(16, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(keras.layers.Dense(1, activation = 'sigmoid'))
    model.summary()

    model.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics = ['accuracy'])

    from tensorflow.keras.callbacks import History
    history = History()
    import datetime, os
    logdir = os.path.join("log",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir,histogram_freq = 1)
    epoch = 400

    r = model.fit(x_train, y_train, batch_size = 32, epochs = epoch, validation_data = [x_test,y_test], callbacks = [tensorboard_callback])
    model.save("saved_model/example_1.h5")
    test_scores = model.evaluate(x_test, y_test)
    epoch_range = range(1, epoch+1)
    print('Test Loss: ', test_scores[0])
    print('Test accuracy: ', test_scores[1] * 100)
    plt.plot(epoch_range, r.history['accuracy'])
    plt.plot(epoch_range, r.history['val_accuracy'])
    
    plt.title('Classification Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.show()
    # Plot training & validation loss values
    plt.plot(epoch_range,r.history['loss'])
    plt.plot(epoch_range, r.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    

    #model = tf.keras.models.load_model('Binary_model.h5')
    y_pred = (model.predict(x_test) > 0.5).astype("int32")
    #y_pred = np.argmax(y_pred, axis=-1)
    
    #print(y_pred)
    cf = confusion_matrix(y_test, y_pred)
    print(cf)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_pred))
    customer = model.predict([[615, 1, 22, 5, 20000, 5, 1, 1,60000, 0, 0]])
    if customer[0] == 1:
        print ("Customer is likely to leave")
    else:
        print ("Customer will stay")
    plt.show()


   

