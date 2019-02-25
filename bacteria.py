import os
from skimage import io
from skimage.transform import resize
import numpy as np
import json
from keras import backend as K


def get_partition_indices(batch_size=64, train_percentage=0.8):
    dataset_size = 29272
    index_list = list(range(dataset_size))
    np.random.shuffle(index_list)
    split_point = int(dataset_size*train_percentage//batch_size*batch_size)
    end_point = int(dataset_size//batch_size*batch_size)
    train_indices = index_list[:split_point]
    test_indices = index_list[split_point:end_point]
    return train_indices, test_indices



def batch_generator(indices,batch_size=64):
    DATASET_ROOT = 'segment_enumeration_dataset'
    NUMPY_DATASET_FOLDER = 'SegmentsNumpy'

    json_string = open(os.path.join(DATASET_ROOT,'enumeration_segments.json')).read()
    json_dict = json.loads(json_string)

    labels = np.load(os.path.join(DATASET_ROOT,'labels.npy'))

    while True:
        #at the end of epoch, reshufle
        np.random.shuffle(indices)
        for i in range(0,len(indices),batch_size):
            batch_images = []
            batch_labels = []
            for j in indices[i:i+batch_size]:
                segment_path = 'Segment_n_' + str(j) + '.npy'
                img = np.load(os.path.join(DATASET_ROOT,NUMPY_DATASET_FOLDER,segment_path))
                label = labels[j]
                batch_images.append(img)
                batch_labels.append(label)

            x = np.array(batch_images)
            y = np.array(batch_labels)
            #Find which format to use (depends on the backend), and compute input_shape
            img_rows, img_cols, channels = 64, 64, 3
            if K.image_data_format() == 'channels_first':
                x = x.reshape(x.shape[0], channels, img_rows, img_cols)
            else:
                x = x.reshape(x.shape[0], img_rows, img_cols, channels)

            yield x, y


#Find which format to use (depends on the backend), and compute input_shape

#dataset resolution
img_rows, img_cols, channels = 64, 64, 3

if K.image_data_format() == 'channels_first':
    input_shape = (channels, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, channels)

#Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
#Two hidden layers
nn = Sequential()
nn.add(Conv2D(64, 3, 3, activation='relu', input_shape=input_shape))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Conv2D(32, 3, 3, activation='relu'))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Flatten())
nn.add(Dense(16, activation='relu'))
nn.add(Dense(10, activation='softmax'))

#Compile the NN
nn.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

# generators
train_indices, validation_indices = get_partition_indices(batch_size=64, train_percentage=0.8)
train_gen = batch_generator(train_indices,batch_size=64)
validation_gen = batch_generator(validation_indices,batch_size=64)

#Start training
history = nn.fit_generator(
                train_gen,
                steps_per_epoch=len(train_indices)/64,
                validation_data=validation_gen,
                validation_steps=len(validation_indices)/64,
                epochs=100)


"""
#Evaluate the model with test set
score = nn.evaluate(x_test, y_test, verbose=0)
print('test loss:', score[0])
print('test accuracy:', score[1])

##Store Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Accuracy plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('mnist_cnn_accuracy.pdf')
plt.close()
#Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('mnist_cnn_loss.pdf')
"""
