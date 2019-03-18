import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from skimage import io
from skimage.transform import resize
import numpy as np
import json
from keras import backend as K



#val percentage is a % of the train percentage
def get_partition_indices(batch_size=64, train_percentage=0.9, val_percentage=0.15):
    dataset_size = 29272
    index_list = list(range(dataset_size))
    np.random.shuffle(index_list)
    test_split_point = int(dataset_size*train_percentage//batch_size*batch_size)
    end_point = int(dataset_size//batch_size*batch_size)
    train_indices = index_list[:test_split_point]
    val_split_point = int(len(train_indices)*(1-val_percentage)//batch_size*batch_size)
    train_indices = index_list[:val_split_point]
    validation_indices = index_list[val_split_point:test_split_point]
    test_indices = index_list[test_split_point:end_point]
    return train_indices, validation_indices, test_indices

# get array of data in memory
def get_data(indices,batch_size=64):
    DATASET_ROOT = 'segment_enumeration_dataset'
    NUMPY_DATASET_FOLDER = 'SegmentsNumpy'

    json_string = open(os.path.join(DATASET_ROOT,'enumeration_segments.json')).read()
    json_dict = json.loads(json_string)

    labels = np.load(os.path.join(DATASET_ROOT,'labels.npy'))

    all_images = []
    for i in indices:
        segment_path = 'Segment_n_' + str(i) + '.npy'
        img = np.load(os.path.join(DATASET_ROOT,NUMPY_DATASET_FOLDER,segment_path))
        all_images.append(img)

    x = np.array(all_images)
    y = labels[indices]

    #Find which format to use (depends on the backend), and compute input_shape
    img_rows, img_cols, channels = 64, 64, 3
    if K.image_data_format() == 'channels_first':
        x = x.reshape(x.shape[0], channels, img_rows, img_cols)
    else:
        x = x.reshape(x.shape[0], img_rows, img_cols, channels)

    return x, y

# get batch generator
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
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
#Two hidden layers
nn = Sequential()
nn.add(Conv2D(128, (4, 4), activation='relu', input_shape=input_shape))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Conv2D(64, (3, 3), activation='relu'))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Conv2D(32, (3, 3), activation='relu'))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Flatten())
#nn.add(Dropout(0.3))
nn.add(Dense(128, activation='relu'))
#nn.add(Dropout(0.3))
nn.add(Dense(64, activation='relu'))
#nn.add(Dropout(0.3))
nn.add(Dense(8, activation='softmax'))


#Compile the NN
nn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# generators
train_indices, validation_indices, test_indices = get_partition_indices(batch_size=64,
                                        train_percentage=0.9, val_percentage=0.15)
x_train, y_train = get_data(train_indices,batch_size=64)
validation_gen = batch_generator(validation_indices,batch_size=64)

from keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=25, verbose=0, mode='min')
mcp_save = ModelCheckpoint('best-weights.hdf5', save_best_only=True, monitor='val_loss', mode='min')

# deal with class imbalance: assign class weights
y_train_only_labels = [np.where(r==1)[0][0] for r in y_train]
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                np.unique(y_train_only_labels), y_train_only_labels)

class_weights = {i:class_weights[i] for i in range(8)}

# data augmentation
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rotation_range=90,
    width_shift_range=0.01,
    height_shift_range=0.01,
    zoom_range= [0.95, 1.05],
    vertical_flip=True,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

#Start training
#history = nn.fit(x_train,y_train, batch_size=64, epochs=150, validation_split=0.15,
#                 class_weight=class_weights, callbacks=[earlyStopping, mcp_save])
#history = nn.fit(x_train,y_train, batch_size=64, epochs=150, validation_split=0.15,
#                 callbacks=[earlyStopping, mcp_save])

history = nn.fit_generator(datagen.flow(x_train, y_train, batch_size=64),
                           steps_per_epoch=len(x_train) / 64,
                           validation_data=validation_gen,
                           validation_steps=len(validation_indices)/64,
                           callbacks=[earlyStopping, mcp_save],
                           epochs=200)


#Restore the model with the best weights we found during training
nn.load_weights('best-weights.hdf5')

#Evaluate the model with test set
x_test, y_test = get_data(test_indices,batch_size=64)
score = nn.evaluate(x_test, y_test, verbose=0)
print('test loss:', score[0])
print('test accuracy:', score[1])


#Accuracy plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('bacteria_count_cnn_accuracy.pdf')
plt.close()
#Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('bacteria_count_cnn_loss.pdf')

#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
#Compute probabilities
Y_pred = nn.predict(x_test)
#Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
#Plot statistics
print( 'Analysis of results' )
target_names = ['1 colony', '2 colonies', '3 colonies', '4 colonies',
    '5 colonies', '6 colonies', 'Confluent growth', 'Outlier']
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

# from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

plot_confusion_matrix(np.argmax(y_test,axis=1), y_pred, target_names, normalize=False)
plt.savefig('bacteria_count_confusion_matrix.pdf')
plt.close()

plot_confusion_matrix(np.argmax(y_test,axis=1), y_pred, target_names, normalize=True)
plt.savefig('bacteria_count_confusion_matrix_normalized.pdf')


"""
#Saving model and weights
from keras.models import model_from_json
nn_json = nn.to_json()
with open('nn.json', 'w') as json_file:
        json_file.write(nn_json)
weights_file = "weights-bacteria_"+str(score[1])+".hdf5"
nn.save_weights(weights_file,overwrite=True)

#Loading model and weights
json_file = open('nn.json','r')
nn_json = json_file.read()
json_file.close()
nn = model_from_json(nn_json)
nn.load_weights(weights_file)
"""
