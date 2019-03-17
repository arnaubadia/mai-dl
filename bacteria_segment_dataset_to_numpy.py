import os
from skimage import io
from skimage.transform import resize
import numpy as np
import json
import keras

DATASET_ROOT = 'segment_enumeration_dataset'
NUMPY_DATASET_FOLDER = 'SegmentsNumpy128'

json_string = open(os.path.join(DATASET_ROOT,'enumeration_segments.json')).read()
json_dict = json.loads(json_string)

dataset_size = 29272
labels = []
for i in range(dataset_size):
    segment_id = 'Segment_n_' + str(i)
    image_path = os.path.join(DATASET_ROOT,json_dict[segment_id]['Segment Relative Path'])
    img = io.imread(image_path)
    # resize the image to 128 x 128. Resize function also normalizes the values
    img = resize(img, (128,128), anti_aliasing=True)
    label = json_dict[segment_id]['data']['segment_type']['data']
    # save image to numpy file
    np.save(os.path.join(DATASET_ROOT,NUMPY_DATASET_FOLDER,segment_id + '.npy'), img)
    # labels can be all stored in the same list
    labels.append(label)

# need to convert to numpy array
y = np.array(labels)
one_hot_y = keras.utils.to_categorical(y)
np.save(os.path.join(DATASET_ROOT, 'labels.npy'), one_hot_y)
