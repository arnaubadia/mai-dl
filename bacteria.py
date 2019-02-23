import os
from skimage import io
from skimage.transform import resize
import numpy as np
import json

DATASET_ROOT = 'segment_enumeration_dataset'

json_string = open(os.path.join(DATASET_ROOT,'enumeration_segments.json')).read()
json_dict = json.loads(json_string)

all_images = []
labels = []
for segment_id in json_dict:
    image_path = os.path.join(DATASET_ROOT,json_dict[segment_id]['Segment Relative Path'])
    img = io.imread(image_path)
    # resize the image to 128 x 128. Resize function also normalizes the values
    img = resize(img, (128,128), anti_aliasing=True)
    label = json_dict[segment_id]['data']['segment_type']['data']
    all_images.append(img)
    labels.append(label)

# need to convert to numpy array
x = np.array(all_images)
y = np.array(labels)
