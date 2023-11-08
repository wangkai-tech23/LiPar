import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import QuantileTransformer
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# change the path and filename to generate different types of attack (or normal) image data
file_path = 'RPM_dataset.csv'

# Read dataset
df = pd.read_csv(file_path)

# Transform all features into the scale of [0,1]
numeric_features = df.dtypes[df.dtypes != 'object'].index
scaler = QuantileTransformer()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Multiply the feature values by 255 to transform them into the scale of [0,255]
df[numeric_features] = df[numeric_features].apply(lambda x: (x*255))


# Generate 9*9 color images

count = 0
ims = []
label = 0
count_attack = 0
count_normal = 0

# set the directory for generated image data. 1 to 4 represent data from four files with different types of attack, 0 represents normal image data.
image_path_attack = "../data_sequential_img/train/4/"
image_path_normal = "../data_sequential_img/train/0_4/"
os.makedirs(image_path_attack)
os.makedirs(image_path_normal)


for i in range(0, len(df)):
    count = count+1
    if count < 27:
        if df.loc[i, 'Label'] == 'T':
            label = 1
        im = df.iloc[i].drop(['Label']).values
        ims = np.append(ims, im)
    else:
        if df.loc[i, 'Label'] == 'T':
            label = 1
        im = df.iloc[i].drop(['Label']).values
        ims = np.append(ims, im)
        ims = np.array(ims).reshape(9, 9, 3)
        array = np.array(ims, dtype=np.uint8)
        new_image = Image.fromarray(array)
        if label == 1:
            count_attack += 1
            new_image.save(image_path_attack+str(count_attack)+'.png')
        else:
            count_normal += 1
            new_image.save(image_path_normal + str(count_normal) + '.png')
        count = 0
        ims = []
        label = 0





