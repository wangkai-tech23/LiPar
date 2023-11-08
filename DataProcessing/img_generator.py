import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import QuantileTransformer
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

file_path = 'normal_dataset.csv'

# Read dataset
df = pd.read_csv(file_path)
# df

# Transform all features into the scale of [0,1]
numeric_features = df.dtypes[df.dtypes != 'object'].index
scaler = QuantileTransformer()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Multiply the feature values by 255 to transform them into the scale of [0,255]
df[numeric_features] = df[numeric_features].apply(lambda x: (x*255))
# df.describe()

df0 = df[df['Label'] == 'R'].drop(['Label'], axis=1)
# df1 = df[df['Label'] == 'T'].drop(['Label'], axis=1)

# df1 = df[df['Label'] == 'RPM'].drop(['Label'], axis=1)
# df2 = df[df['Label'] == 'gear'].drop(['Label'], axis=1)
# df3 = df[df['Label'] == 'DoS'].drop(['Label'], axis=1)
# df4 = df[df['Label'] == 'Fuzzy'].drop(['Label'], axis=1)

# # Generate 9*9 color images for class 0 (Normal)
# # Change the numbers 9 to the number of features n in your dataset if you use a different dataset, reshape(n,n,3)

count = 0
ims = []

image_path = "../data_img/train/0/"
os.makedirs(image_path)

for i in range(0, len(df0)):
    count = count+1
    if count < 27:
        im = df0.iloc[i].values
        ims = np.append(ims, im)
    else:
        im = df0.iloc[i].values
        ims = np.append(ims, im)
        ims = np.array(ims).reshape(9, 9, 3)
        array = np.array(ims, dtype=np.uint8)
        new_image = Image.fromarray(array)
        new_image.save(image_path+str(i//27)+'.png')
        count = 0
        ims = []


# count = 0
# ims = []
#
# image_path = "../data_img/train/RPM_attack/"
# os.makedirs(image_path)
#
# for i in range(0, len(df1)):
#     count = count+1
#     if count < 27:
#         im = df1.iloc[i].values
#         ims = np.append(ims, im)
#     else:
#         im = df1.iloc[i].values
#         ims = np.append(ims, im)
#         ims = np.array(ims).reshape(9, 9, 3)
#         array = np.array(ims, dtype=np.uint8)
#         new_image = Image.fromarray(array)
#         new_image.save(image_path + str(i // 27) + '.png')
#         count = 0
#         ims = []


