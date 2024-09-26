# LiPar

LiPar is a lightweight parallel learning model for practical in-vehicle network intrusion detection. LiPar has great detection performance, running efficiency, and lightweight model size, which can be well adapted to the in-vehicle environment practically and protect the in-vehicle CAN bus security.

You can see the details in our paper **LiPar: A Lightweight Parallel Learning Model for Practical In-Vehicle Network Intrusion Detection**. ([arXiv:2311.08000](https://arxiv.org/abs/2311.08000))

## The Dataset

The dataset we used is [Car-Hacking Dataset](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) (you can fine the details of this dataset by the link). We also upload the data files in `./OriginalDataset/`. Due to the limitation of upload file size, we compressed each data file into a `.rar` file. You can get the original data by unzipping the files. 

## The Data Processing

The codes for data processing are uploaded in `./DataProcessing/`. Here are the steps for our data processing:

1. Data preprocessing and data cleaning: `data_process.py` is used to process attack datasets, including `DoS_Attack_dataset`, `Fuzzy_Attack_dataset`, `Spoofing_the_RPM_gauge_dataset` and `Spoofing_the_drie_gear_dataset`. `data_process_normal.py` is only used to process `normal` dataset. You can change the `file_path` and the new file name in `df.to_csv` function in the code to preprocess the datasets one by one. Then, it will generate five preprocessed data files respectively. The generated datasets can be found in the compressed file `./DataProcessing/PreprocessedData.rar`. 
2. Image data generating: `img_generator_seq.py` is used to process one-dimensional data sequentially into RGB image data. For attack datasets, there are both attack messages and normal messages. So, if an image is composed entirely of normal messages, we will label it as normal image. Otherwise, we will label it as attack image. Therefore, each attack dataset can generate two sets of image. You can set different directory addresses in `image_path_attack` and `image_path_normal` to store the generated normal and attack images. For each set of new generated images, the images will be named from 1 to `n` (`n` is the total number of images in the set) by sequence. When you finish processing one data file, you can change the filename and path in `file_path` to process other data files. The files we used are the preprocessed data files obtained in the previous step. For normal dataset, certainly, the program will only generate one set of normal image. At the end, you will obtain 9 sets of images in different directories.
3. Dataset partitioning: The directory we used to store all images is `./data_sequential_img/train/`. Then, we need to divide the train set, validation set and the test set from all the image data. `split_trainset.py` is used to divide 30% of the total image data into validation and test set, the directory of which named `./data_sequential_img/val/`. Futhermore, `split_testset.py` is used to divide $\frac{1}{3}$ of the images in `val` set into test set named `./data_sequential_img/test/`. Finally, the ratio of images in the training set, validation set, and test set is `7:2:1`. Also, you can change the path and directory name to anything you want by modifying `Train_Dir`, `Val_Dir` and `Test_Dir` in the code.

## The Models Training and Testing

For learning-based models, first of all, the model should be trained, and then we can obtain the optimal weights and parameters value in the model through training. Finally, the weights and parameters obtained are loaded into the model for testing, so as to verify the final detection performance of the model.

Therefore, we construct two programs for each model, respectively for model training process and testing process. You can find files like `train.py`, `train_LSTM.py` or  `train_CANet.py` are for model training process, and files like `predict.py`, `predict_LSTM.py` or `predict_CANet.py` are for model testing process.

Take STParNet as an example, all of the files for model construction, model training and testing are stored under the same directory `./STParNet/`. Before training, you should import the model you want to train by code like `from par_DW_LSTM4 import ParDWLSTM`. Here, the `par_DW_LSTM4` is the name of python file which contructs the model and `ParDWLSTM` is the main class of the model. You should also set the path of input images. The `image_path` refers to the directory where all the images are located, the `root`  in `train_dataset` refers to the directory of training set and the `root` in `validate_dataset` refers to the directory of validation set. Besides, `save_path` refers to the path for saving the file of optimal weights and parameters, which is in .pth format.

Similarly, when testing models, we may need to change the data path used for testing at `image_path` and the path to load the trained model parameter file at `model_weight_path` in `predict.py`.

## The Model Size Statistics

The codes for model size testing are uploaded in `./MemoryTest/`. Running each program can obtain the memory consumption of each part of the model.

## The Resource Adaptation Algorithm

The code of resource adaption algorithm is uploaded in `./ResouceAdaptationModel/`. The specific parameters in code need to be set according to the actual requirements.
















