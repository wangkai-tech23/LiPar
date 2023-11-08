import os
import shutil
import warnings
warnings.filterwarnings("ignore")

# Create folders to store images
Train_Dir = '../data_sequential_img/train/'
Val_Dir = '../data_sequential_img/val/'
Test_Dir = '../data_sequential_img/test/'

def mymovefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" %(srcfile))
    else:
        fpath, fname = os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.move(srcfile, dstfile)
        #print ("move %s -> %s"%(srcfile,dstfile))

for subdir in os.listdir(Train_Dir):
    # print(subdir)
    if subdir == '.DS_Store':
        pass
    else:
        print(subdir)
        subimgs = []
        sum = len(os.listdir(os.path.join(Train_Dir, subdir)))
        Numbers = 3 * sum // 10  # size of test&val set (30%)
        start = sum - Numbers

        for i in range(start, sum):
            filename = str(i) + '.png'
            filepath = os.path.join(Train_Dir, subdir, filename)
            subimgs.append(filepath)

        for img in subimgs[:]:
            dest_path = img.replace(Train_Dir, Val_Dir)
            mymovefile(img, dest_path)


print('Finish creating val and test set')


