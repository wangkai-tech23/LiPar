import os
import shutil
import warnings
warnings.filterwarnings("ignore")

# Create folders to store images
Train_Dir = '../data_img/train/'
Val_Dir = '../data_img/val/'
Test_Dir = '../data_img/test/'

allimgs = []
for subdir in os.listdir(Val_Dir):
    print(subdir)
    for filename in os.listdir(os.path.join(Val_Dir, subdir)):
        filepath = os.path.join(Val_Dir, subdir, filename)
        print(filepath)
        allimgs.append(filepath)


def mymovefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" %(srcfile))
    else:
        fpath, fname = os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.move(srcfile, dstfile)
        #print ("move %s -> %s"%(srcfile,dstfile))


for img in allimgs:
    dest_path = img.replace(Val_Dir, Train_Dir)
    mymovefile(img, dest_path)
print('Finish concat')


