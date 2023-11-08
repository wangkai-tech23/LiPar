import os
import shutil
import warnings
warnings.filterwarnings("ignore")

# Create folders to store images
Train_Dir = '../img_label/train/'
Val_Dir = '../img_label/val/'
Test_Dir = '../img_label/test/'

def mymovefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" %(srcfile))
    else:
        fpath, fname = os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.move(srcfile, dstfile)
        #print ("move %s -> %s"%(srcfile,dstfile))


allimgs = []
subdir = 'RPM_normal'
start = 1
Dir = Train_Dir

print(subdir)
sum = len(os.listdir(os.path.join(Dir, 'normal')))
sum_s = len(os.listdir(os.path.join(Dir, subdir)))
print(sum_s)
end = start + sum_s

# rename other normal data ID to the continuation of normal ID, and move files
for i in range(end-1, start-1, -1):
    filename = str(i) + '.png'
    new_name = str(i + sum) + '.png'
    img = os.path.join(Dir, subdir, new_name)
    os.rename(os.path.join(Dir, subdir, filename), img)
    dest_path = img.replace(subdir, 'normal')
    mymovefile(img, dest_path)


# rename start ID to 0
# for i in range(start, end):
#     filename = str(i) + '.png'
#     new_name = str(i-start) + '.png'
#     img = os.path.join(Dir, subdir, new_name)
#     os.rename(os.path.join(Dir, subdir, filename), img)

# i = 0
# for filename in os.listdir(os.path.join(Dir, subdir)):
#     new_name = str(i) + '.png'
#     img = os.path.join(Dir, subdir, new_name)
#     os.rename(os.path.join(Dir, subdir, filename), img)
#     i += 1




print('Finish concat')


