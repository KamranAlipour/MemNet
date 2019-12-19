import numpy as np
import cv2
import glob

for f in glob.glob('frames_split_1024_train/*'):
    print(f)
    img = cv2.imread(f)
    np.save('npy_'+f.split('.')[0]+'.npy',img)
