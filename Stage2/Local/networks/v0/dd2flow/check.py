import os
import cv2

root = "training_dataset/N/"
for f in os.listdir(root):
    cdepth_path = os.path.join("training_dataset/D/", f[0:-4]+".png")
    cdepth = cv2.imread(cdepth_path, -1)
    print(f)
    print(cdepth.shape)