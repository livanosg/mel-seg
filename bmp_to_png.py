from glob import glob

import cv2

ph2_derm = sorted(glob(pathname="/home/livanosg/projects/mel-seg/data/ph2/data/**/**_Dermoscopic_Image/**.bmp", recursive=True))
ph2_lesion = sorted(glob(pathname="/home/livanosg/projects/mel-seg/data/ph2/data/**/**_lesion/**.bmp", recursive=True))

for image in ph2_derm:
    cv2.imwrite(image.replace("bmp", "png"), cv2.imread(image))

for image in ph2_lesion:
    cv2.imwrite(image.replace("bmp", "png"), cv2.imread(image))
