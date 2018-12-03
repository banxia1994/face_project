# coding:utf-8
#
from mtcnn_.src import detect_faces,show_bboxes
from PIL import Image
import cv2
import numpy as np

img = Image.open('./data/gallery/1.jpg')
img1 = np.asarray(img)
img2 = img1[:,:,(2,1,0)]
imgcv = cv2.imread('./data/gallery/1.jpg')
bounding_boxes,landmarks = detect_faces(img)
img = show_bboxes(img,bounding_boxes,landmarks)
img.save('a.jpg')
