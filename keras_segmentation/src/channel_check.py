import cv2 

path = 'training/b1-99445_mask.png'

img = cv2.imread(path) 
print(img.shape)
print(img)
