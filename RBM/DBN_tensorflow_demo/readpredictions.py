import numpy as np
import cv2

predictions = np.load("/media/files/yp/rbm/output/predictions/predictions.npy")
# predictions = np.load("/media/files/yp/rbm/predictions.npy")
length, classes = np.shape(predictions)
predictions_label = []
for i in range(length):
    if predictions[i][0] > predictions[i][1]:
        predictions_label.append(255)
    else:
        predictions_label.append(0)
predictions_label = np.array(predictions_label)
print 'read file done'
img_path = '/media/files/yp/rbm/label07.png'
img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = np.shape(img_gray)
predictions_img = predictions_label.reshape(height, width)
# cv2.imshow('predictions_img', predictions_img)
cv2.imwrite('/media/files/yp/rbm/label07_predictions_img.jpg', predictions_img)
