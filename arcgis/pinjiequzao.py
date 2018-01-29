import cv2
import numpy as np

# img1_path = 'D:/pic_div/2015/35.png'
# img2_path = 'D:/pic_div/2015/36.png'
# img3_path = 'D:/pic_div/2015/37.png'
# img4_path = 'D:/pic_div/2015/38.png'
# img5_path = 'D:/pic_div/2015/39.png'
# img1 = cv2.imread(img1_path)
# img2 = cv2.imread(img2_path)
# img3 = cv2.imread(img3_path)
# img4 = cv2.imread(img4_path)
# img5 = cv2.imread(img5_path)
# img = np.column_stack((img1, img2, img3, img4, img5))
# cv2.imwrite('C:/Users/qq619/Desktop/8.jpg', img)


# img1_path = 'C:/Users/qq619/Desktop/1.jpg'
# img2_path = 'C:/Users/qq619/Desktop/2.jpg'
# img3_path = 'C:/Users/qq619/Desktop/3.jpg'
# img4_path = 'C:/Users/qq619/Desktop/4.jpg'
# img5_path = 'C:/Users/qq619/Desktop/5.jpg'
# img6_path = 'C:/Users/qq619/Desktop/6.jpg'
# img7_path = 'C:/Users/qq619/Desktop/7.jpg'
# img8_path = 'C:/Users/qq619/Desktop/8.jpg'
# img1 = cv2.imread(img1_path)
# img2 = cv2.imread(img2_path)
# img3 = cv2.imread(img3_path)
# img4 = cv2.imread(img4_path)
# img5 = cv2.imread(img5_path)
# img6 = cv2.imread(img6_path)
# img7 = cv2.imread(img7_path)
# img8 = cv2.imread(img8_path)
# img = np.row_stack((img1, img2, img3, img4, img5, img6, img7, img8))
# cv2.imwrite('C:/Users/qq619/Desktop/2015.jpg', img)

# img = cv2.imread('C:/Users/qq619/Desktop/change_detection.jpg')
# result = cv2.medianBlur(img, 15)
# cv2.imwrite('C:/Users/qq619/Desktop/quzao15.jpg', result)

init_mat = np.zeros([8000, 5000, 3])
for i in range(8):
    for j in range(5):
        img_path = 'D:/pic_div/2015/' + str(i*5+j) + '.png'
        img = cv2.imread(img_path)
        init_mat[i * 1000: (i + 1) * 1000, j * 1000:  (j + 1) * 1000, :] = img
cv2.imwrite('C:/Users/qq619/Desktop/20152.jpg', init_mat)

# for i in range(40):
#     img_path = 'C:/Users/qq619/Desktop/img_meanshift/' + str(i) + '.jpg'
#     img = cv2.imread(img_path)
#     result = cv2.medianBlur(img, 9)
#     cv2.imwrite('C:/Users/qq619/Desktop/quzao/' + str(i) + '.jpg', result)
