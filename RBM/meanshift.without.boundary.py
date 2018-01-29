# coding=utf-8
import numpy as np
import cv2
import random
import sys
import time

# if (len(sys.argv) < 2):
#     print 'Error'
# else:
#     Input_Image = sys.argv[1];
#
# # load image in "original_image"
#
# K = cv2.imread(Input_Image, 1)

# load image in "original_image"

start_time = time.time()

K = cv2.imread("/media/files/yp/rbm/15/1507.png", 1)

row = K.shape[0]
col = K.shape[1]

J = row * col
Size = row, col, 3
R = np.zeros(Size, dtype=np.uint8)
D = np.zeros((J, 5))
arr = np.array((1, 3))

# cv2.imshow("image", K)

counter = 0
iter = 1.0

threshold = 60
current_mean_random = True
current_mean_arr = np.zeros((1, 5))
below_threshold_arr = []

# 将图像 K[rows][col] 转换至特征空间 D. D 的维度为 [rows*col][5]
for i in range(0, row):
    for j in range(0, col):
        arr = K[i][j]

        for k in range(0, 5):
            if (k >= 0) & (k <= 2):
                D[counter][k] = arr[k]
            else:
                if k == 3:
                    D[counter][k] = i
                else:
                    D[counter][k] = j
        counter += 1

while len(D) > 0:
    # print J
    print len(D)
    # 从特征空间中选择一个随机行并将其分配为当前的均值
    if current_mean_random:
        current_mean = random.randint(0, len(D) - 1)
        for i in range(0, 5):
            current_mean_arr[0][i] = D[current_mean][i]
    below_threshold_arr = []
    for i in range(0, len(D)):
        # print "Entered here"
        ecl_dist = 0
        color_total_current = 0
        color_total_new = 0
        # 找出随机选择的行的欧氏距离 即与所有其他行的当前均值
        for j in range(0, 5):
            ecl_dist += ((current_mean_arr[0][j] - D[i][j]) ** 2)

        ecl_dist = ecl_dist ** 0.5

        # 检查计算的距离是否在阈值内。 如果是，采取这些行，并将其添加到列表 below_threshold_arr

        if ecl_dist < threshold:
            below_threshold_arr.append(i)
            # print "came here"

    mean_R = 0
    mean_G = 0
    mean_B = 0
    mean_i = 0
    mean_j = 0
    current_mean = 0
    mean_col = 0

    # 对于找到并放置在below_threshold_arr列表中的所有行，计算红色，绿色，蓝色和索引位置的平均值

    for i in range(0, len(below_threshold_arr)):
        mean_R += D[below_threshold_arr[i]][0]
        mean_G += D[below_threshold_arr[i]][1]
        mean_B += D[below_threshold_arr[i]][2]
        mean_i += D[below_threshold_arr[i]][3]
        mean_j += D[below_threshold_arr[i]][4]

    mean_R = mean_R / len(below_threshold_arr)
    mean_G = mean_G / len(below_threshold_arr)
    mean_B = mean_B / len(below_threshold_arr)
    mean_i = mean_i / len(below_threshold_arr)
    mean_j = mean_j / len(below_threshold_arr)

    # 找出这些平均值与当前均值的距离，并将其与iter进行比较

    mean_e_distance = ((mean_R - current_mean_arr[0][0]) ** 2 + (mean_G - current_mean_arr[0][1]) ** 2 + (
    mean_B - current_mean_arr[0][2]) ** 2 + (mean_i - current_mean_arr[0][3]) ** 2 + (
                       mean_j - current_mean_arr[0][4]) ** 2)

    mean_e_distance = mean_e_distance ** 0.5

    nearest_i = 0
    min_e_dist = 0
    counter_threshold = 0
    # 如果小于iter，找到下面的那个行，其中i，j最接近mean_i，mean_j
    # 这是因为mean_i和mean_j可能是十进制值，它们不对应于Image数组中的实际像素

    if mean_e_distance < iter:
        # print "Entered here"
        '''    
        for i in range(0, len(below_threshold_arr)):
            new_e_dist = ((mean_i - D[below_threshold_arr[i]][3])**2 + (mean_j - D[below_threshold_arr[i]][4])**2 + (mean_R - D[below_threshold_arr[i]][0])**2 +(mean_G - D[below_threshold_arr[i]][1])**2 + (mean_B - D[below_threshold_arr[i]][3])**2)
            new_e_dist = new_e_dist**0.5
            if(i == 0):
                min_e_dist = new_e_dist
                nearest_i = i
            else:
                if(new_e_dist < min_e_dist):
                    min_e_dist = new_e_dist
                    nearest_i = i
'''
        new_arr = np.zeros((1, 3))
        new_arr[0][0] = mean_R
        new_arr[0][1] = mean_G
        new_arr[0][2] = mean_B

        # 找到时，使用下面的threshold_arr中的行的颜色将下面的所有行着色，其中的下一行的颜色为i，j最接近mean_i和mean_j
        for i in range(0, len(below_threshold_arr)):
            index1 = int(D[below_threshold_arr[i]][3])
            index2 = int(D[below_threshold_arr[i]][4])
            R[index1][index2] = new_arr

            # 现在也不要使用那些已经着色一次的行。

            D[below_threshold_arr[i]][0] = -1
        current_mean_random = True
        new_D = np.zeros((len(D), 5))
        counter_i = 0

        for i in range(0, len(D)):
            if D[i][0] != -1:
                new_D[counter_i][0] = D[i][0]
                new_D[counter_i][1] = D[i][1]
                new_D[counter_i][2] = D[i][2]
                new_D[counter_i][3] = D[i][3]
                new_D[counter_i][4] = D[i][4]
                counter_i += 1

        D = np.zeros((counter_i, 5))

        counter_i -= 1
        for i in range(0, counter_i):
            D[i][0] = new_D[i][0]
            D[i][1] = new_D[i][1]
            D[i][2] = new_D[i][2]
            D[i][3] = new_D[i][3]
            D[i][4] = new_D[i][4]

    else:
        current_mean_random = False

        current_mean_arr[0][0] = mean_R
        current_mean_arr[0][1] = mean_G
        current_mean_arr[0][2] = mean_B
        current_mean_arr[0][3] = mean_i
        current_mean_arr[0][4] = mean_j

        # cv2.imwrite("image"+ str(len(below_threshold_arr)) +".png", K)

        # if(len(total_array) >= 40000):
        # break
# cv2.imshow("finalImage", R)
cv2.imwrite("finalImage.png", R)
# print  arr

execution_time = (time.time() - start_time)
print 'Execution finished in ' + str(round(execution_time, 2)) + 'secs'

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#Axis3D(I)'''
