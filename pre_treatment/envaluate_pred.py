# coding=utf-8
import cv2

# pred_path = "C:/Users/qq619/Desktop/combined_MS/gf2457/combined_MS_prediction.jpg"
# label_path = "C:/Users/qq619/Desktop/label2457.jpg"
pred_path = "C:/Users/qq619/Desktop/SAE_softmax_V/2016gf2457.jpg"
label_path = "E:/result/pre_post_treatment/score/label2457.jpg"

pred_img = cv2.imread(pred_path, 0)
print(pred_img.shape)
label_img = cv2.imread(label_path, 0)
print(label_img.shape)

# 防止尺寸不匹配，det 为图片尺寸的差值
row1, col1 = pred_img.shape
row2, col2 = label_img.shape
if row1 > row2:
    det = row1 - row2
    pred_img = pred_img[det:, :]
elif row1 < row2:
    det = row2 - row1
    label_img = label_img[det:, :]
if col1 > col2:
    det = col1 - col2
    pred_img = pred_img[:, :-det]
elif col1 < col2:
    det = col2 - col1
    label_img = label_img[:, :-det]

TP = 0
FP = 0
TN = 0
FN = 0

row, col = pred_img.shape
for i in range(row):
    for j in range(col):
        if pred_img[i][j] == 255 and label_img[i][j] == 255:
            # 变化的区域被检测到，即正检数 TP
            TP = TP + 1
        if pred_img[i][j] == 255 and label_img[i][j] != 255:
            # 未变化的区域被检测为变化，即误检数 FP
            FP = FP + 1
        if pred_img[i][j] != 255 and label_img[i][j] == 255:
            # 变化的区域被检测为未变化，即漏检数 FN
            FN = FN + 1
        if pred_img[i][j] != 255 and label_img[i][j] != 255:
            # 未变化的区域被检测到，即负检数 TN
            TN = TN + 1

precision = TP * 1.0 / (TP + FP)
# 精确率，正确检测到的变化区域 占 预测为变化的区域 的比重
recall = TP * 1.0 / (TP + FN)
# 召回率，正确检测到的变化区域 占 标签为变化的区域 的比重
f1 = (2 * precision * recall) / (precision + recall)
# F-measure，精确率和召回率的加权调和平均，这里取 F1
print("像素总数：" + str(row * col))
print("变化的区域被检测到，即正检数 TP ：" + str(TP))
print("未变化的区域被检测为变化，即误检数 FP ：" + str(FP))
print("变化的区域被检测为未变化，即漏检数 FN ：" + str(FN))
print("未变化的区域被检测到，即负检数 TN ：" + str(TN))
print("精确率：" + str(precision) + "  召回率：" + str(recall) + "  f1指数：" + str(f1))
