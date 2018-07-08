# coding=utf-8
import datetime
import sqlite3
import sys
import cv2
import numpy as np


# def view_bar(num, total):
#     rate = num * 1.0 / total
#     rate_num = int(rate * 100)
#     r = '\r[%s%s]%d%%' % ("*" * rate_num, "-" * (100 - rate_num), rate_num,)
#     sys.stdout.write(r)
#     sys.stdout.flush()


def getIdsInArea(rgb):
    conn = sqlite3.connect("area.db")
    c = conn.cursor()
    c.execute("SELECT id FROM area WHERE (r=? AND g=? AND b=?)", (int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return c.fetchall()


def insertIntoArea(rgb):
    conn = sqlite3.connect("area.db")
    c = conn.cursor()
    c.execute("INSERT INTO area VALUES (?,?,?,?,?,?,?)",
              (None, int(rgb[0]), int(rgb[1]), int(rgb[2]), 0, False, 1))
    conn.commit()
    conn.close()
    return getIdsInArea(rgb)


def insertIntoPixel(rgb, i, j, areaId):
    conn = sqlite3.connect("area.db")
    c = conn.cursor()
    c.execute("INSERT INTO pixel VALUES (?,?,?,?,?,?)",
              (int(rgb[0]), int(rgb[1]), int(rgb[2]), areaId, i, j))
    conn.commit()
    conn.close()


def getPixsInPixel(areaId):
    conn = sqlite3.connect("area.db")
    c = conn.cursor()
    c.execute("SELECT x, y FROM pixel WHERE area_id=?", [areaId])
    return c.fetchall()


def isPixInRange(i, j, pix, pix_distance):
    distance = pow(pow(pix[0] - i, 2) + pow(pix[1] - j, 2), 0.5)
    if distance <= pix_distance:
        return True
    return False


def updateTotalNum(areaId):
    conn = sqlite3.connect("area.db")
    c = conn.cursor()
    c.execute("UPDATE area SET total_num=total_num+1 WHERE id=?", [areaId])
    conn.commit()
    conn.close()


def convert2Areas(m_shift_img, pix_distance):
    count = 0
    # 将meanshift图像转变为对象，并将信息存储在数据库中
    row, col, dim = m_shift_img.shape
    for i in range(row):
        for j in range(col):
            count = count + 1
            print "第 " + str(count) + " 号像素："
            idxList = getIdsInArea(m_shift_img[i][j])
            if len(idxList) != 0:
                # 如果这个像素的颜色值在area表内，则找到对应的areaId
                # 通过areaId查询pixel表， 检查距离是否在pix_distance内
                # 若在pix_distance内，则将当前像素坐标添加到pixel表内，并使用刚刚查询到的areaId，并给area表的total_num加1
                # 若不在范围内，则新建一组信息，并将信息添加到两个表内（先area后pixel）
                print str(i) + " " + str(j) + " 的RGB值在area表内"
                IN_AREA_RANGE = False
                for areaId in idxList[0]:
                    pixelList = getPixsInPixel(areaId)
                    for pix in pixelList:
                        if isPixInRange(i, j, pix, pix_distance):
                            print "而且在" + str(areaId) + "号 area 的范围内，现在将 " + str(i) + " " + str(j) + " 插入表内"
                            insertIntoPixel(m_shift_img[i][j], i, j, areaId)
                            updateTotalNum(areaId)
                            IN_AREA_RANGE = True
                            break
                    if IN_AREA_RANGE:
                        break
                if not IN_AREA_RANGE:
                    print "但不在已有的 area 范围内，现在新建信息添加到两个表内"
                    areaId = insertIntoArea(m_shift_img[i][j])
                    insertIntoPixel(m_shift_img[i][j], i, j, areaId[0][0])
            else:
                # 如果像素的颜色值不在area表内，则新建一组信息，并将信息添加到两个表内（先area后pixel）
                print str(i) + " " + str(j) + " 不在area表内，现在新建信息添加到两个表内"
                areaId = insertIntoArea(m_shift_img[i][j])
                insertIntoPixel(m_shift_img[i][j], i, j, areaId[0][0])
            # view_bar(count, row * col)


def combineMS(m_shift_path, prediction_path, pix_distance):
    # 初始化清空两个表的内容
    conn = sqlite3.connect("area.db")
    c = conn.cursor()
    c.execute("DELETE FROM area")
    c.execute("DELETE FROM pixel")
    c.execute("DELETE FROM sqlite_sequence")
    conn.commit()
    conn.close()

    m_shift_img = cv2.imread(m_shift_path)
    prediction_img = cv2.imread(prediction_path)
    # 防止尺寸不匹配
    row1, col1, dim1 = m_shift_img.shape
    row2, col2, dim2 = prediction_img.shape
    if row1 > row2:
        det = row1 - row2
        m_shift_img = m_shift_img[det:, :, :]
    elif row1 < row2:
        det = row2 - row1
        prediction_img = prediction_img[det:, :, :]
    if col1 > col2:
        det = col1 - col2
        m_shift_img = m_shift_img[:, :-det, :]
    elif col1 < col2:
        det = col2 - col1
        prediction_img = prediction_img[:, :-det, :]

    convert2Areas(m_shift_img, pix_distance)  # 先将meanshift图像转为对象，并存储在数据库中

    prediction_img = prediction_img[:, :, 0]
    row, col = prediction_img.shape
    conn = sqlite3.connect("area.db")
    c = conn.cursor()
    for i in range(row):
        for j in range(col):
            if prediction_img[i][j] == 255:
                # 若像素是变化像素，寻找该像素在哪个对象内，增加该对象的变化像素数量
                c.execute("UPDATE area SET change_num=change_num+1 WHERE "
                          "id=(SELECT area_id FROM pixel WHERE x=? AND y=?)",
                          (i, j))

    result = np.zeros([row, col])
    c.execute("SELECT id FROM area WHERE is_changed=TRUE")
    changedIds = c.fetchall()
    for areaId in changedIds:
        c.execute("SELECT x, y FROM pixel WHERE area_id=?", areaId)
        changedPixs = c.fetchall()
        for pix in changedPixs:
            result[pix[0]][pix[1]] = 255

    # year = m_shift_path.split("gf")[0]
    # cv2.imwrite("/media/files/yp/rbm/pic_div/combine_MS/combined_MS_prediction_ref_to_" + year + ".jpg", result)
    return result, row, col


FIRST_RUN = True
start = datetime.datetime.now()
# m_shift_path1 = "/media/files/yp/rbm/pic_div/combine_MS/2015gf2457.TIF"
# m_shift_path2 = "/media/files/yp/rbm/pic_div/combine_MS/2016gf2457.TIF"
# prediction_path = "/media/files/yp/rbm/pic_div/combine_MS/2016gf2457.jpg"
m_shift_path1 = "C:/Users/qq619/Desktop/combined_MS/gf2457/2015gf2457.TIF"
m_shift_path2 = "C:/Users/qq619/Desktop/combined_MS/gf2457/2016gf2457.TIF"
prediction_path = "C:/Users/qq619/Desktop/combined_MS/gf2457/2016gf2457.jpg"
pix_distance = 10
if FIRST_RUN:
    conn = sqlite3.connect("area.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE pixel
          (r int, g int, b int, area_id int, x int, y int, PRIMARY KEY (x, y),
          FOREIGN KEY (area_id) REFERENCES area(id))''')
    c.execute('''CREATE TABLE area
          (id INTEGER PRIMARY KEY AUTOINCREMENT, r int, g int, b int, change_num int, is_changed boolean,
          total_num int)''')
    c.execute("CREATE TRIGGER AREA_UPDATE AFTER UPDATE OF change_num ON area "
              "WHEN new.change_num>old.total_num*0.3 BEGIN UPDATE area SET is_changed=TRUE; END;")
    print "数据库、表和触发器建立完成！"
    conn.commit()
    conn.close()
result1, row1, col1 = combineMS(m_shift_path1, prediction_path, pix_distance)
result2, row2, col2 = combineMS(m_shift_path2, prediction_path, pix_distance)
result = np.zeros([row1, col1])
for i in range(row1):
    for j in range(col1):
        if result1[i][j] == 0 and result2[i][j] == 0:
            continue
        else:
            result[i][j] = 255
cv2.imwrite("/media/files/yp/rbm/pic_div/combine_MS/combined_MS_prediction.jpg", result)
end = datetime.datetime.now()
print "耗时：{0}秒".format((end - start).seconds)
