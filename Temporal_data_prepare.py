# 准备动态时间流数据
from bob.ip.optflow.liu.sor import flow
import cv2
import xlrd
import numpy as np
from skimage import img_as_ubyte


def detect_face(img):
    face_cascade = cv2.CascadeClassifier("cascade.xml")
    facesrects = face_cascade.detectMultiScale(img, 1.1, 2, minSize=(30, 30))
    return facesrects[0]


data = np.zeros((247, 48, 48, 20))
book = xlrd.open_workbook("CASME2-coding-20181220.xlsx")
sheet = book.sheet_by_index(0)
row_length = sheet.nrows

for i in range(1, row_length):
    row = sheet.row_values(i)
    path = "/media/like/Documents/CASME-II/CASME2-RAW/sub" + row[0] + "/" + row[1]
    base_img = cv2.imread(path + "/img" + str(int(row[9])) + ".jpg")

    facerect = detect_face(base_img)

    for j in range(0, 10):

        cal_img1 = cv2.imread(path + "/img" + str(int(row[9]) - 5 + j) + ".jpg")
        cal_img2 = cv2.imread(path + "/img" + str(int(row[9]) - 4 + j) + ".jpg")

        cal_img1 = cv2.cvtColor(cal_img1, cv2.COLOR_BGR2GRAY)
        cal_img2 = cv2.cvtColor(cal_img2, cv2.COLOR_BGR2GRAY)
        test_vx, test_vy, test_warpi2 = flow(cal_img1, cal_img2)

        np.clip(test_vx, -1, 1, test_vx)
        test_vx = img_as_ubyte(test_vx)
        test_vx = test_vx[facerect[1]:facerect[1] + facerect[3], facerect[0]:facerect[0] + facerect[2]]
        test_vx = cv2.resize(test_vx, (48, 48))

        np.clip(test_vy, -1, 1, test_vy)
        test_vy = img_as_ubyte(test_vy)
        test_vy = test_vy[facerect[1]:facerect[1] + facerect[3], facerect[0]:facerect[0] + facerect[2]]
        test_vy = cv2.resize(test_vy, (48, 48))

        data[i - 1, :, :, j] = test_vx
        data[i - 1, :, :, j + 10] = test_vy

    print(path + "/img" + str(int(row[4])) + ".jpg")

np.save("Temporal_data", data)
