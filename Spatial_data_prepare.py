# 准备静态空间流数据
import numpy as np
import xlrd
import cv2


def detect_face(img):
    face_cascade = cv2.CascadeClassifier("cascade.xml")
    facesrects = face_cascade.detectMultiScale(img, 1.1, 2, minSize=(30, 30))
    return facesrects[0]


label_dict = {"disgust": 0, "happiness": 1, "repression": 2, "surprise": 3, "others": 4}
label = np.zeros((247, 1))
data = np.zeros((247, 48*48))
book = xlrd.open_workbook("CASME2-coding-20181220.xlsx")
sheet = book.sheet_by_index(0)
row_length = sheet.nrows

for i in range(1, row_length):
    row = sheet.row_values(i)
    path = "/media/like/Documents/CASME-II/CASME2-RAW/sub" + row[0] + "/" + row[1] + "/" + "img" + str(int(row[9])) + ".jpg"
    bgr = cv2.imread(path)
    faceRect = detect_face(bgr)
    bgr = bgr[faceRect[1]:faceRect[1] + faceRect[3], faceRect[0]:faceRect[0] + faceRect[2]]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    data[i-1, :] = cv2.resize(gray, (48, 48)).reshape(1, -1)
    label[i-1, 0] = label_dict[row[8]]
    print(path)

np.save('Spatial_data', data)
np.save('label', label)
