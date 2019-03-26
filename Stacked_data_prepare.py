# 准备堆叠空间流数据
import numpy as np
import xlrd
import cv2


def detect_face(img):
    face_cascade = cv2.CascadeClassifier("cascade.xml")
    facesrects = face_cascade.detectMultiScale(img, 1.1, 2, minSize=(30, 30))
    return facesrects[0]


n = 3
data = np.zeros((247, 48, 48, n * n))
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
    gray = cv2.resize(gray, (48 * n, 48 * n))
    for j in range(0, n * n):
        '''
        if j < 2:
            data[i - 1, :, :, j] = gray[j * 48: (j + 1) * 48, 0: 48]
        else:
            data[i - 1, :, :, j] = gray[(j - 2) * 48: (j - 1) * 48, 48: 2 * 48]
        '''
        if j < 3:
            data[i - 1, :, :, j] = gray[j * 48: (j + 1) * 48, 0: 48]
        elif j < 6:
            data[i - 1, :, :, j] = gray[(j - 3) * 48: (j - 2) * 48, 48: 2 * 48]
        else:
            data[i - 1, :, :, j] = gray[(j - 6) * 48: (j - 5) * 48, 2 * 48: 3 * 48]
        '''
        if j < 4:
            data[i - 1, :, :, j] = gray[j * 48: (j + 1) * 48, 0: 48]
        elif j < 8:
            data[i - 1, :, :, j] = gray[(j - 4) * 48: (j - 3) * 48, 48: 2 * 48]
        elif j < 12:
            data[i - 1, :, :, j] = gray[(j - 8) * 48: (j - 7) * 48, 2 * 48: 3 * 48]
        else:
            data[i - 1, :, :, j] = gray[(j - 12) * 48: (j - 11) * 48, 3 * 48: 4 * 48]
        '''
    print(path)

np.save('Stacked_data', data)
