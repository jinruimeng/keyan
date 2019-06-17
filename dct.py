import readAndWriteDataSet
import getCovMatrix
import cv2

if __name__ == '__main__':
    type = u'oushi'
    # path = u'/Users/jinruimeng/Downloads/keyan/'
    path = u'D:\\Users\\rmjin\\Downloads\\'

    # 读取数据
    channelDataPath = path + "channelDataP.xlsx"
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)
    channelData = getCovMatrix.getAbs(channelDataAll[0])

    Y = cv2.dct(channelData[0, :])
    print(Y.shape)
