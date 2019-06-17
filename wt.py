import readAndWriteDataSet
import getCovMatrix
import pywt

if __name__ == '__main__':
    type = u'oushi'
    # path = u'/Users/jinruimeng/Downloads/keyan/'
    path = u'D:\\Users\\rmjin\\Downloads\\'

    # 读取数据
    channelDataPath = path + "channelDataP.xlsx"
    channelDataAll = readAndWriteDataSet.excelToMatrixList(channelDataPath)
    channelData = getCovMatrix.getAbs(channelDataAll[0])

    coeffs = pywt.dwt(channelData[0, :], 'haar')
    cA, cD = coeffs

    while len(cA) > 5:
        cA = pywt.dwt(cA, 'haar')[0]

    print(len(cA))
    print(cA)
