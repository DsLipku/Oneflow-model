import glob
import SimpleITK as sitk
from matplotlib import pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
path = '/data2/lijiayang/MRI/original'

def showNii(img):
    for i in range(img.shape[0]):
        plt.imshow(img[i,:,:],cmap='gray')
        plt.show()

def ReadNii(path):
    path = '/'.join([path,'*.nii'])
    file = glob.glob(path)
    print(len(file))
    file=file[:3]
    for f in file:
        img_nii = sitk.ReadImage(f)
        img = sitk.GetArrayFromImage(img_nii)
        showNii(img)


def getlrbyfft(imgdata, factor):
    '''

    :param imgdata: numpy 数组，例如一张MRI图像形状为（256，256，256）。
    :param factor: 每个方向上的缩放因子，例如(2,2,2)，其意味着在每个方向上去掉一半的高频信息。
    :return:
    '''
    x, y, z = imgdata.shape

    x_centure = x // 2
    y_centure = y // 2
    z_centure = z // 2
    x_off = x // (2 * factor[0])
    y_off = y // (2 * factor[1])
    z_off = z // (2 * factor[2])
    # 傅里叶变换->把低频shift到中心
    _hrfft_shift = np.fft.fftshift(np.fft.fftn(imgdata))
    # 去掉一部分高频分量->ishift
    _lrfft_ishit = np.fft.ifftshift(
        _hrfft_shift[x_centure - x_off:x_centure + x_off, y_centure - y_off:y_centure + y_off,
        z_centure - z_off:z_centure + z_off])
    # 傅里叶逆变换到实数域，取绝对值
    _lr_ifft = np.abs(np.fft.ifftn(_lrfft_ishit))
    # 修改数据类型用于深度学习模型运算
    _lr_ifft = _lr_ifft.astype(np.float32)
    return _lr_ifft


def getLR(imgdata, factor):
    x, y, z = imgdata.shape
    x_centure = x // 2
    y_centure = y // 2
    z_centure = z // 2
    x_off = x // (2 * factor[0])
    y_off = y // (2 * factor[1])
    z_off = z // (2 * factor[2])
    imgfft = np.fft.fftn(imgdata)
    imgfft[x_centure - x_off:x_centure + x_off, y_centure - y_off:y_centure + y_off,
        z_centure - z_off:z_centure + z_off] = 0
    imgifft = np.fft.ifftn(imgfft)
    img_out = abs(imgifft)

    return img_out



#img = cv2.imread('../00017_gray_out.png')
#
#print(img.shape)
#lr = getlrbyfft(img, (1, 1, 1))
#print(lr.shape)
#plt.imshow(lr)
#plt.show()
#
