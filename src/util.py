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

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])
    


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        path = '/'.join([path,'*.nii'])
        file = glob.glob(path)
        file = file[:5]
        #for f in file:
        #    img_nii = sitk.ReadImage(f)
        #    img = sitk.GetArrayFromImage(img_nii)
        self.nii_filenames = file
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)

