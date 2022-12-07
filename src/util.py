import glob
import SimpleITK as sitk
from matplotlib import pyplot as plt

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

