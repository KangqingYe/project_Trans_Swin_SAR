from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from torchvision import transforms as T
from torchvision.transforms import functional as F
import random
from PIL import Image
from sklearn import preprocessing
from skimage import exposure
from tqdm import tqdm
import numpy as np
from scipy import ndimage
import torch
from scipy.ndimage.interpolation import zoom


def WL(data, WC, WW):
    # WC: 窗位     WW：窗宽
    min = (2 * WC - WW) / 2.0
    max = (2 * WC + WW) / 2.0
    # print(max, min)
    idx_max = np.where(data > max)
    idx_min = np.where(data < min)
    idx_in = np.where((data >= min) & (data <= max))

    data = (data - min) * 254 / (max - min)
    data[idx_max] = 255
    data[idx_min] = 0
    return data

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def create_slice(subjects, datapath):
    all_slices = []
    for subject in tqdm(subjects):
        img = sitk.ReadImage(datapath + '/imagesTr/' + subject[: -7] + '_0000.nii.gz')
        img_array = sitk.GetArrayFromImage(img)
        gt = sitk.ReadImage(datapath + '/labelsTr/' + subject)
        gt_array = sitk.GetArrayFromImage(gt)
        img_array = img_array/255
        # img_array = (img_array - np.mean(img_array))/np.std(img_array)
        for i in range(img_array.shape[0]):
            # img_silce = img_array[i, :, :]/255
            # img_silce=exposure.equalize_hist(img_silce) #进行直方图均衡化
            all_slices.append((img_array[i, :, :], gt_array[i, :, :]))
    return all_slices


class TestDataset(Dataset):
    def __init__(self, all_slices):
        self.all_slices = all_slices

    def __len__(self):
        return len(self.all_slices)

    def __getitem__(self, item):
        single_slice = self.all_slices[item]
        image = single_slice[0]
        GT = single_slice[1]
        if random.random() > 0.5:
            image, GT = random_rot_flip(image, GT)
        elif random.random() > 0.5:
            image, GT = random_rotate(image, GT)
        image = zoom(image, (224/512, 224/512), order=3)  # why not 3?
        GT = zoom(GT, (224/512, 224/512), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        GT = torch.from_numpy(GT.astype(np.float32)).unsqueeze(0)

        # aspect_ratio = image.shape[1]/image.shape[0]
        # image = Image.fromarray(image)
        # GT = Image.fromarray(GT)

        # Transform = []
        # if random.random() <0.5:
        #     ResizeRange = random.randint(410,614)
        #     Transform.append(T.Resize((int(ResizeRange*aspect_ratio),ResizeRange),interpolation=Image.NEAREST))
        # if random.random() < 0.3:
        #     RotationRange = random.randint(-30,30)
        #     Transform.append(T.RandomRotation((RotationRange,RotationRange)))

        # Transform = T.Compose(Transform)
        
        # image = Transform(image)
        # GT = Transform(GT)

        # if random.random() < 0.3:
        #     image = F.hflip(image)
        #     GT = F.hflip(GT)

        # if random.random() < 0.3:
        #     image = F.vflip(image)
        #     GT = F.vflip(GT)

        # image = Transform(image)
        
        # Transform =[]

        # Transform.append(T.Resize((int(512*aspect_ratio)-int(512*aspect_ratio)%16,512),interpolation=Image.NEAREST))
        # Transform.append(T.ToTensor())
        # Transform = T.Compose(Transform)

        # image = Transform(image)
        # GT = Transform(GT)

        return image, GT


def get_dataloader(batch_size, all_slice, shuffle=True, num_workers=4):
    dataset = TestDataset(all_slice)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=False)
    return dataloader