import numpy as np
import os
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dataset_path', type=str,
                        default='/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task004_Liver1/', help='root dir for data')
    parser.add_argument('--fixed_dataset_path', type=str,
                        default='/data1/ykq/Unet_f/data/', help='fixed dir for data')
    parser.add_argument('--slice_thickness', default=1, help='target z spacing')
    parser.add_argument('--down_scale', default=0.5, help='down sampling scale')
    parser.add_argument('--size', default=48, help='minmal slices per nii file')
    parser.add_argument('--expand_slice', default=20, help='expand slice')
    args = parser.parse_args()
    return args


class LITS_fix:
    def __init__(self, args):
        self.raw_root_path = args.raw_dataset_path
        self.fixed_path = args.fixed_dataset_path

        if not os.path.exists(self.fixed_path):  # 创建保存目录
            os.makedirs(self.fixed_path + 'imagesTr/')
            os.makedirs(self.fixed_path + 'labelsTr/')

        self.fix_data(args)  # 对原始图像进行修剪并保存

    def fix_data(self, args):
        print('the raw dataset total numbers of samples is :', len(os.listdir(self.raw_root_path + 'labelsTr/')))
        for subject in tqdm(os.listdir(self.raw_root_path + 'labelsTr/')):
            print(subject)
            if subject in os.listdir(self.fixed_path + 'labelsTr/'):
                continue
            # 将CT和金标准入读内存
            ct = sitk.ReadImage(self.raw_root_path + '/imagesTr/' + subject[: -7] + '_0000.nii.gz', sitk.sitkInt16)
            ct_array = sitk.GetArrayFromImage(ct)
            seg = sitk.ReadImage(self.raw_root_path + '/labelsTr/' + subject, sitk.sitkInt8)
            seg_array = sitk.GetArrayFromImage(seg)
            print(ct_array.shape, seg_array.shape)

            # 对CT数据在横断面上进行降采样(下采样),并进行重采样,将所有数据的z轴的spacing调整到1mm
            ct_array = ndimage.zoom(ct_array,(ct.GetSpacing()[-1] / args.slice_thickness, args.down_scale, args.down_scale),order=3)
            seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / args.slice_thickness, args.down_scale, args.down_scale), order=0)
            print(ct_array.shape, seg_array.shape)

            # 找到肝脏区域开始和结束的slice，并各向外扩张
            z = np.any(seg_array, axis=(1, 2))
            start_slice, end_slice = np.where(z)[0][[0, -1]]

            #俩个方向上个扩张个slice
            start_slice = max(0, start_slice - args.expand_slice)
            end_slice = min(seg_array.shape[0], end_slice + args.expand_slice)

            # 如果这时候剩下的slice数量不足size，直接放弃，这样的数据很少
            if end_slice - start_slice < args.size - 1:  # 过滤掉不足以生成一个切片块的原始样本
                continue

            print(str(start_slice) + '--' + str(end_slice))

            ct_array = ct_array[start_slice:end_slice + 1, :, :]  # 截取原始CT影像中包含肝脏区间及拓张的所有切片
            seg_array = seg_array[start_slice:end_slice + 1, :, :]

            #最终将数据保存为nii文件
            new_ct = sitk.GetImageFromArray(ct_array)

            new_ct.SetDirection(ct.GetDirection())
            new_ct.SetOrigin(ct.GetOrigin())
            new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / args.down_scale),
                               ct.GetSpacing()[1] * int(1 / args.down_scale), args.slice_thickness))

            new_seg = sitk.GetImageFromArray(seg_array)

            new_seg.SetDirection(ct.GetDirection())
            new_seg.SetOrigin(ct.GetOrigin())
            new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], args.slice_thickness))

            sitk.WriteImage(new_ct, self.fixed_path + 'imagesTr/' + subject[: -7] + '_0000.nii.gz')
            sitk.WriteImage(new_seg, self.fixed_path + 'labelsTr/' + subject)


def main():
    LITS_fix(get_args())


if __name__ == '__main__':
    main()