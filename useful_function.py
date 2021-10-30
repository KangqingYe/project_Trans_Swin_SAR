import numpy as np
import h5py
from tqdm import tqdm
import pickle
'''
Preprocess code
'''

'''
convert heart label from [0, 205, 420, 500, 550, 600, 820, 850] to [0, 1, 2, 3, 4, 5, 6, 7]
'''
import os
import SimpleITK as sitk
DIR_PATH = '/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task012_Heart/labelsTr1/'
files = os.listdir(DIR_PATH)
for filename in files:
    print(filename)
    img = sitk.ReadImage(DIR_PATH + filename)
    heart_label = sitk.GetArrayFromImage(img)

    origin =img.GetOrigin()
    direction = img.GetDirection()
    space = img.GetSpacing()

    heart_label[heart_label == 205] = 1
    heart_label[heart_label == 420] = 2
    heart_label[heart_label == 500] = 3
    heart_label[heart_label == 550] = 4
    heart_label[heart_label == 600] = 5
    heart_label[heart_label == 820] = 6
    heart_label[heart_label == 850] = 7
    savedImg = sitk.GetImageFromArray(heart_label)
    savedImg.SetOrigin(origin)
    savedImg.SetDirection(direction)
    savedImg.SetSpacing(space)
    sitk.WriteImage(savedImg, '/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task012_Heart/labelsTr/' + filename)

'''
rename 'ct_train_10xx_image.nii' to 'heart_x_0000.nii.gz'
'''
import os
DIR_PATH = '/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task012_Heart/'
files = os.listdir(DIR_PATH + 'CT_images/')
for filename in files:
    print(filename)
    name, suffix = os.path.splitext(filename)
    num = int(name[11:13])
    new_name = DIR_PATH +'imagesTr/heart_' + str(num) + '_0000.nii.gz'
    old_name = os.path.join(DIR_PATH + 'CT_images/', filename)

    img = sitk.ReadImage(old_name)
    heart = sitk.GetArrayFromImage(img)

    origin =img.GetOrigin()
    direction = img.GetDirection()
    space = img.GetSpacing()

    savedImg = sitk.GetImageFromArray(heart)
    savedImg.SetOrigin(origin)
    savedImg.SetDirection(direction)
    savedImg.SetSpacing(space)
    sitk.WriteImage(savedImg, new_name)

'''
generate dataset.json for oasis
'''
import os
import json

training = []
DIR_PATH = '/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task011_oasis/labelsTr/'
files = os.listdir(DIR_PATH)
for filename in files:
    name, suffix = os.path.splitext(filename)
    d = {"image":'./imagesTr/' + name + ".gz", "label": './labelsTr/' + filename}
    training.append(d)

filename='/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task011_oasis/dataset1.json'
with open(filename,'w') as file_obj:
    json.dump(training,file_obj)

'''
rename 'mr_train_10xx_image.nii' to 'heartMR_x_0000.nii.gz'
'''
import SimpleITK as sitk
import os
DIR_PATH = '/data1/lsm/dataset/Heart_Raw_data/MR_image/'
OUTPUT_PATH = '/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task002_HeartMR/'
files = os.listdir(DIR_PATH)
for filename in files:
    print(filename)
    # name, suffix = os.path.splitext(filename)
    num = int(filename[11:13])
    new_name = OUTPUT_PATH +'imagesTr/heartMR_' + str(num) + '_0000.nii.gz'
    old_name = os.path.join(DIR_PATH, filename)

    img = sitk.ReadImage(old_name)
    heart = sitk.GetArrayFromImage(img)

    origin =img.GetOrigin()
    direction = img.GetDirection()
    space = img.GetSpacing()

    savedImg = sitk.GetImageFromArray(heart)
    savedImg.SetOrigin(origin)
    savedImg.SetDirection(direction)
    savedImg.SetSpacing(space)
    sitk.WriteImage(savedImg, new_name)
'''
convert heart label from [0, 205, 420, 500, 550, 600, 820, 850] to [0, 1, 2, 3, 4, 5, 6, 7]
'''
import os
import SimpleITK as sitk
DIR_PATH = '/data1/lsm/dataset/Heart_Raw_data/MR_label/'
OUTPUT_PATH = '/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task002_HeartMR/'
files = os.listdir(DIR_PATH)
for filename in files:
    print(filename)
    num = int(filename[11:13])
    new_name = OUTPUT_PATH +'labelsTr/heartMR_' + str(num) + '.nii.gz'

    img = sitk.ReadImage(DIR_PATH + filename)
    heart_label = sitk.GetArrayFromImage(img)

    origin =img.GetOrigin()
    direction = img.GetDirection()
    space = img.GetSpacing()
    # print(set(heart_label.flatten()))

    heart_label[heart_label == 205] = 1
    heart_label[heart_label == 420] = 2
    heart_label[heart_label == 421] = 2
    heart_label[heart_label == 500] = 3
    heart_label[heart_label == 550] = 4
    heart_label[heart_label == 600] = 5
    heart_label[heart_label == 820] = 6
    heart_label[heart_label == 850] = 7
    savedImg = sitk.GetImageFromArray(heart_label)
    savedImg.SetOrigin(origin)
    savedImg.SetDirection(direction)
    savedImg.SetSpacing(space)
    sitk.WriteImage(savedImg, new_name)

'''
generate dataset.json for HeartMR
'''
import os
import json

training = []
DIR_PATH = '/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task002_HeartMR/labelsTr/'
files = os.listdir(DIR_PATH)
for filename in files:
    name, suffix = os.path.splitext(filename)
    d = {"image":'./imagesTr/' + name + ".gz", "label": './labelsTr/' + filename}
    training.append(d)

filename='/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task002_HeartMR/dataset.json'
with open(filename,'w') as file_obj:
    json.dump(training,file_obj)

"""
evaluation result of nnUNet
"""

import pandas as pd
import os
import SimpleITK as sitk
from medpy.metric import binary
DIR_PATH = '/data1/ykq/nnUNet_dataset/nnUNet_trained_models/nnUNet/3d_fullres/Task002_HeartMR/nnUNetTrainerV2__nnUNetPlansv2.1/output_model/'
GT_PATH = '/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task002_HeartMR/labelsTr/'
files = os.listdir(DIR_PATH)
result = pd.DataFrame(None, columns=['subject','label','dc','sensitivity','precision','assd'])
idx = 0
for filename in files:
    if filename[-7 :] == ".nii.gz":
        print(filename)
        pred = sitk.ReadImage(DIR_PATH + filename)
        pred_array = sitk.GetArrayFromImage(pred)
        gt = sitk.ReadImage(GT_PATH + filename)
        gt_array = sitk.GetArrayFromImage(gt)
        for i in range(1,8):
            dc = binary.dc(pred_array == i, gt_array == i)
            sensitivity = binary.sensitivity(pred_array == i, gt_array == i)
            precision = binary.precision(pred_array == i, gt_array == i)
            assd = binary.assd(pred_array == i, gt_array == i,gt.GetSpacing())
            single_result = pd.DataFrame({"subject":filename, "label": i, "dc" : dc, "sensitivity": sensitivity, "precision": precision, "assd": assd}, index=[idx])
            print(single_result)
            result = pd.concat([result, single_result], axis = 0)
            idx += 1
result.to_csv(DIR_PATH+'result.csv',index = False)

"""
calculate w
"""
GT_PATH = '/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task012_Heart/labelsTr/'
files = os.listdir(GT_PATH)
label_num = np.zeros(8)
for filename in files:
    if filename[-7 :] == ".nii.gz":
        print(filename)
        gt = sitk.ReadImage(GT_PATH + filename)
        gt_array = sitk.GetArrayFromImage(gt)
        for i in range(0,8):
            num = np.sum(gt_array == i)
            label_num[i] += num 
w = np.zeros(8)
N = label_num.sum()
for i in range(8):
    w[i] = (N - label_num[i])/label_num[i]
print(label_num)
print(w)

"""
change label 2 in liver to 1
"""
import os
import SimpleITK as sitk
from medpy.metric import binary
GT_PATH = '/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task003_Liver/labelsTr/'
TAR_PATH = '/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task004_Liver1/labelsTr/'
files = os.listdir(GT_PATH)
for filename in files:
    if filename[-7 :] == ".nii.gz":
        print(filename)
        gt = sitk.ReadImage(GT_PATH + filename)
        gt_array = sitk.GetArrayFromImage(gt)
        origin =gt.GetOrigin()
        direction = gt.GetDirection()
        space = gt.GetSpacing()

        gt_array[gt_array == 2] = 1
        savedImg = sitk.GetImageFromArray(gt_array)
        savedImg.SetOrigin(origin)
        savedImg.SetDirection(direction)
        savedImg.SetSpacing(space)
        sitk.WriteImage(savedImg, TAR_PATH + filename)

"""
load npz files
"""

pkl_file = open('D:/IMR/dataset/Task03_Liver/liver_0.npz', 'rb')
a = pickle.load(pkl_file)
a = np.load('D:/IMR/dataset/Task03_Liver/liver_0.npz')
a = np.load('heartMR_12.npz')
b = a['data']
print(b[0].min())
print(b[0].max())
print(set(b[1].flatten()))
save(b[0],'heartMR_12_0.nii.gz')
save(b[1],'heartMR_12_1.nii.gz')
print(b[1].shape)

"""
get info of images
"""

datapath = 'D:\IMR\dataset\Task03_Liver\imagesTr2\imagesTr\\'
files = os.listdir(datapath)
spacing_list0 = []
spacing_list1 = []
spacing_list2 = []
shape_list0 = []
for filename in files:
    print(filename)
    img = sitk.ReadImage(datapath + filename)
    img_array = sitk.GetArrayFromImage(img) # (?,512,512)
    img_spacing = img.GetSpacing()
    spacing_list0.append(img_spacing[0])
    spacing_list1.append(img_spacing[1])
    spacing_list2.append(img_spacing[2])
    shape_list0.append(img_array.shape[0])
    # print(img.GetSpacing())
    # print(img.GetOrigin())
    # print(img.GetDirection())
    print(img_array.shape)
    print(img_spacing)
    # print(img_array.max(), img_array.min())
print(np.mean(shape_list0), np.median(shape_list0), max(shape_list0), min(shape_list0))
print(np.mean(spacing_list0), np.median(spacing_list0), min(spacing_list0), max(spacing_list0))
print(np.mean(spacing_list1), np.median(spacing_list1), min(spacing_list1), max(spacing_list1))
print(np.mean(spacing_list2), np.median(spacing_list2), min(spacing_list2), max(spacing_list2))
'''
change the spacing
'''
DIR_PATH = '/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task004_liver1/labelsTr1/'
TAR_PATH = '/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task004_liver1/labelsTr/'
files = os.listdir(DIR_PATH)

for filename in tqdm(files):
    if filename[-7 :] == ".nii.gz":
        print(filename)
        img = sitk.ReadImage(DIR_PATH + filename)
        img_array = sitk.GetArrayFromImage(img)
        new_seg = sitk.GetImageFromArray(img_array)

        new_seg.SetDirection(img.GetDirection())
        new_seg.SetOrigin(img.GetOrigin())
        new_seg.SetSpacing((img.GetSpacing()[0] * int(1 / 0.5),
                            img.GetSpacing()[1] * int(1 / 0.5), 1))
        sitk.WriteImage(new_seg, TAR_PATH + filename)
'''
change h5 file in Synapse data to nii.gz
'''
DIR_PATH = '/data1/ykq/project_TransUNet/data/Synapse/test_vol_h5/'
TAR_PATH = '/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task005_Abdomen/'
files = os.listdir(DIR_PATH)

for filename in tqdm(files):
    print(filename)
    num = int(filename[4:8])
    # data = np.load(DIR_PATH + filename)
    # image = data['image']
    # GT = data['label']#(512,512)
    data = h5py.File(DIR_PATH + filename,'r')
    image = np.array(data['image'])
    image = np.swapaxes(image,1,2)
    GT = np.array(data['label'])#(100,512,512)
    GT = np.swapaxes(GT,1,2)
    new_ct = sitk.GetImageFromArray(255*image)
    new_ct.SetSpacing((1, 1, 1))
    sitk.WriteImage(new_ct, TAR_PATH + 'imagesTr/abdomen_' + str(num) + '_0000.nii.gz')
    new_seg = sitk.GetImageFromArray(GT)
    new_seg.SetSpacing((1, 1, 1))
    sitk.WriteImage(new_seg, TAR_PATH+ 'labelsTr/abdomen_' + str(num) + '.nii.gz')
'''
gen_fold from nnUNet pickle
'''
pkl_file = open('/data1/ykq/nnUNet_dataset/nnUNet_preprocessed/Task004_liver1/splits_final.pkl', 'rb')
a = pickle.load(pkl_file)
mask_train = pd.DataFrame({"subject": a[0]['train'], "test": pd.Series(np.zeros(len(a[0]['train'])))})
mask_test = pd.DataFrame({"subject": a[0]['val'], "test": pd.Series(np.ones(len(a[0]['val'])))})
fold_0 = pd.concat([mask_test,mask_train],ignore_index=True)
fold_0.to_csv('fold0_liver.csv',index=False)
mask_train = pd.DataFrame({"subject": a[1]['train'], "test": pd.Series(np.zeros(len(a[1]['train'])))})
mask_test = pd.DataFrame({"subject": a[1]['val'], "test": pd.Series(np.ones(len(a[1]['val'])))})
fold_1 = pd.concat([mask_test,mask_train],ignore_index=True)
fold_1.to_csv('fold1_liver.csv',index=False)
'''
change npz file in Synapse data to nii.gz
'''
DIR_PATH = '/data1/ykq/project_TransUNet/data/Synapse/train_npz/'
TAR_PATH = '/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task005_Abdomen/'
files = os.listdir(DIR_PATH)
num_list = []
for filename in files:
    num = int(filename[4:8])
    num_list.append(num)
num_list_uni = np.unique(num_list)
print(num_list_uni)
for num in tqdm(num_list_uni):
    num_slices = np.sum(num_list == num)
    img_array = np.zeros((num_slices, 512, 512))
    GT_array = np.zeros((num_slices, 512, 512))
    for i in range(num_slices):
        name = "case%04.0d_slice%03.0d.npz" % (num,i)
        data = np.load(DIR_PATH + name)
        image = data['image']
        GT = data['label']#(512,512)
        image = np.swapaxes(image,0,1)
        GT = np.swapaxes(GT,0,1)
        img_array[i,:,:] = image
        GT_array[i,:,:] = GT
    new_ct = sitk.GetImageFromArray(255*img_array)
    new_ct.SetSpacing((1, 1, 1))
    sitk.WriteImage(new_ct, TAR_PATH + 'imagesTr/testabdomen_' + str(num) + '_0000.nii.gz')
    new_seg = sitk.GetImageFromArray(GT_array)
    new_seg.SetSpacing((1, 1, 1))
    sitk.WriteImage(new_seg, TAR_PATH+ 'labelsTr/testabdomen_' + str(num) + '.nii.gz')

"""
change nii.gz file to npy.h5 and npz
"""
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data))/_range

data_path = '/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task004_liver1/'
fold_data_path = '/data1/ykq/fold/fold0_liver.csv'
save_path = '/data1/ykq/project_TransUNet/data/Liver/fold0/'
fold_data = pd.read_csv(fold_data_path)
train_subject = list(fold_data[fold_data['test'] == 0]['subject'])
test_subject = list(fold_data[fold_data['test'] == 1]['subject'])
# generate train data
for subject in tqdm(train_subject):
    # GT
    GT = sitk.ReadImage(data_path + 'labelsTr/' + subject)
    GT_array = sitk.GetArrayFromImage(GT)
    # img
    img = sitk.ReadImage(data_path + '/imagesTr/' + subject[: -7] + '_0000.nii.gz')
    img_array = sitk.GetArrayFromImage(img)
    img_array = np.clip(img_array, -500, 500)
    img_array = normalization(img_array)
    # npz
    num = "".join(list(filter(str.isdigit, subject)))
    for i in range(img_array.shape[0]):
        new_name = "case%04.0d_slice%03.0d.npz" % (int(num),i)
        np.savez(save_path + "train_npz/" + new_name, image = img_array[i,:,:], label = GT_array[i,:,:])
# # generate test data
for subject in tqdm(test_subject):
    # GT
    GT = sitk.ReadImage(data_path + 'labelsTr/' + subject)
    GT_array = sitk.GetArrayFromImage(GT)
    GT_spacing = GT.GetSpacing()
    # img
    img = sitk.ReadImage(data_path + '/imagesTr/' + subject[: -7] + '_0000.nii.gz')
    img_array = sitk.GetArrayFromImage(img)
    img_array = np.clip(img_array, -500, 500)
    img_array = normalization(img_array)
    # h5
    num = "".join(list(filter(str.isdigit, subject)))
    new_name = "case%04.0d.npy.h5" % int(num)
    with h5py.File(save_path + 'test_vol_h5/' + new_name, 'a') as f:
        f.create_dataset('image',data = img_array)
        f.create_dataset('label',data = GT_array)
        f.create_dataset('spacing',data = np.array(GT_spacing))

"""
generate list txt
"""
DIR_PATH = '/data1/ykq/project_TransUNet/data/Liver/fold0/test_vol_h5/'
files = os.listdir(DIR_PATH)
files_name = [i[:-7] for i in files]
str = '\n'
f=open("/data1/ykq/project_TransUNet/TransUNet/lists/lists_liver/fold0/test_vol.txt","w")
f.write(str.join(files_name))
f.close()
f=open("/data1/ykq/project_TransUNet/TransUNet/lists/lists_liver/fold0/all.lst","w")
f.write(str.join(files))
f.close()
# generate train.txt
DIR_PATH = '/data1/ykq/project_TransUNet/data/Liver/fold0/train_npz/'
files = os.listdir(DIR_PATH)
files_name = [i[:-4] for i in files]
str = '\n'
f=open("/data1/ykq/project_TransUNet/TransUNet/lists/lists_liver/fold0/train.txt","w")
f.write(str.join(files_name))
f.close()
"""
evaluate result of project_TransUNet
"""
parser = argparse.ArgumentParser()
parser.add_argument('--DIR_PATH',type = str)
parser.add_argument('--num_classes',type = str)
args = parser.parse_args()

files = os.listdir(args.DIR_PATH)
num_list_uni = np.unique([int(i[4:8]) for i in files])
result = pd.DataFrame(None, columns=['subject','label','dc','assd'])
idx = 0
for num in tqdm(num_list_uni):
    pred_name = "case%04.0d_pred.nii.gz" % num
    gt_name = "case%04.0d_gt.nii.gz" % num
    pred = sitk.ReadImage(args.DIR_PATH + pred_name)
    pred_array = sitk.GetArrayFromImage(pred)
    gt = sitk.ReadImage(args.DIR_PATH + gt_name)
    gt_array = sitk.GetArrayFromImage(gt)
    for i in range(int(args.num_classes)):
        dc = binary.dc(pred_array == i, gt_array == i)
        # assd = binary.assd(pred_array == i, gt_array == i,gt.GetSpacing())
        assd = 0
        single_result = pd.DataFrame({"subject":num, "label": i, "dc" : dc, "assd": assd}, index=[idx])
        print(single_result)
        result = pd.concat([result, single_result], axis = 0)
        idx += 1
result.to_csv(args.DIR_PATH+'result.csv',index = False)

"""
calculate evaluation result
"""
parser = argparse.ArgumentParser()
parser.add_argument('--DIR_PATH',type = str)
args = parser.parse_args()
DIR_PATH_0 = args.DIR_PATH
DIR_PATH_1 = DIR_PATH_0.replace('fold0','fold1',1)
result0 = pd.read_csv(DIR_PATH_0+'result.csv')
result1 = pd.read_csv(DIR_PATH_1+'result.csv')
result_all = pd.concat([result1,result1],axis=0)
result_cal = result_all[result_all['label']!=0]
print(np.mean(result_cal['dc']))
"""
prepare data for nnUNet predict
"""
parser = argparse.ArgumentParser()
parser.add_argument('--datapath',type = str)
parser.add_argument('--fold_data_path',type = str)
args = parser.parse_args()
if not os.path.exists(args.datapath + '/fold0/imagesTs/'):
        os.makedirs(args.datapath + '/fold0/imagesTs/')
if not os.path.exists(args.datapath + '/fold1/imagesTs/'):
        os.makedirs(args.datapath + '/fold1/imagesTs/')
# fold0
fold_data = pd.read_csv(args.fold_data_path)
train_subject = list(fold_data[fold_data['test'] == 1]['subject'])
for subject in tqdm(train_subject):
    img = sitk.ReadImage(args.datapath + '/imagesTr/' + subject[: -7] + '_0000.nii.gz')
    sitk.WriteImage(img, args.datapath + '/fold0/imagesTs/' + subject[: -7] + '_0000.nii.gz')
#fold1
fold_data = pd.read_csv(args.fold_data_path.replace('fold0','fold1',1))
train_subject = list(fold_data[fold_data['test'] == 1]['subject'])
for subject in tqdm(train_subject):
    img = sitk.ReadImage(args.datapath + '/imagesTr/' + subject[: -7] + '_0000.nii.gz')
    sitk.WriteImage(img, args.datapath + '/fold1/imagesTs/' + subject[: -7] + '_0000.nii.gz')