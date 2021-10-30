import os
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from medpy.metric import binary
from sklearn import preprocessing
from skimage import exposure
from utils.dataloader import WL
from tqdm import tqdm
import argparse
import torch.nn.functional as nnf

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = binary.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum()==0:
        return 1
    else:
        return 0

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data))/_range

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--fold_data_path', default = '/media/gdp/date/ykq/Unet_f/output/fold1_liver.csv', help='fold csv file path')
    parser.add_argument('--datapath', help='nii.gz data path')
    parser.add_argument('--save_path', default='/media/gdp/date/ykq/Unet_f/output/Swin_UNet/fold0/')
    parser.add_argument('--data_name', help='data name: liver, heart_ct, heart_mr, abdomen')
    parser.add_argument('--model', help='model_path')
    parser.add_argument('--n_classes', type=int, help='model_path')

    args = parser.parse_args()

    return args

def main(args):
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(args.gpu)
        # device = 'cpu'
    else:
        device = 'cpu'
    print('using gpu:' + str(torch.cuda.current_device()))

    data = pd.read_csv(args.fold_data_path)
    test_subject = list(data[data['test'] == 1]['subject'])
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    model = torch.load(args.model, map_location={'cuda:0':'cuda:3'}).to(device)# , map_location={'cuda:0':'cuda:1'}
    model.eval()
    result = pd.DataFrame(None, columns=['subject','label','dc','assd'])
    idx = 0

    for subject in tqdm(test_subject):
        print(subject)
        img = sitk.ReadImage(args.datapath + '/imagesTr/' + subject[: -7] + '_0000.nii.gz')
        img_array = sitk.GetArrayFromImage(img)
        gt = sitk.ReadImage(args.datapath + '/labelsTr/' + subject)
        gt_array = sitk.GetArrayFromImage(gt)
        pred_array = torch.zeros((img_array.shape[0], img_array.shape[1], img_array.shape[2]))
        if args.data_name == 'liver':
            for i in range(img_array.shape[0]):
                img_silce = WL(img_array[i, :, :],0, 1000)
                img_silce=exposure.equalize_hist(img_silce) #进行直方图均衡化
                img_silce = preprocessing.MinMaxScaler().fit_transform(img_silce)
                pred_slice = model(torch.tensor(img_silce).float().to(device).unsqueeze(0).unsqueeze(0))#[1,8,512,512]
                pred_array[i, :, :] = torch.argmax(pred_slice,dim=1)[0]
        elif args.data_name == 'HeartCT':
            for i in range(img_array.shape[0]):
                img_silce = WL(img_array[i, :, :],0, 2000)
                img_silce=exposure.equalize_hist(img_silce) #进行直方图均衡化
                img_silce = preprocessing.MinMaxScaler().fit_transform(img_silce)
                pred_slice = model(torch.tensor(img_silce).float().to(device).unsqueeze(0).unsqueeze(0))#[1,8,512,512]
                pred_array[i, :, :] = torch.argmax(pred_slice,dim=1)[0]
        elif args.data_name == 'HeartMR':
            percentile_99_5 = np.percentile(img_array, 99.5)
            img_array = np.clip(img_array,0,percentile_99_5)
            img_array = normalization(img_array)
            # if img.GetDirection() == (-0.0, 0.0, -1.0, 1.0, -0.0, 0.0, 0.0, 1.0, -0.0):
            for i in range(img_array.shape[0]):
                img_silce = torch.tensor(img_array[i, :, :]).float().to(device).unsqueeze(0).unsqueeze(0)
                img_silce = nnf.interpolate(img_silce,[224,224])
                pred_silce = model(img_silce)#[1,8,512,512]
                pred_silce = nnf.interpolate(torch.argmax(pred_silce,dim=1).unsqueeze(0).float(),[img_array.shape[1],img_array.shape[2]])
                pred_array[i, :, :] = pred_silce[0].long()
            # if img.GetDirection() == (1.0, 0.0, 0.0, 0.0, -0.0, 1.0, 0.0, 1.0, -0.0):
            #     for i in range(img_array.shape[2]):
            #         pad_wid = int((img_array[:,:,i].shape[1]-img_array[:,:,i].shape[0])/2)
            #         img_slice = np.pad(img_array[:,:,i],((pad_wid,pad_wid),(0,0)),'constant', constant_values=(0))
            #         img_slice = np.swapaxes(img_slice,0,1)
            #         img_silce = torch.tensor(img_slice).float().to(device).unsqueeze(0).unsqueeze(0)
            #         img_silce = nnf.interpolate(img_silce,[224,224])
            #         img1 = sitk.GetImageFromArray(img_silce.squeeze(0).detach().cpu().numpy())
            #         sitk.WriteImage(img1, args.save_path+'test_img.nii.gz')
            #         pred_silce = model(img_silce)
            #         img1 = sitk.GetImageFromArray(torch.argmax(pred_silce,dim=1).float().detach().cpu().numpy())
            #         sitk.WriteImage(img1, args.save_path+'test_pred.nii.gz')
            #         pred_silce = nnf.interpolate(torch.argmax(pred_silce,dim=1).unsqueeze(0).float(),[img_array.shape[1],img_array.shape[1]])
            #         pred_silce = torch.transpose(pred_silce[0], 1, 2).squeeze(0)
            #         print(pred_silce.unique())
            #         pred_silce = pred_silce[pad_wid:pad_wid+img_array[:,:,i].shape[0],:]
            #         pred_array[:, :, i] = pred_silce.long()

        elif args.data_name == 'Abdomen':
            for i in range(img_array.shape[0]):
                img_silce = torch.tensor(img_array[i, :, :]/255).float().to(device).unsqueeze(0).unsqueeze(0)
                img_silce = nnf.interpolate(img_silce,[224,224])
                pred_silce = model(img_silce)#[1,8,512,512]
                pred_silce = nnf.interpolate(torch.argmax(pred_silce,dim=1).unsqueeze(0).float(),[512,512])
                pred_array[i, :, :] = pred_silce[0].long()
        pred_array = pred_array.detach().cpu().numpy()
        # print(np.unique(pred_array))
        for i in range(args.n_classes):
            dc = calculate_metric_percase(pred_array == i, gt_array == i)
            # assd = binary.assd(pred_array == i, gt_array == i,gt.GetSpacing())
            assd=0
            single_result = pd.DataFrame({"subject":subject, "label": i, "dc" : dc, "assd": assd}, index=[idx])
            print(single_result)
            result = pd.concat([result, single_result], axis = 0)
            idx += 1
        result.to_csv(args.save_path + 'result_300.csv',index = False)
        savedImg = sitk.GetImageFromArray(pred_array)
        savedImg.SetOrigin(gt.GetOrigin())
        savedImg.SetDirection(gt.GetDirection())
        savedImg.SetSpacing(gt.GetSpacing())
        sitk.WriteImage(savedImg, args.save_path + subject)

if __name__ == '__main__':
    main(get_args())