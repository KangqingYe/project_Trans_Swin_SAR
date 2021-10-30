import pandas as pd
import os
import SimpleITK as sitk
from medpy.metric import binary
import numpy as np
from tqdm import tqdm
import argparse
from skimage import exposure
import h5py
"""
evaluate result of project_TransUNet
"""
def calculate_metric_percase(pred, gt,spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = binary.dc(pred, gt)
        hd = binary.hd(pred, gt,spacing)
        assd = binary.assd(pred,gt,spacing)
        precision = binary.precision(pred,gt)
        sensitivity = binary.sensitivity(pred,gt)
        return dice, hd, assd, precision, sensitivity
    elif pred.sum() > 0 and gt.sum()==0:
        return 1,None,None,1,1
    else:
        return 0,-1,-1,0,0

parser = argparse.ArgumentParser()
parser.add_argument('--DIR_PATH',type = str)
parser.add_argument('--num_classes',type = str)
parser.add_argument('--GT_PATH',type = str)
args = parser.parse_args()
print(args.DIR_PATH)
if args.GT_PATH == None:
    files = os.listdir(args.DIR_PATH)
    files = [i for i in files if i[-7 :] == ".nii.gz"]
    num_list_uni = np.unique([int(i[4:8]) for i in files])
    result = pd.DataFrame(None, columns=['subject','label','dc','hd','assd','precision','sensitivity'])
    idx = 0
    for num in tqdm(num_list_uni):
        pred_name = "case%04.0d_pred.nii.gz" % num
        gt_name = "case%04.0d_gt.nii.gz" % num
        pred = sitk.ReadImage(args.DIR_PATH + pred_name)
        pred_array = sitk.GetArrayFromImage(pred)
        gt = sitk.ReadImage(args.DIR_PATH + gt_name)
        gt_array = sitk.GetArrayFromImage(gt)
        for i in range(int(args.num_classes)):
            dice, hd, assd, precision, sensitivity = calculate_metric_percase(pred_array == i, gt_array == i,gt.GetSpacing())
            single_result = pd.DataFrame({"subject":num, "label": i, "dc" : dice, "hd":hd,"assd":assd,"precision":precision,"sensitivity":sensitivity}, index=[idx])
            print(single_result)
            result = pd.concat([result, single_result], axis = 0)
            idx += 1
    result.to_csv(args.DIR_PATH+'result.csv',index = False)
else:
    files = os.listdir(args.DIR_PATH)
    result = pd.DataFrame(None, columns=['subject','label','dc','hd','assd','precision','sensitivity'])#,'sensitivity','precision','assd'
    idx = 0
    for filename in files:
        if filename[-7 :] == ".nii.gz":
            pred = sitk.ReadImage(args.DIR_PATH + filename)
            pred_array = sitk.GetArrayFromImage(pred)
            gt = sitk.ReadImage(args.GT_PATH + filename)
            gt_array = sitk.GetArrayFromImage(gt)
            for i in range(int(args.num_classes)):
                dice, hd, assd, precision, sensitivity = calculate_metric_percase(pred_array == i, gt_array == i,gt.GetSpacing())
                single_result = pd.DataFrame({"subject":filename, "label": i, "dc" : dice, "hd":hd,"assd":assd,"precision":precision,"sensitivity":sensitivity}, index=[idx])
                print(single_result)
                result = pd.concat([result, single_result], axis = 0)
                idx += 1
    result.to_csv(args.DIR_PATH+'result.csv',index = False)