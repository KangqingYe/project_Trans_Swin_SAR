import argparse
import numpy as np
import pandas as pd
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--datapath', default='/data1/ykq/nnUNet_dataset/nnUNet_raw/nnUNet_raw_data/Task004_Liver1/labelsTr/')
    parser.add_argument('--folds', type=int, default=2)
    parser.add_argument('--outdir', default='/data1/ykq/Unet_f/output')

    args = parser.parse_args()

    return args


def main(args):
    np.random.seed(args.seed)
    subject = os.listdir(args.datapath)
    subject_split = np.array_split(subject, args.folds)
    subject_frame = pd.DataFrame({"subject": subject})
    for i in range(args.folds):
        mask_test = pd.DataFrame({"subject": subject_split[i], "test": pd.Series(np.ones(len(subject_split[i])))})
        mask_frame = pd.merge(subject_frame, mask_test, on='subject', how='left')
        mask_frame['test'] = mask_frame['test'].fillna('0')
        mask_frame.to_csv(args.outdir + '/fold' + str(i) + '.csv', index=False)


if __name__ == '__main__':
    main(get_args())