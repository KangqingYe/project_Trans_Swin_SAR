import torch
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.SAR_UNet import Se_PPP_ResUNet
from utils.dataloader_abdomen import get_dataloader, create_slice
from utils.loss_function import get_loss
import pickle
import argparse
import os
import SimpleITK as sitk

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--pkl_data', action='store_true', help='using pkl data')
    parser.add_argument('--fold_data_path', default = '/media/gdp/date/ykq/Unet_f/output/fold0_liver.csv', help='fold csv file path')
    parser.add_argument('--datapath', help='nii.gz data path')
    parser.add_argument('--save_path', default='/media/gdp/date/ykq/Unet_f/output/Swin_UNet/fold0/')
    parser.add_argument('--n_classes', default=8)

    args = parser.parse_args()

    return args

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(args.gpu)
    else:
        device = 'cpu'  

    print('using gpu:' + str(torch.cuda.current_device()))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.pkl_data:
        train_slices = open(args.save_path + "train_slices.pkl", 'rb')
        train_slices = pickle.load(train_slices)
    else:
        fold_data = pd.read_csv(args.fold_data_path)
        train_subject = list(fold_data[fold_data['test'] == 0]['subject'])
        train_slices = create_slice(train_subject, args.datapath)
        pickle.dump(train_slices, open(args.save_path + "train_slices.pkl", 'wb'))

    batch_size = 12
    epochs = 300
    learning_rate = 1e-03
    n_classes = int(args.n_classes)

    model = Se_PPP_ResUNet(1, n_classes, deep_supervision=False).to(device)
    start_epoch = 0    
    
    loss_config = {"name": 'categorical_cross_entropy'}#,"class_weights":[0.5,1,1,1,1,1,1,1,1],"class_weights" = [ 0.05902828, 16.94103315]
    loss_ce = get_loss(loss_config, device)
    loss_list = []

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    print('Begin dataloader.')
    train_dataloader = get_dataloader(batch_size, train_slices, shuffle=True, num_workers=4)

    for epoch in range(start_epoch,epochs):
        train_dataloader = tqdm(train_dataloader)
        model.train()
        loss_value = [0, 0]
        for img_batch, mask_batch in train_dataloader:
            assert not np.any(np.isnan(img_batch.numpy())), "image nan"
            assert not np.any(np.isnan(mask_batch.numpy())),"mask nan"
            img_batch = img_batch.float().to(device)#[batch_size,1,512,512]
            mask_batch = mask_batch.long().to(device).squeeze(1)#[batch_size,512,512]
            predict_batch = model(img_batch)#[batch_size,7,512,512]
            loss = loss_ce(predict_batch, mask_batch)# + loss_dc(predict_batch, mask_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value[0] += img_batch.size()[0]
            loss_value[1] += loss.detach().cpu().numpy() * img_batch.size()[0]
            train_dataloader.set_description("Epoch %d :Train loss %f" % (epoch, loss_value[1] / loss_value[0]))
        loss_list.append(loss_value[1] / loss_value[0])
        plt.plot(loss_list)
        a = pd.DataFrame(loss_list)
        a.to_csv(args.save_path + 'loss.csv',index=False)
        plt.savefig(args.save_path + 'progress.png')
        scheduler.step()
        torch.save(model, args.save_path + 'model_' + str(epoch+1) + '.pth')

if __name__ == '__main__':
    main(get_args())