import torch
import torch.nn as nn
from glob import glob
import os
import numpy as np
import argparse
import cv2
import albumentations as A
import torch.nn.functional as F
import math
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import matplotlib.pyplot as plt


from utils import BCEDiceLoss
import utils as ut
from dataset import Dataset, KvasirSEG_Dataset, CVCClinicDB_Dataset, ISICDataset, ISIC18, NPY_datasets
from collections import OrderedDict

# Networks
from networks.mixer_unet import Mixer_UNet
# from networks.kan_unet import KANU_Net
from networks.unet import UNet
from networks.vnet import Vnet
from networks.r2_unet import R2Unet
from networks.r2_unet_mixer import R2UnetMixer
from networks.unetpp import ResUnetPlusPlus
from networks.att_unet import AttentionUNet
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from networks.MAXFormer import MAXFormer
from networks.unext import UNext

from networks.mixer_unet_without_mixer_block import Mixer_UNet_Without_MB
from networks.mixer_unet_without_eca import Mixer_UNet_Without_ECA
from networks.mixer_unet_with_eca_without_MB import Mixer_UNet_With_ECA_Without_MB

from loggings import get_logger
from torchvision import transforms
from datetime import datetime


from skimage.measure import find_contours
from dataset_utils import myNormalize, myToTensor, myRandomHorizontalFlip, myRandomVerticalFlip, myRandomRotation, myResize

import transf

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model train')
    parser.add_argument('--model', type=str, default='Mixer_UNet', choices=['Mixer_UNet', 'KANU_Net', 'UNet', 'Vnet', 'R2Unet', 'ResUnetPlusPlus',
                                                                             'AttentionUNet','TransUNet', 'R2UnetMixer', 'Mixer_UNet_Without_MB',
                                                                             'Mixer_UNet_Without_ECA', 'Mixer_UNet_With_ECA_Without_MB', 'MAXFormer',
                                                                             'UNext'])
    parser.add_argument('--dataset', type=str, default='BUSI', choices=['BUSI', 'Kvasir', 'CVCClinicDB', 'ISIC18'])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--model_dir', type=str, default='experiences')
    parser.add_argument('--loss', type=str, default='bcediceloss')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--early_stopping_rounds', type=int, default=20)

    args = parser.parse_args()

    model_name = args.model
    fold = args.fold
    dataset = args.dataset
    model_dir = args.model_dir
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    img_size = args.img_size
    loss = args.loss
    gpu = args.gpu
    early_stopping = args.early_stopping_rounds

    work_dir = os.path.join("logs", model_name)
    os.makedirs(work_dir, exist_ok=True)

    start_time = datetime.now()
    print(f"{start_time.strftime('%Y/%m/%d %H:%M:%S.%f')}: Training started.")
    logfilename = os.path.join(
            work_dir, f"{model_name}_{num_epochs}_{dataset}_{fold}.log")
    logger=get_logger(name=f"{model_name}_{num_epochs}_{dataset}_{fold}.log", log_file=logfilename)

    logger.info(f"\n model: {model_name}")
    logger.info(f"\n dataset: {dataset}")
    logger.info(f"\n fold: {fold}")
    logger.info(f"\n epochs: {num_epochs}")
    logger.info(f"\n batch_size: {batch_size}")
    logger.info(f"\n img_size: {img_size}")

    if dataset == 'BUSI':

        num_classes = 1
        num_channels = 3

        DATA_DIR = os.path.join('data', 'splitted', 'train')
        DATA_DIR_TEST = os.path.join('data', 'splitted', 'test')
        BUS_DATA_PATH = os.path.join('data', 'BUS')

        imagesListTrain = []
        makskListTrain = []

        for idx in range(5):
            if idx == fold:
                imagesListValid = glob(f"{DATA_DIR}/split{fold}/images"+'/*.png')
                maskListValid = glob(f"{DATA_DIR}/split{fold}/masks"+'/*.png')
            else:

                for img in glob(f'{DATA_DIR}/split{idx}/images'+'/*.png'):
                    imagesListTrain.append(img)
                for msk in glob(f'{DATA_DIR}/split{idx}/masks'+'/*.png'):
                    makskListTrain.append(msk)

        imagesListTest = glob(os.path.join(DATA_DIR_TEST, 'images', '*.png'))
        makskListTest = glob(os.path.join(DATA_DIR_TEST, 'masks', '*.png'))

        BUS_test_images = glob(os.path.join(BUS_DATA_PATH, 'original', '*.png'))
        BUS_test_masks = glob(os.path.join(BUS_DATA_PATH, 'GT', '*.png'))

        print(f"Number of training images : {len(imagesListTrain)}, Number of training masks: {len(makskListTrain)}")
        print(f"Number of images of validation images : {len(imagesListValid)}, number of validation masks: {len(maskListValid)}")
        print(f"Number of testing images : {len(imagesListTest)}, Number of testing masks : {len(makskListTest)}")
        print(f"Number of testing images on BUS dataset: {len(BUS_test_images)}, Number of testing masks : {len(BUS_test_masks)}")

        num_masks_test = len(makskListTest)

        transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Transpose(p=0.5),
                ])

        # Create the dataloader for training and validation
        train_dataset = Dataset(imagesListTrain, makskListTrain, transform=transform)
        valid_dataset = Dataset(imagesListValid, maskListValid)
        test_dataset = Dataset(imagesListTest, makskListTest)
        bus_test_dataset = Dataset(BUS_test_images, BUS_test_masks)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        bus_test_loader = torch.utils.data.DataLoader(bus_test_dataset, batch_size=batch_size, shuffle=False)

    elif dataset == 'Kvasir':

        # preprocceing #
        train_transform = transf.ExtCompose([transf.ExtResize((224,224)),
                                        transf.ExtRandomRotation(degrees=90),
                                        transf.ExtRandomHorizontalFlip(),
                                        transf.ExtToTensor(),
                                        ])

        val_transform = transf.ExtCompose([transf.ExtResize((224,224)),
                                    transf.ExtToTensor(),
                                    ])


        train_dataset = KvasirSEG_Dataset(dataset_type='train', transform=train_transform)
        valid_dataset = KvasirSEG_Dataset(dataset_type='val', transform=val_transform)
        test_dataset = KvasirSEG_Dataset(dataset_type='test', transform=val_transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif dataset == 'CVCClinicDB':

        # preprocceing #
        train_transform = transf.ExtCompose([transf.ExtResize((224,224)),
                                        transf.ExtRandomRotation(degrees=90),
                                        transf.ExtRandomHorizontalFlip(),
                                        transf.ExtToTensor(),
                                        ])

        val_transform = transf.ExtCompose([transf.ExtResize((224,224)),
                                    transf.ExtToTensor(),
                                    ])


        train_dataset = CVCClinicDB_Dataset(dataset_type='train', transform=train_transform)
        valid_dataset = CVCClinicDB_Dataset(dataset_type='val', transform=val_transform)
        test_dataset = CVCClinicDB_Dataset(dataset_type='test', transform=val_transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif dataset == 'ISIC18':

        datasets = 'isic18'
        if datasets == 'isic18':
            data_path = './data/isic2018/'
        elif datasets == 'isic17':
            data_path = './data/isic2017/'
        else:
            raise Exception('datasets in not right!')

        train_transformer = transforms.Compose([
            myNormalize(datasets, train=True),
            myToTensor(),
            myRandomHorizontalFlip(p=0.5),
            myRandomVerticalFlip(p=0.5),
            myRandomRotation(p=0.5, degree=[0, 360]),
            myResize(224, 224)
        ])
        test_transformer = transforms.Compose([
            myNormalize(datasets, train=False),
            myToTensor(),
            myResize(224, 224)
        ])

        train_dataset = NPY_datasets(data_path, train_transformer, train=True)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=0)

        val_dataset = NPY_datasets(data_path, test_transformer, train=False)
        valid_loader = torch.utils.data.DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=0,
                                    drop_last=True)

        test_loader = torch.utils.data.DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=0,
                                    drop_last=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'Mixer_UNet':
        model = Mixer_UNet(n_channels=3, n_classes=1).to(DEVICE)
    # elif model_name == 'KANU_Net':
    #     model = KANU_Net().to(DEVICE)
    elif model_name == 'UNet':
        model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    elif model_name == 'Vnet':
        model = Vnet(in_features=3, out_features=1).to(DEVICE)
    elif model_name == 'R2Unet':
        model = R2Unet(in_features=3, out_features=1).to(DEVICE)
    elif model_name == 'ResUnetPlusPlus':
        model = ResUnetPlusPlus(channel=3).to(DEVICE)
    elif model_name == 'AttentionUNet':
        model = AttentionUNet().to(DEVICE)
    elif model_name == "TransUNet":
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 1
        config_vit.n_skip = 3
        vit_name = 'R50-ViT-B_16'
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(224 / 16), int(224 / 16))
        model = ViT_seg(config_vit, img_size=224, num_classes=1).cuda()
    elif model_name == "R2UnetMixer":
        model = R2UnetMixer(in_features=3, out_features=1).to(DEVICE)
    elif model_name == 'Mixer_UNet_Without_ECA':
        model = Mixer_UNet_Without_ECA(n_channels=3, n_classes=1).to(DEVICE)
    elif model_name == 'Mixer_UNet_Without_MB':
        model = Mixer_UNet_Without_MB(n_channels=3, n_classes=1).to(DEVICE)
    elif model_name == 'Mixer_UNet_With_ECA_Without_MB':
        model = Mixer_UNet_With_ECA_Without_MB(n_channels=3, n_classes=1).to(DEVICE)
    elif model_name == "UNext":
        model = UNext(n_features=3, out_features=1, device='cuda').to(DEVICE)
    elif model_name == "MAXFormer":
        model = MAXFormer().to(DEVICE)

    # define the scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda')

    # define the optimizer and the scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    BCE_dice_loss = ut.BCEDiceLoss().to(DEVICE)

    if not os.path.exists(os.path.join(model_dir, model_name, dataset)):
        os.makedirs(os.path.join(model_dir, model_name, dataset))

    # Run the training and validation for the specified number of epochs
    train_loss_list = []
    val_loss_list = []

    best_dice = 0
    trigger = 0

    log = OrderedDict([
            ('epoch', []),
            ('loss', []),
            ('iou', []),
            ('dice', []),
            ('val_loss', []),
            ('val_iou', []),
            ('val_dice', []),
        ])

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        train_log = ut.train_fn(train_loader, model, optimizer, BCE_dice_loss, scaler)
        val_log = ut.val_fn(valid_loader, model,  BCE_dice_loss)

        print('loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_SP %.4f - val_ACC %.4f'
                % (train_log['loss'], train_log['iou'], train_log['dice'], val_log['loss'], val_log['iou'], val_log['dice'], val_log['SE'],
                val_log['PC'], val_log['F1'], val_log['SP'], val_log['ACC']))

        scheduler.step(val_log['loss'])

        log['epoch'].append(epoch)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])

        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv(f'{os.path.join(model_dir, model_name, dataset)}_{fold}.csv', index=False)

        trigger += 1

        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), f'{os.path.join(model_dir, model_name, dataset)}_{fold}.pth')
            best_dice = val_log['dice']
            print("=> saved best model. Best dice: {}".format(best_dice))
            trigger = 0

        # early stopping

        if trigger >= early_stopping:
            print("=> early stopping")
            break
        torch.cuda.empty_cache()

    print('testing...')

    # Test the model on the test dataset
    def test(model, test_loader, device, loss_fn):
        model.eval()

        avg_meters = {'loss': ut.AverageMeter(),
                    'iou': ut.AverageMeter(),
                    'dice': ut.AverageMeter(),
                    'SE':ut.AverageMeter(),
                    'PC':ut.AverageMeter(),
                    'F1':ut.AverageMeter(),
                    'SP':ut.AverageMeter(),
                    'ACC':ut.AverageMeter()
                        }

        with torch.no_grad():
            for image, target in test_loader:
                image, target = image.to(device=device, dtype=torch.float), target.to(device=DEVICE, dtype=torch.float)
                output = model(image)
                test_loss = loss_fn(output, target).item()
                iou, dice, SE, PC, F1, SP, ACC = ut.iou_score(output, target)

                avg_meters['loss'].update(test_loss, image.size(0))
                avg_meters['iou'].update(iou, image.size(0))
                avg_meters['dice'].update(dice, image.size(0))
                avg_meters['SE'].update(SE, image.size(0))
                avg_meters['PC'].update(PC, image.size(0))
                avg_meters['F1'].update(F1, image.size(0))
                avg_meters['SP'].update(SP, image.size(0))
                avg_meters['ACC'].update(ACC, image.size(0))

        return (OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg),
                            ('dice', avg_meters['dice'].avg),
                            ('SE', avg_meters['SE'].avg),
                            ('PC', avg_meters['PC'].avg),
                            ('F1', avg_meters['F1'].avg),
                            ('SP', avg_meters['SP'].avg),
                            ('ACC', avg_meters['ACC'].avg)
                            ]), avg_meters)

    # Test from last epoch
    torch.save(model.state_dict(), f'{os.path.join(model_dir, model_name, dataset)}_training_{fold}.pth')

    test_log, _ = test(model, test_loader, DEVICE, BCE_dice_loss)
    print('test_loss %.4f - test_iou %.4f - test_dice %.4f - test_SE %.4f - test_PC %.4f - test_F1 %.4f - test_SP %.4f - test_ACC %.4f'
                % (test_log['loss'], test_log['iou'], test_log['dice'], test_log['SE'],
                test_log['PC'], test_log['F1'], test_log['SP'], test_log['ACC']))

    logger.info("\n Test (last epoch): test_loss %.4f - test_iou %.4f - test_dice %.4f - test_SE %.4f - test_PC %.4f - test_F1 %.4f - test_SP %.4f - test_ACC %.4f" % (test_log['loss'], test_log['iou'], test_log['dice'], test_log['SE'],
                test_log['PC'], test_log['F1'], test_log['SP'], test_log['ACC']))

    if dataset == 'BUSI':
        bus_test_log, _ = test(model, bus_test_loader, DEVICE, BCE_dice_loss)
        print('BUS test_loss %.4f - test_iou %.4f - test_dice %.4f - test_SE %.4f - test_PC %.4f - test_F1 %.4f - test_SP %.4f - test_ACC %.4f'
                    % (bus_test_log['loss'], bus_test_log['iou'], bus_test_log['dice'], bus_test_log['SE'],
                    bus_test_log['PC'], bus_test_log['F1'], bus_test_log['SP'], bus_test_log['ACC']))

        logger.info("\n BUS Test: test_loss %.4f - test_iou %.4f - test_dice %.4f - test_SE %.4f - test_PC %.4f - test_F1 %.4f - test_SP %.4f - test_ACC %.4f"
         % (bus_test_log['loss'], bus_test_log['iou'], bus_test_log['dice'], bus_test_log['SE'],
                    bus_test_log['PC'], bus_test_log['F1'], bus_test_log['SP'], bus_test_log['ACC']))

    # Test from best saved model
    print('Testing from best saved model...')
    model.load_state_dict(torch.load(f'{os.path.join(model_dir, model_name, dataset)}_{fold}.pth', map_location=DEVICE), strict=True)

    test_log, _ = test(model, test_loader, DEVICE, BCE_dice_loss)
    print('test_loss %.4f - test_iou %.4f - test_dice %.4f - test_SE %.4f - test_PC %.4f - test_F1 %.4f - test_SP %.4f - test_ACC %.4f'
                % (test_log['loss'], test_log['iou'], test_log['dice'], test_log['SE'],
                test_log['PC'], test_log['F1'], test_log['SP'], test_log['ACC']))
    logger.info("\n Test (best model): test_loss %.4f - test_iou %.4f - test_dice %.4f - test_SE %.4f - test_PC %.4f - test_F1 %.4f - test_SP %.4f - test_ACC %.4f" % (test_log['loss'], test_log['iou'], test_log['dice'], test_log['SE'],
                test_log['PC'], test_log['F1'], test_log['SP'], test_log['ACC']))

    if dataset == 'BUSI':
        bus_test_log, _ = test(model, bus_test_loader, DEVICE, BCE_dice_loss)
        print('BUS test_loss %.4f - test_iou %.4f - test_dice %.4f - test_SE %.4f - test_PC %.4f - test_F1 %.4f - test_SP %.4f - test_ACC %.4f'
                    % (bus_test_log['loss'], bus_test_log['iou'], bus_test_log['dice'], bus_test_log['SE'],
                    bus_test_log['PC'], bus_test_log['F1'], bus_test_log['SP'], bus_test_log['ACC']))

        logger.info("\n BUS Test (best model): test_loss %.4f - test_iou %.4f - test_dice %.4f - test_SE %.4f - test_PC %.4f - test_F1 %.4f - test_SP %.4f - test_ACC %.4f"
         % (bus_test_log['loss'], bus_test_log['iou'], bus_test_log['dice'], bus_test_log['SE'],
                    bus_test_log['PC'], bus_test_log['F1'], bus_test_log['SP'], bus_test_log['ACC']))
