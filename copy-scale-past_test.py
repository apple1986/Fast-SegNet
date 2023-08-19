import os.path as osp
import os.path
import numpy as np
import random
import cv2
import torch
from torch.utils import data
import pickle
from torchvision.transforms import InterpolationMode
import torchvision
from PIL import Image
from math import ceil, floor
from skimage import measure
import copy
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import d2lzh_pytorch as d2l

## copy and scale
from utils.colorize_mask import camvid_colorize_mask, most_colorize_mask
import torch

def pos_img(img, gt, label_id=0):
    # YN = random.randint(0,1)
    # if (YN) :
    #     return img, gt

    img_ori = copy.deepcopy(img)
    gt_ori = copy.deepcopy(gt)

    img_change = copy.deepcopy(img)
    gt_change = copy.deepcopy(gt)

    # h_img, w_img, _ = img.shape
    # w_img, h_img = img.size
    ## 1. extract the selected object
    gt = (gt * (gt == label_id))

    bw = (gt > 0).astype(int)
    labels = measure.label(bw)

    x_area = []
    y_area = []

    if (len(np.unique(labels)) > 1):
        ## get obj coord
        ## select one object
        for n in range(1, labels.max() + 1):  # labels.max()+1
            obj_num = np.sum(labels == n)
            if obj_num <= 400:
                continue
            x, y = (labels == n).nonzero()  ## obtain idx
            x.tolist()
            x_area.extend(x)
            y.tolist()
            y_area.extend(y)

    return x_area, y_area


def copy_obj_scale(img, gt, img2, gt2, label_id, scale_rate):
    # YN = random.randint(0,1)
    # if (YN) :
    #     return img, gt
    img_ori = copy.deepcopy(img)
    gt_ori = copy.deepcopy(gt)

    img_ori1 = copy.deepcopy(img2)
    gt_ori1 = copy.deepcopy(gt2)

    img_ori2 = copy.deepcopy(img2)
    gt_ori2 = copy.deepcopy(gt2)

    gt_temp = copy.deepcopy(gt)

    img_change = copy.deepcopy(img2)
    gt_change = copy.deepcopy(gt2)

    x_area, y_area = pos_img(img=img_ori2, gt=gt_ori2, label_id=3)



    h_img, w_img, _ = img2.shape

    # print(h_img,w_img)

    ## 1. extract the selected object
    gt = (gt * (gt == label_id))
    # gt_temp = (gt_temp * (gt_temp == 2))
    # gt = gt + gt_temp

    bw = (gt > 0).astype(int)
    labels = measure.label(bw)

    if (len(np.unique(labels)) > 1):
        ## get obj coord
        ## select one object
        for n in range(1, labels.max() + 1):  # labels.max()+1
            obj_num = np.sum(labels == n)
            if (obj_num >= 7000) or (obj_num <= 6000):
                continue

            x, y = (labels == n).nonzero()  ## obtain idx
            ## 2. select bounding box: obtain the begining coord
            # x, y = (gt_1 > 0).nonzero()
            # if (any(x)):
            sel_obj = gt_ori[min(x):max(x) + 1, min(y):max(y) + 1]
            sel_img = img_ori[min(x):max(x) + 1, min(y):max(y) + 1, :]
            h_pre, w_pre = sel_obj.shape
            sel_img = Image.fromarray(sel_img)
            sel_obj = Image.fromarray(sel_obj)

            ## sacling
            shape_aug_img = torchvision.transforms.Resize(size=(int(h_pre * scale_rate), int(w_pre * scale_rate)),
                                                          interpolation=InterpolationMode.BICUBIC)
            shape_aug_gt = torchvision.transforms.Resize(size=(int(h_pre * scale_rate), int(w_pre * scale_rate)),
                                                         interpolation=InterpolationMode.NEAREST)

            img_want = shape_aug_img(sel_img)
            gt_want = shape_aug_gt(sel_obj)

            sel_img = np.array(img_want)
            sel_obj = np.array(gt_want)

            h_obj, w_obj = sel_obj.shape
            # print(h_obj,w_obj)



            if len(x_area) == 0:
                x_end_pos = random.randint(0, h_img)
                y_end_pos = random.randint(0, w_img)
            else:
                length = len(x_area)-1
                idx1 = random.randint(0, length)
                x_end_pos = x_area[idx1]+10
                y_end_pos = y_area[idx1]

                # print(x_beg_pos,y_beg_pos)

            if x_end_pos > h_img:
                x_end_pos = h_img

            if (x_end_pos - h_obj) >= 0:
                x_beg_pos = x_end_pos - h_obj
            else:
                x_beg_pos = 0
                x_end_pos = x_beg_pos + h_obj

            if (y_end_pos - w_obj) >= 0:
                y_beg_pos = y_end_pos - w_obj
            else:
                y_beg_pos = 0
                y_end_pos = y_beg_pos + w_obj

            ## copy obj
            gt_change[x_beg_pos:x_end_pos, y_beg_pos:y_end_pos] = sel_obj[0:h_obj, 0:w_obj] + 1
            img_change[x_beg_pos:x_end_pos, y_beg_pos:y_end_pos] = sel_img[0:x_end_pos - x_beg_pos,
                                                                      0:y_end_pos - y_beg_pos]
    idx = np.where(gt_change != (label_id +1))
    # idx = np.where((gt_change != label_id) & (gt_change != 2))
    gt_change[idx] = gt_ori1[idx]
    idx1 = np.where(gt_change == (label_id+1))
    gt_change[idx1] = 9

    for n in range(0, 3):
        img_changed_tmp = img_change[:, :]
        img_ori_tmp = img_ori1[:, :]
        img_changed_tmp[idx] = img_ori_tmp[idx]
        img_change[:, :] = img_changed_tmp
        # gt_ori[np.argwhere(gt_used == label_id)]=label_id

    return img_change, gt_change

root_path1 = os.path.split(os.path.abspath(__file__))[0]
root_path_train = os.path.join(root_path1, 'cityscapes/leftImg8bit/train/bochum')
root_path_trainnot = os.path.join(root_path1, 'cityscapes/gtFine/train/bochum')

root_path_cam = os.path.join(root_path1, 'camvid/train')
root_path_cam_trainnot = os.path.join(root_path1, 'camvid/trainannot')

save_path = os.path.join(root_path1, 'new_camvid')

save_path_label = os.path.join(save_path, 'color')
save_path_train = os.path.join(save_path, 'train')
save_path_trainannot = os.path.join(save_path, 'gt')

lst_train = os.listdir(root_path_cam)
lst_trainnot = os.listdir(root_path_cam_trainnot)

lst_train.remove('.gitkeep')
lst_trainnot.remove('.gitkeep')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

i = len(lst_train)


for i in range(i):
    img_int_path = os.path.join(root_path_train,'bochum_000000_003674_leftImg8bit.png')
    img_int_gt_path=os.path.join(root_path_trainnot,'bochum_000000_003674_gtFine_labelTrainIds.png')


    img_init= Image.open(img_int_path)
    img_init = np.array(img_init)
    # img = torch.tensor(img)
    # img = np.array(img)
    # h_img, w_img, _ = img.shape#
    gt_init = Image.open(img_int_gt_path)
    gt_init = np.array(gt_init)


    img_path = os.path.join(root_path_cam, lst_train[i])
    gt_path = os.path.join(root_path_cam_trainnot, lst_trainnot[i])


    img = Image.open(img_path)
    img = np.array(img)

    # img = torch.tensor(img)
    # img = np.array(img)
    # h_img, w_img, _ = img.shape
    gt = Image.open(gt_path)
    gt = np.array(gt)

    # if(img_init.shape != img.shape):
    #     x, y, _ = img.shape
    #     img_init = cv2.resize(img_init, dsize=(y, x), interpolation=cv2.INTER_LINEAR)
    #     gt_init = cv2.resize(gt_init, dsize=(y, x), interpolation=cv2.INTER_NEAREST)

    # gt_per = (gt_init * (gt_init == 11))
    # bw = (gt_per > 0).astype(int)
    # labels = measure.label(bw)
    # if (len(np.unique(labels)) > 1):
    #     ## get obj coord
    #     ## select one object
    #     for n in range(1, labels.max() + 1):  # labels.max()+1
    #         obj_num = np.sum(labels == n)
    #         print(obj_num)

    # gt = torch.tensor(gt)

    # save_img = os.path.join(save_path, lst_train[i])
    # output_color = camvid_colorize_mask(gt)
    # output_color.save(save_img)

    # label_ids = [1,2]
    # label_id = label_ids[random.randint(0, 4)]

    scale_rates=random.uniform(0.6,0.9)
    # img_aug, gt_aug = obj_scale_aug(img, gt, label_id=9, scale_rate=1.8) #random.uniform(1, 3)
    img_aug, gt_aug = copy_obj_scale(img_init, gt_init, img, gt,  label_id=11, scale_rate = scale_rates)
    # img_aug, gt_aug = obj_scale_aug(img, gt, label_id=9, scale_rate=np.random.randint(1,4))
    img_aug = Image.fromarray(img_aug)
    save_img = os.path.join(save_path_train, "PC_" + lst_train[i])
    img_aug.save(save_img)

    # print(gt_aug)
    gt_aug1 = Image.fromarray(gt_aug.astype(np.uint8)).convert('P')
    save_img = os.path.join(save_path_trainannot, "PC_" + lst_train[i])
    gt_aug1.save(save_img)

    save_img = os.path.join(save_path_label, "PC_" + lst_train[i])
    output_color = most_colorize_mask(gt_aug)
    output_color.save(save_img)

