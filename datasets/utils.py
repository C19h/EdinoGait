import glob
import os
import pickle
import uuid

import PIL.Image as Image
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.sparse import csr_matrix, save_npz
from ultralytics import YOLO
from crop_utils import get_single_image_crop_demo
from unet import EVFlowNet
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = YOLO('./pretrained/best.pt')
# model.to(device)
sensor_size = [260, 346]
base_num_channels = 32
kernel_size = 3

def gen_bboxes(view_path):
    # print(f'开始处理{view_path}')
    event_file = glob.glob(os.path.join(view_path, 'event.txt'))
    ts_f = glob.glob(os.path.join(view_path, 'frames_timestamp.txt'))

    event_list = []
    with open(event_file[0], 'r') as file:
        # 逐行读取文件内容
        for line in file:
            # 将每一行的数据拆分为数字，并将其转换为整数
            row = [int(value) for value in line.split()]
            # 将该行数据添加到数组中
            event_list.append(row)
    event_list = np.array(event_list)

    ts_list = []
    with open(ts_f[0], 'r') as file:
        # 逐行读取文件内容
        for line in file:
            # 将每一行的数据拆分为数字，并将其转换为整数
            row = [int(value) for value in line.split()]
            # 将该行数据添加到数组中
            ts_list.append(row)
    ts_list = np.array(ts_list)

    x, y, t, p = event_list[:, 0], event_list[:, 1], event_list[:, 2], event_list[:, 3]
    bboxes = []
    index = []

    for i in range(len(ts_list)-1):
        events_in_interval = (t >= ts_list[i, 0]) & (t < ts_list[i+1, 0])
        if np.any(events_in_interval):
            x_, y_, t_, p_ = x[events_in_interval], y[events_in_interval], t[
                events_in_interval], p[events_in_interval]
            pts = {'x': x_, 'y': y_, 'ts': t_, 'p': p_}
            # Normalize the data
            pts['ts'] = pts['ts'] - np.min(pts['ts'])
            max_time = np.max(pts['ts'])
            frame_size = sensor_size
            # Generate 3 color image
            Xtore = events2Tore3C(pts['x'], pts['y'], pts['ts'], [max_time], 3, frame_size)
            norm_X = normalize_to_image(Xtore)
            img = Image.fromarray(norm_X, 'RGB')
            temp_img_path = f"./tmp_{uuid.uuid4()}.jpg"
            img.save(temp_img_path)
            # Predict
            try:
                res = model.predict(temp_img_path, save=False, conf=0.6,verbose=False)[0].boxes
                if res.data.shape[0] > 0:
                    confidences = res.data[:, 4]
                    max_conf_index = torch.argmax(confidences)
                    bbox = res.xywh[max_conf_index].cpu().numpy()
                    x_center, y_center, width, height = bbox
                    bbox = np.array([x_center, y_center, width, height])
                    bboxes.append(bbox)
                    index.append(i)
                    
            finally:
                # Remove the temporary image file after prediction
                os.remove(temp_img_path)
    output = {
        "index": torch.from_numpy(np.array(index)).cpu(),
        "bboxes": torch.from_numpy(np.array(bboxes)).cpu()
    }
    with open(os.path.join(view_path, 'bboxes.pkl'), 'wb') as f:
        pickle.dump(output, f)
    print(f'完成{view_path}')


def preprocess(view_path):
    images = sorted(glob.glob(os.path.join(view_path, '*.jpg')))
    event_file = glob.glob(os.path.join(view_path, 'event.txt'))
    ts_f = glob.glob(os.path.join(view_path, 'frames_timestamp.txt'))
    bbox = glob.glob(os.path.join(view_path, 'bboxes.pkl'))
    event_list = []
    with open(event_file[0], 'r') as file:
        # 逐行读取文件内容
        for line in file:
            # 将每一行的数据拆分为数字，并将其转换为整数
            row = [int(value) for value in line.split()]
            # 将该行数据添加到数组中
            event_list.append(row)
    event_list = np.array(event_list)

    ts_list = []
    with open(ts_f[0], 'r') as file:
        # 逐行读取文件内容
        for line in file:
            # 将每一行的数据拆分为数字，并将其转换为整数
            row = [int(value) for value in line.split()]
            # 将该行数据添加到数组中
            ts_list.append(row)
    ts_list = np.array(ts_list)

    with open(bbox[0], 'rb') as file:
        # 逐行读取文件内容
        bboxes = pickle.load(file)
    x, y, t, p = event_list[:, 0], event_list[:, 1], event_list[:, 2], event_list[:, 3]
    frames_idx = bboxes['index']
    image_list = []
    Xtores = []
    for i, idx in enumerate(frames_idx):
        # events_in_interval = (event_list[:, 2] >= ts_list[idx, 0]) & (event_list[:, 2] < ts_list[idx, 1])
        events_in_interval_c = (event_list[:, 2] >= ts_list[idx, 0]) & (event_list[:, 2] < ts_list[idx+1, 0])
        if np.any(events_in_interval_c):
            # x_d, y_d, t_d, p_d = x[events_in_interval], y[events_in_interval], t[events_in_interval], p[events_in_interval]
            x_, y_, t_, p_ = x[events_in_interval_c], y[events_in_interval_c], t[events_in_interval_c], p[events_in_interval_c]
            bbox = bboxes['bboxes'][i]
            # ===================================Xtores====================================================
            pts = {'x': x_, 'y': y_, 'ts': t_, 'p': p_}
            pts['ts'] = pts['ts'] - np.min(pts['ts'])
            max_time = np.max(pts['ts'])
            Xtore = events2ToreFeature(pts['x'], pts['y'], pts['ts'], pts['p'], [max_time], 4, sensor_size)
            Xtore = Xtore.reshape((260, 346, -1))
            crop_Xtore, _, _ = get_single_image_crop_demo(
                Xtore,
                bbox,
                kp_2d=None,
                scale=1,
                crop_size=64,
                norm=False)
            Xtores.append(crop_Xtore)
            # vis_gray(Xtore[:, :, 0],'./g.png')

            # ===================================images====================================================
            image = cv2.imread(images[idx], cv2.IMREAD_GRAYSCALE)
            image = image[..., np.newaxis]
            crop_im, _, _ = get_single_image_crop_demo(
                image,
                bbox,
                kp_2d=None,
                scale=1,
                crop_size=64,
                norm=False)
            image_list.append(crop_im)
            # vis_gray(crop_im,'./crop_gray_{}.png'.format(i))
          

    to_spar(image_list, view_path, 'gray.npz')
    to_spar(Xtores, view_path, 'xtore.npz')


    print(f'完成{view_path}')

def events2ToreFeature(x, y, ts, pol, sampleTimes, k, frameSize):
    oldPosTore, oldNegTore = np.inf * np.ones((frameSize[0], frameSize[1], 2 * k)), np.inf * np.ones(
        (frameSize[0], frameSize[1], 2 * k))
    Xtore = np.zeros((frameSize[0], frameSize[1], 2 * k, len(sampleTimes)), dtype=np.single)

    priorSampleTime = -np.inf

    for sampleLoop, currentSampleTime in enumerate(sampleTimes):
        addEventIdx = (ts >= priorSampleTime) & (ts < currentSampleTime)

        p = addEventIdx & (pol > 0)
        newPosTore = np.full((frameSize[0], frameSize[1], k), np.inf)
        for i, j, t in zip(x[p], y[p], ts[p]):
            newPosTore[j, i] = np.sort(np.partition(np.append(newPosTore[j, i], currentSampleTime - t), k)[:k])

        p = addEventIdx & (pol <= 0)
        newNegTore = np.full((frameSize[0], frameSize[1], k), np.inf)
        for i, j, t in zip(x[p], y[p], ts[p]):
            newNegTore[j, i] = np.sort(np.partition(np.append(newNegTore[j, i], currentSampleTime - t), k)[:k])

        oldPosTore += (currentSampleTime - priorSampleTime)
        oldNegTore += (currentSampleTime - priorSampleTime)

        oldPosTore = np.sort(np.concatenate((oldPosTore, newPosTore), axis=2), axis=2)[:, :, :k]
        oldNegTore = np.sort(np.concatenate((oldNegTore, newNegTore), axis=2), axis=2)[:, :, :k]

        Xtore[:, :, :, sampleLoop] = np.concatenate((oldPosTore, oldNegTore), axis=2).astype(np.single)

        priorSampleTime = currentSampleTime

    # Scale the Tore surface
    minTime = 150
    maxTime = 5e6

    Xtore[np.isnan(Xtore)] = maxTime
    Xtore[Xtore > maxTime] = maxTime

    Xtore = np.log(Xtore + 1)
    Xtore = Xtore - np.log(minTime + 1)
    Xtore[Xtore < 0] = 0
    max = np.max(Xtore)
    Xtore = 1 - (Xtore / max)
    return Xtore

def to_spar(input, view_path, c):
    input_array = np.array(input)
    input_array = input_array.reshape(-1, 1)
    input_array_spar = csr_matrix(input_array)
    save_npz(os.path.join(view_path, c), input_array_spar)



def get_bboxes(events):
    x_, y_, t_, p_ = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    pts = {'x': x_, 'y': y_, 'ts': t_, 'p': p_}
    # Normalize the data
    pts['ts'] = pts['ts'] - np.min(pts['ts'])
    max_time = np.max(pts['ts'])
    frame_size = [260, 346]
    # Generate 6 color image
    Xtore = events2Tore3C(pts['x'], pts['y'], pts['ts'], [max_time], 3, frame_size)
    norm_X = normalize_to_image(Xtore)
    img = Image.fromarray(norm_X, 'RGB')
    temp_img_path = f"./tmp_{uuid.uuid4()}.jpg"
    img.save(temp_img_path)
    # Predict
    try:
        res = model.predict(temp_img_path, save=False, conf=0.6)[0].boxes
        if res.data.shape[0] > 0:
            confidences = res.data[:, 4]
            max_conf_index = torch.argmax(confidences)
            bbox = res.xywh[max_conf_index].cpu().numpy()
            x_center, y_center, width, height = bbox
            new_side = max(width, height)
            bbox = np.array([x_center, y_center, new_side, new_side])
        else:
            bbox = None
    finally:
        # Remove the temporary image file after prediction
        os.remove(temp_img_path)

    return bbox, Xtore

def events2Tore3C(x, y, ts, sampleTimes, k, frameSize):
    toreFeature = np.inf * np.ones((frameSize[0], frameSize[1], k))
    Xtore = np.zeros((frameSize[0], frameSize[1], k, len(sampleTimes)), dtype=np.single)

    priorSampleTime = -np.inf

    for sampleLoop, currentSampleTime in enumerate(sampleTimes):
        addEventIdx = (ts >= priorSampleTime) & (ts < currentSampleTime)

        newTore = np.full((frameSize[0], frameSize[1], k), np.inf)
        for i, j, t in zip(x[addEventIdx], y[addEventIdx], ts[addEventIdx]):
            newTore[j, i] = np.sort(np.partition(np.append(newTore[j, i], currentSampleTime - t), k)[:k])

        toreFeature += (currentSampleTime - priorSampleTime)
        toreFeature = np.sort(np.concatenate((toreFeature, newTore), axis=2), axis=2)[:, :, :k]

        Xtore[:, :, :, sampleLoop] = toreFeature.astype(np.single)

        priorSampleTime = currentSampleTime

    # Scale the Tore surface
    minTime = 150
    maxTime = 5e6

    Xtore[np.isnan(Xtore)] = maxTime
    Xtore[Xtore > maxTime] = maxTime

    Xtore = np.log(Xtore + 1)
    Xtore = Xtore - np.log(minTime + 1)
    Xtore[Xtore < 0] = 0

    return Xtore
def vis(total_frames, ind, pth):
    if len(total_frames.shape) == 3:
        norm_X = normalize_to_image(total_frames[ind])
    else:
        if total_frames.shape[-1] >3:
            norm_X = normalize_to_image(total_frames[ind][..., -3:])
        else:
            norm_X = normalize_to_image(total_frames[ind])
    img = Image.fromarray(norm_X)
    img.save(pth)

def normalize_to_image(array):
    array = np.squeeze(array)

    min_val = np.min(array)
    max_val = np.max(array)
    normalized = (array - min_val) / (max_val - min_val) * 255

    image = normalized.astype(np.uint8)

    return image

def event_formatting(xs, ys, ts, ps):
    """
    Reset sequence-specific variables.
    :param xs: [N] numpy array with event x location
    :param ys: [N] numpy array with event y location
    :param ts: [N] numpy array with event timestamp
    :param ps: [N] numpy array with event polarity ([-1, 1])
    :return xs: [N] tensor with event x location
    :return ys: [N] tensor with event y location
    :return ts: [N] tensor with normalized event timestamp
    :return ps: [N] tensor with event polarity ([-1, 1])
    """

    xs = torch.from_numpy(xs)
    ys = torch.from_numpy(ys)
    ts = torch.from_numpy(ts)
    ps = torch.from_numpy(ps) * 2 - 1
    ts = (ts - ts[0]) / (ts[-1] - ts[0])
    return xs.type(torch.float32), ys.type(torch.float32), ts.type(torch.float32), ps.type(torch.float32)


def create_voxel_encoding(xs, ys, ts, ps, voxel_bins=5, sensor_size=[346, 260]):
    """
    Creates a spatiotemporal voxel grid tensor representation with a certain number of bins,
    as described in Section 3.1 of the paper 'Unsupervised Event-based Learning of Optical Flow,
    Depth, and Egomotion', Zhu et al., CVPR'19..
    Events are distributed to the spatiotemporal closest bins through bilinear interpolation.
    Positive events are added as +1, while negative as -1.
    :param xs: [N] tensor with event x location
    :param ys: [N] tensor with event y location
    :param ts: [N] tensor with normalized event timestamp
    :param ps: [N] tensor with event polarity ([-1, 1])
    :return [B x H x W] event representation
    """

    return events_to_voxel(
        xs,
        ys,
        ts,
        ps,
        voxel_bins,
        sensor_size,
    )


def events_to_voxel(xs, ys, ts, ps, num_bins, sensor_size=sensor_size):
    """
    Generate a voxel grid from input events using temporal bilinear interpolation.
    """

    assert len(xs) == len(ys) and len(ys) == len(ts) and len(ts) == len(ps)

    voxel = []
    ts = ts * (num_bins - 1)
    zeros = torch.zeros(ts.size())
    for b_idx in range(num_bins):
        weights = torch.max(zeros, 1.0 - torch.abs(ts - b_idx))
        voxel_bin = events_to_image(xs, ys, ps * weights, sensor_size=sensor_size)
        voxel.append(voxel_bin)

    return torch.stack(voxel)


def events_to_image(xs, ys, ps, sensor_size=sensor_size):
    """
    Accumulate events into an image.
    """

    device = xs.device
    img_size = list(sensor_size)
    img = torch.zeros(img_size).to(device)

    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    img.index_put_((ys, xs), ps, accumulate=True)

    return img


def create_cnt_encoding(xs, ys, ts, ps):
    """
    Creates a per-pixel and per-polarity event count representation.
    :param xs: [N] tensor with event x location
    :param ys: [N] tensor with event y location
    :param ts: [N] tensor with normalized event timestamp
    :param ps: [N] tensor with event polarity ([-1, 1])
    :return [2 x H x W] event representation
    """

    return events_to_channels(xs, ys, ps, sensor_size=sensor_size)


def events_to_channels(xs, ys, ps, sensor_size=sensor_size):
    """
    Generate a two-channel event image containing event counters.
    """

    assert len(xs) == len(ys) and len(ys) == len(ps)

    mask_pos = ps.clone()
    mask_neg = ps.clone()
    mask_pos[ps < 0] = 0
    mask_neg[ps > 0] = 0

    pos_cnt = events_to_image(xs, ys, ps * mask_pos, sensor_size=sensor_size)
    neg_cnt = events_to_image(xs, ys, ps * mask_neg, sensor_size=sensor_size)

    return torch.stack([pos_cnt, neg_cnt])
