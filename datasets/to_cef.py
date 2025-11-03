import glob
import os
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm
import PIL.Image as Image
from utils import vis
from crop_utils import get_single_image_crop_demo
sys.path.append('/root/autodl-tmp/EdinoGait')
height = 260
width = 346
k = 4
chunk_len_us = 28341
minTime, maxTime = chunk_len_us / k, 28341 * 2


# total_events -> [(x,y,t,p)]
def process_event_stream(ts_list, total_events, height, width, chunk_len_us, k, minTime, maxTime, pos_fifo=None,
                         neg_fifo=None):
    # Remove consecutive events -> make sure that events for a certain pixel are at least minTime away
    if minTime > 0:
        total_events = total_events[::-1]  # Reverse to extract higher time-events -> unique sort later in increase order again
        orig_time = total_events[:, 2].copy()  # Save a copy of the original time-stamps
        total_events[:, 2] = total_events[:, 2] - np.mod(total_events[:, 2], minTime)  # Binarize by minTime
        uniq, inds = np.unique(total_events, return_index=True, axis=0)  # Extract unique binarized events
        total_events = total_events[inds]
        total_events[:, 2] = orig_time[inds]  # Roll back to the original time-stamps

    min_max_values = []
    # List time-window timestamp endings
    # tw_ends = np.arange(int(total_events[:, 2].max()), int(total_events[:, 2].min()), -chunk_len_us)[::-1]
    # tw_ends = np.arange(int(ts_list[-1, 0]), int(ts_list[0, 0]), -chunk_len_us)[::-1]

    # True to return the FIFOs
    return_mem = pos_fifo is not None and not neg_fifo is None

    # FIFOs of size K for the positive and negative events
    if pos_fifo is None: pos_fifo = np.full((height, width, k), -np.inf, dtype=np.float64)
    if neg_fifo is None: neg_fifo = np.full((height, width, k), -np.inf, dtype=np.float64)

    frames = []
    time_steps = []
    for i in range(len(ts_list)-1):
        # Select events within the slice
        tw_events_inds = (total_events[:, 2] > ts_list[i, 0]) & (total_events[:, 2] <= ts_list[i+1, 0])
        new_pos, new_neg, min_max = get_representation(total_events, tw_events_inds, k, height, width)
        if new_pos is None or (new_pos.sum() == 0.0 and new_neg.sum() == 0.0):
            print(f'*** {i} | Empty window: p0 {total_events[(tw_events_inds) & (total_events[:, 3] == 0)].shape[0]} | p1 {total_events[(tw_events_inds) & (total_events[:, 3] == 0)].shape[1]}')  # ', cat_id, num_sample, sampleLoop
            # print(f'*** {sampleLoop} | Empty window: p0 {total_events[(tw_events_inds) & (total_events[:,3]==0)].shape[0]} | p1 {total_events[(tw_events_inds) & (total_events[:,3]==0)].shape[1]}')   # ', cat_id, num_sample, sampleLoop

        # Update fifos. Append new events, move zeros to the beggining and retain last k events for each pixel/polarity
        pos_fifo = np.sort(np.concatenate([pos_fifo, new_pos], axis=2), axis=2)[:, :, -k:]
        neg_fifo = np.sort(np.concatenate([neg_fifo, new_neg], axis=2), axis=2)[:, :, -k:]

        # Build frame by stacking positive and negative fifo representations
        frame = np.stack([neg_fifo, pos_fifo], axis=-1)
        frames.append(frame)
        time_steps.append(ts_list[i+1, 0])
        min_max_values.append(min_max)

    if len(frames) == 0: return []
    frames = np.stack(frames)
    time_steps = np.array(time_steps)

    # Make each window in the range (0, maxTime)
    diff = maxTime - time_steps
    diff = diff[:, None, None, None, None].astype('float64')
    frames = frames + diff  # Make newer events to have higher value than the older ones
    frames[frames < 0] = 0
    frames = (frames / maxTime).astype(
        'float64')  # Make newer events to have a value close to 1 and older ones a value close to 0
    if return_mem:
        return (frames, min_max_values), (pos_fifo, neg_fifo)
    else:
        return frames, min_max_values


def get_representation(total_events, tw_events_inds, k, height, width):
    # Select events for the current time-window
    pos_inds = (tw_events_inds) & (total_events[:, 3] == 1)
    pos_events = total_events[pos_inds]
    # Sort events by y, x, timestamp
    pos_events = pos_events[np.lexsort((pos_events[:, 2], pos_events[:, 1], pos_events[:, 0]))]
    # Aggregate events per pixel. Get unique event coordinates -> avoid duplicates
    unique_coords_pos, unique_indexes_pos = np.unique(pos_events[:, :2], return_index=True, axis=0)
    new_pos = events_to_frame_v0(pos_events, unique_coords_pos, unique_indexes_pos, height, width, k)
    # new_pos = events_to_frame(pos_events, unique_coords_pos, unique_indexes_pos, height, width, k)

    # Select  events for the current time-window
    neg_inds = (tw_events_inds) & (total_events[:, 3] == 0)
    neg_events = total_events[neg_inds]
    # Sort events by y, x, timestamp
    neg_events = neg_events[np.lexsort((neg_events[:, 2], neg_events[:, 1], neg_events[:, 0]))]
    # Aggregate events per pixel. Get unique event coordinates -> avoid duplicates
    unique_coords_neg, unique_indexes_neg = np.unique(neg_events[:, :2], return_index=True, axis=0)
    new_neg = events_to_frame_v0(neg_events, unique_coords_neg, unique_indexes_neg, height, width, k)
    # new_neg = events_to_frame(neg_events, unique_coords_neg, unique_indexes_neg, height, width, k)

    # More recent samples are close to zero 0
    if len(unique_coords_pos) == 0 and len(unique_coords_neg) == 0:
        mins = maxs = (0, 0)
    elif len(unique_coords_pos) == 0:
        mins, maxs = unique_coords_neg.min(0), unique_coords_neg.max(0)
    elif len(unique_coords_neg) == 0:
        mins, maxs = unique_coords_pos.min(0), unique_coords_pos.max(0)
    else:
        mins, maxs = np.concatenate([unique_coords_pos, unique_coords_neg], axis=0).min(0), np.concatenate(
            [unique_coords_pos, unique_coords_neg], axis=0).max(0)
    min_max = (int(mins[1]), int(maxs[1]), int(mins[0]), int(maxs[0]))

    return new_pos, new_neg, min_max


def events_to_frame_v0(events, unique_coords_pos, unique_indexes_pos, height, width, k):
    # Initialize positive frame
    new_pos = np.full((height, width, k), 0, dtype=np.float64)  # Initialize frame representation
    if not len(unique_coords_pos) == 0:
        agg_pos = np.split(events[:, 2], unique_indexes_pos[1:])
        # Get only the last k events for each coordinate
        agg_k_pos = [pix_agg[-k:] for pix_agg in agg_pos]
        # List of the last k events per pixel
        agg_k_pos = np.array([np.pad(pix_agg, (k - len(pix_agg), 0)) for pix_agg in agg_k_pos])
        new_pos[unique_coords_pos[:, 1], unique_coords_pos[:, 0]] = agg_k_pos
    return new_pos

def execute_2frame(view_path, target_path):
    event_file = glob.glob(os.path.join(view_path, 'event.txt'))
    ts_f = glob.glob(os.path.join(view_path, 'frames_timestamp.txt'))
    bbox = glob.glob(os.path.join(view_path, 'bboxes.pkl'))
    event_list = []

    images = sorted(glob.glob(os.path.join(view_path, '*.jpg')))

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

    # events_in_interval = (event_list[:, 2] >= ts_list[0, 0]) & (event_list[:, 2] < ts_list[-1, 0])
    # event_list = event_list[events_in_interval]

    with open(bbox[0], 'rb') as file:
        # 逐行读取文件内容
        bboxes = pickle.load(file)

    total_frames, min_max_values = process_event_stream(ts_list, event_list, height, width, chunk_len_us, k, minTime,
                                                        maxTime)
    # vis(total_frames,50)
    # vis_gray(images, 50)
    b, h, w, _, p = total_frames.shape
    total_frames = total_frames.reshape(b, h, w, -1)
    croped_frames = []
    croped_gray = []
    frames_idx = bboxes['index']
    for i, idx in enumerate(frames_idx):
        croped_frame, _, _ = get_single_image_crop_demo(
            total_frames[idx],
            bboxes['bboxes'][i],
            kp_2d=None,
            scale=1,
            crop_size=64,
            norm=False)
        croped_frames.append(croped_frame)
        # image = cv2.imread(images[idx], cv2.IMREAD_GRAYSCALE)
        # image = image[..., np.newaxis]
        # crop_im, _, _ = get_single_image_crop_demo(
        #     image,
        #     bboxes['bboxes'][i],
        #     kp_2d=None,
        #     scale=1,
        #     crop_size=64,
        #     norm=False)
        # croped_gray.append(crop_im)
    croped_frames = np.array(croped_frames).astype(np.float32)

    # ind = 5
    # vis(croped_frames, ind, f'./Xtores_{ind}.jpg')
    # vis(total_frames, ind, f'./o_Xtores_{ind}.jpg')

    croped_frames = croped_frames.reshape(-1, k * p)
    f_spar = csr_matrix(croped_frames)
    save_npz(target_path, f_spar)
    del total_frames, croped_frames, f_spar, bboxes


def main():
    root_dirs = ['../ClefGait_/db/HNU-Gait/dark', '../ClefGait_/db/HNU-Gait/light']
    origin_path_list = []
    target_path_list = []
    for root_dir in root_dirs:
        for id_dir in os.listdir(root_dir):  # 遍历id目录
            id_path = os.path.join(root_dir, id_dir)
            if os.path.isdir(id_path):
                for type_dir in os.listdir(id_path):  # 遍历type目录
                    type_path = os.path.join(id_path, type_dir)
                    if os.path.isdir(type_path):
                        for view_dir in os.listdir(type_path):  # 遍历view目录
                            view_path = os.path.join(type_path, view_dir)
                            target_path = os.path.join(view_path, 'continuous.npz')
                            if os.path.exists(target_path):
                                continue
                            else:
                                origin_path_list.append(view_path)
                                target_path_list.append(target_path)
    if origin_path_list:
        Parallel(n_jobs=8)(
            delayed(execute_2frame)(origin_path_list[i], target_path_list[i]) for i
            in
            tqdm(range(len(origin_path_list))))
    print('complete!!')


def check():
    root_dir = './db/HNU-Gait'
    for c in os.listdir(root_dir):
        c_path = os.path.join(root_dir, c)
        for id_dir in os.listdir(c_path):  # 遍历id目录
            id_path = os.path.join(c_path, id_dir)
            if os.path.isdir(id_path):
                for type_dir in os.listdir(id_path):  # 遍历type目录
                    type_path = os.path.join(id_path, type_dir)
                    if os.path.isdir(type_path):
                        for view_dir in os.listdir(type_path):  # 遍历view目录
                            con_f = os.path.join(type_path, view_dir, 'continuous.npz')
                            delete(con_f)


def delete(f):
    if os.path.isfile(f):  # 如果文件存在，删除它
        # os.remove(f)
        # print(f"已删除文件: {f}")
        pass
    else:
        print(f"文件 {f} 不存在")


if __name__ == '__main__':
    main()
    # check()
