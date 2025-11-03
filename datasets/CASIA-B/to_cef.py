import glob
import os
import sys
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm
import matplotlib.pyplot as plt
from script_utils import get_single_image_crop_demo
import pickle
import events_to_frames
import PIL.Image as Image
import json
sys.path.append('/root/autodl-tmp/EdinoGait')
height = 128
width = 128
k = 4
chunk_len_us = 24*1000
minTime, maxTime = chunk_len_us / k, 256 * 1000

def execute_2frame(view_path):
    origin_path = glob.glob(os.path.join(view_path, 'data.txt'))[0]
    target_path = os.path.join(view_path, 'continuous.npz')
    data = []
    with open(origin_path, 'r') as file:
        # 逐行读取文件内容
        for line in file:
            # 将每一行的数据拆分为数字，并将其转换为整数
            row = [int(value) for value in line.split()]
            # 将该行数据添加到数组中
            data.append(row)
        data = np.array(data)
    data = np.column_stack((data[:, 1], data[:, 2], data[:, 0], data[:, 3]))
    total_frames, bboxes, Xtores, min_max_values = events_to_frames.process_event_stream(data, height, width, chunk_len_us, k, minTime, maxTime)
    if total_frames is None:
        print(origin_path)
    else:
        # total_frames, bboxes = events_to_frames.gen_xtore(data, height, width, chunk_len_us, k)
        b, h, w, _, p = total_frames.shape
        total_frames = total_frames.reshape(b, h, w, -1)

        croped_frames = []
        for i in range(len(total_frames)):
            croped_frame, _, _ = get_single_image_crop_demo(
                total_frames[i].reshape(h, w, -1),
                bboxes[i],
                kp_2d=None,
                scale=1.1,
                crop_size=64,
                norm=False)
            croped_frames.append(croped_frame)
        croped_frames = np.array(croped_frames).astype(np.float32)
        bboxes = np.array(bboxes)
        with open(os.path.join(view_path, 'bboxes.pkl'), 'wb') as f:
            pickle.dump(bboxes, f)
            
        # ind = 5
        # vis(croped_frames, ind, f'./Xtores_{ind}.jpg')
        # vis(total_frames, ind, f'./o_Xtores_{ind}.jpg')

        croped_frames = croped_frames.reshape(-1, k * p)
        f_spar = csr_matrix(croped_frames)
        save_npz(target_path, f_spar)
        del total_frames, croped_frames, f_spar, bboxes

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

def check():
    root_dir = './db/EV_CASIA_B'

    for id_dir in os.listdir(root_dir):  # 遍历id目录
        id_path = os.path.join(root_dir, id_dir)
        if os.path.isdir(id_path):
            for type_dir in os.listdir(id_path):  # 遍历type目录
                type_path = os.path.join(id_path, type_dir)
                if os.path.isdir(type_path):
                    for view_dir in os.listdir(type_path):  # 遍历view目录
                        con_f = os.path.join(type_path, view_dir, 'continuous.npz')
                        b_f = os.path.join(type_path, view_dir, 'bboxes.pkl')
                        delete(con_f)
                        delete(b_f)


def delete(f):
    if os.path.isfile(f):  # 如果文件存在，删除它
        # os.remove(f)
        # print(f"已删除文件: {f}")
        pass
    else:
        print(f"文件 {f} 不存在")


def main(mode):
    root_dirs = ['./db/EV_CASIA_B']
    origin_path_list = []
    with open('./datasets/CASIA-B/CASIA-B.json', "rb") as f:
        partition = json.load(f)
    train = partition["TRAIN_SET"]
    test = partition["TEST_SET"]
    if mode == 'test':
        set = test
    else:
        set = test + train
    for root_dir in root_dirs:
        for id_dir in os.listdir(root_dir):  # 遍历id目录
            if id_dir not in set:
                continue
            else:
                id_path = os.path.join(root_dir, id_dir)
                if os.path.isdir(id_path):
                    for type_dir in os.listdir(id_path):  # 遍历type目录
                        type_path = os.path.join(id_path, type_dir)
                        if os.path.isdir(type_path):
                            for view_dir in os.listdir(type_path):  # 遍历view目录
                                view_path = os.path.join(type_path, view_dir)
                                target_path = os.path.join(view_path, 'continuous.npz')
                                # if os.path.exists(target_path):
                                #     continue
                                # else:
                                origin_path_list.append(view_path)
    if origin_path_list:
        Parallel(n_jobs=8)(
            delayed(execute_2frame)(origin_path_list[i]) for i in tqdm(range(len(origin_path_list))))
    print('complete!!')


if __name__ == '__main__':
    # execute_2frame('./db/EV_CASIA_B/114/nm-04/090')
    main('test')
    # check()

