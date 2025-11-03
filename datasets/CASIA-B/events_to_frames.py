import numpy as np
from ultralytics import YOLO
import torch
import os
import sys
sys.path.append('./pretrained')
model = YOLO('./pretrained/best.pt')
model.to('cuda')
import PIL.Image as Image
import uuid


# total_events -> [(x,y,t,p)]
def process_event_stream(total_events, height, width, chunk_len_us, k, minTime, maxTime, pos_fifo=None, neg_fifo=None,
                         return_tw_ends=False):
    # Remove consecutive events -> make sure that events for a certain pixel are at least minTime away
    if minTime > 0:
        total_events = total_events[
                       ::-1]  # Reverse to extract higher time-events -> unique sort later in increase order again
        orig_time = total_events[:, 2].copy()  # Save a copy of the original time-stamps
        total_events[:, 2] = total_events[:, 2] - np.mod(total_events[:, 2], minTime)  # Binarize by minTime
        uniq, inds = np.unique(total_events, return_index=True, axis=0)  # Extract unique binarized events
        total_events = total_events[inds]
        total_events[:, 2] = orig_time[inds]  # Roll back to the original time-stamps

    min_max_values = []
    # List time-window timestamp endings
    tw_ends = np.arange(int(total_events[:, 2].max()), int(total_events[:, 2].min()), -chunk_len_us)[::-1]

    tw_init = -np.inf

    # True to return the FIFOs
    return_mem = pos_fifo is not None and not neg_fifo is None

    # FIFOs of size K for the positive and negative events
    if pos_fifo is None: pos_fifo = np.full((height, width, k), -np.inf, dtype=np.float64)
    if neg_fifo is None: neg_fifo = np.full((height, width, k), -np.inf, dtype=np.float64)

    frames = []
    time_steps = []
    bboxes = []
    Xtores = []
    for tw_num, current_tw_end in enumerate(tw_ends):

        # Select events within the slice
        tw_events_inds = (total_events[:, 2] > tw_init) & (total_events[:, 2] <= current_tw_end)
        if np.any( tw_events_inds):
        # Get pos/neg frame representations
            bbox, Xtore = get_bboxes(total_events[tw_events_inds])
        else:
            continue
        if bbox is None:
            continue
        bboxes.append(bbox)
        Xtores.append(Xtore)
        new_pos, new_neg, min_max = get_representation(total_events, tw_events_inds, k, height, width)
        if new_pos is None or (new_pos.sum() == 0.0 and new_neg.sum() == 0.0):
            if tw_num != 0: print(
                f'*** {tw_num} | Empty window: p0 {total_events[(tw_events_inds) & (total_events[:, 3] == 0)].shape[0]} | p1 {total_events[(tw_events_inds) & (total_events[:, 3] == 0)].shape[1]}')  # ', cat_id, num_sample, sampleLoop
            # print(f'*** {sampleLoop} | Empty window: p0 {total_events[(tw_events_inds) & (total_events[:,3]==0)].shape[0]} | p1 {total_events[(tw_events_inds) & (total_events[:,3]==0)].shape[1]}')   # ', cat_id, num_sample, sampleLoop
            continue

        # Update fifos. Append new events, move zeros to the beggining and retain last k events for each pixel/polarity
        pos_fifo = np.sort(np.concatenate([pos_fifo, new_pos], axis=2), axis=2)[:, :, -k:]
        neg_fifo = np.sort(np.concatenate([neg_fifo, new_neg], axis=2), axis=2)[:, :, -k:]

        # Build frame by stacking positive and negative fifo representations
        frame = np.stack([neg_fifo, pos_fifo], axis=-1)
        frames.append(frame)

        tw_init = current_tw_end
        min_max_values.append(min_max)
        time_steps.append(current_tw_end)

    if len(frames) == 0: return None, None, None, None
    frames = np.stack(frames)
    time_steps = np.array(time_steps)

    # Make each window in the range (0, maxTime)
    diff = maxTime - time_steps
    diff = diff[:, None, None, None, None].astype('float64')
    frames = frames + diff  # Make newer events to have higher value than the older ones
    frames[frames < 0] = 0
    frames = (frames / maxTime).astype('float64')  # Make newer events to have a value close to 1 and older ones a value close to 0
    if return_mem:
        return (frames, min_max_values), (pos_fifo, neg_fifo)
    else:
        if return_tw_ends:
            return frames, bboxes, min_max_values, tw_ends
        else:
            return frames, bboxes, Xtores, min_max_values


def get_bboxes(events):
    x_, y_, t_, p_ = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    pts = {'x': x_, 'y': y_, 'ts': t_, 'p': p_}
    # Normalize the data
    pts['ts'] = pts['ts'] - np.min(pts['ts'])
    max_time = np.max(pts['ts'])
    frame_size = [128, 128]
    # Generate 6 color image
    Xtore = events2Tore3C(pts['x'], pts['y'], pts['ts'], [max_time], 3, frame_size)
    norm_X = normalize_to_image(Xtore)
    img = Image.fromarray(norm_X, 'RGB')
    temp_img_path = f"./tmp_{uuid.uuid4()}.jpg"
    img.save(temp_img_path)
    # Predict
    try:
        res = model.predict(temp_img_path, save=False, conf=0.6, verbose=False)[0].boxes
        if res.data.shape[0] > 0:
            confidences = res.data[:, 4]
            max_conf_index = torch.argmax(confidences)
            bbox = res.xywh[max_conf_index].cpu().numpy()
        else:
            bbox = None
    finally:
        # Remove the temporary image file after prediction
        os.remove(temp_img_path)

    return bbox, Xtore

def normalize_to_image(array):
    array = np.squeeze(array)

    min_val = np.min(array)
    max_val = np.max(array)
    normalized = (array - min_val) / (max_val - min_val) * 255

    image = normalized.astype(np.uint8)

    return image


def events_to_frame_v0(events, unique_coords_pos, unique_indexes_pos, height, width, k):
    # Initialize positive frame
    new_pos = np.full((height, width, k), 0, dtype=np.float64)  # Initialize frame representation
    if not len(unique_coords_pos) == 0:
        agg_pos = np.split(events[:, 2], unique_indexes_pos[1:])
        # Get only the last k events for each coordinate
        agg_k_pos = [pix_agg[-k:] for pix_agg in agg_pos]
        # List of the last k events per pixel
        agg_k_pos = np.array([np.pad(pix_agg, (k - len(pix_agg), 0)) for pix_agg in agg_k_pos])
        new_pos[width-unique_coords_pos[:, 1]-1, unique_coords_pos[:, 0]] = agg_k_pos
    return new_pos


# Create a frame representation of the given events
def events_to_frame(events, unique_coords_pos, unique_indexes_pos, height, width, k):
    # Initialize frame
    new_pos = np.full((height * width * k), 0, dtype=np.float64)  # Initialize frame representation
    if not len(unique_coords_pos) == 0:
        true_inds, k_inds = [], []
        prev_ind = -1
        # true_inds: calculate the positions of events belonging to each coordinate
        # k_inds: calculate the position k of each event
        for num_item, i in enumerate(unique_indexes_pos):
            current_true_ind = 1 + np.arange(max(prev_ind, i - k, 0), i)
            true_inds.append(current_true_ind)
            k_inds.append(np.arange(k - len(current_true_ind), k))
            prev_ind = i
        true_inds = np.concatenate(true_inds)
        k_inds = np.concatenate(k_inds)

        events = events[true_inds]
        # Transform pixel and k array coordinates to ravel array position
        true_coords = np.concatenate([events[:, [1, 0]], k_inds[:, None]], axis=1, dtype='int64')
        true_coords_inds = np.ravel_multi_index(true_coords.transpose(), (height, width, k))

        # Add time-stamp information to the empty ravel frame
        new_pos[true_coords_inds] = events[:, 2]
    # Reshape ravel frame
    new_pos = new_pos.reshape(height, width, k)
    return new_pos


# Transform a list of events into a positive and negative frame
# Frames contains the last k events (their time-stamp) from total_events for each pixel
# This frame only contains event information from the events (total_events) of the current time-window
# Older events from the FIFOs will be added later if needed
# total_events -> [(x,y,t,p)]
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


def events2Tore3C(x, y, ts, sampleTimes, k, frameSize):
    toreFeature = np.inf * np.ones((frameSize[0], frameSize[1], k))
    Xtore = np.zeros((frameSize[0], frameSize[1], k, len(sampleTimes)), dtype=np.single)

    priorSampleTime = -np.inf

    for sampleLoop, currentSampleTime in enumerate(sampleTimes):
        addEventIdx = (ts >= priorSampleTime) & (ts < currentSampleTime)

        newTore = np.full((frameSize[0], frameSize[1], k), np.inf)
        for i, j, t in zip(x[addEventIdx], y[addEventIdx], ts[addEventIdx]):
            v = i
            u = frameSize[0] - j - 1
            newTore[u, v] = np.sort(np.partition(np.append(newTore[u, v], currentSampleTime - t), k)[:k])

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

    return Xtore
