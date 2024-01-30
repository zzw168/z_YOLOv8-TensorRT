import time

from models import TRTModule  # isort:skip
import argparse
from pathlib import Path

import cv2
import torch

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list


def main(engine='./best.engine', bgr=cv2.imread('./data/01.jpg'), device='cuda:0') -> None:
    engine = './best.engine'
    device = 'cuda:0'
    device = torch.device(device)
    Engine = TRTModule(engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    # for image in images:
    #     save_image = save_path / image.name
    #     bgr = cv2.imread(str(image))
    draw = bgr.copy()
    bgr, ratio, dwdh = letterbox(bgr, (W, H))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)
    dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
    tensor = torch.asarray(tensor, device=device)
    # inference
    start_time = time.time()
    data = Engine(tensor)
    end_time = time.time()
    time_consume = end_time - start_time
    print(time_consume)
    bboxes, scores, labels = det_postprocess(data)
    if bboxes.numel() == 0:
        # if no bounding box
        print(' no object!')
    bboxes -= dwdh
    bboxes /= ratio

    for (bbox, score, label) in zip(bboxes, scores, labels):
        bbox = bbox.round().int().tolist()
        cls_id = int(label)
        cls = CLASSES[cls_id]
        color = COLORS[cls]
        cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
        cv2.putText(draw,
                    f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, [225, 255, 255],
                    thickness=2)
    cv2.imshow('result', draw)
    cv2.waitKey(0)


def set_cap(cap):  # 设置视频截图参数（不压缩图片，节省压缩过程时间）
    W = 1280
    H = 720
    fps = 60.0
    # while W != W1 and H != H1:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, fps)
    W1 = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H1 = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps1 = cap.get(cv2.CAP_PROP_FPS)
    print(f"设置{W1}*{H1}  FPS={fps1}")


def z_infer():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f'无法打开摄像头{0}')
        return
    set_cap(cap)

    engine = './best.engine'
    device = 'cuda:0'
    device = torch.device(device)
    Engine = TRTModule(engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f'无法读取画面{0}')
        bgr = frame
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        # inference
        start_time = time.time()
        data = Engine(tensor)
        end_time = time.time()
        time_consume = end_time - start_time
        print('%f ms' % (time_consume * 1000))

        bboxes, scores, labels = det_postprocess(data)
        if bboxes.numel() == 0:
            # if no bounding box
            print(' no object!')
            cv2.imshow('result', draw)
            cv2.waitKey(1)
            continue
        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw,
                        f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)
        cv2.imshow('result', draw)
        cv2.waitKey(1)


if __name__ == '__main__':
    z_infer()
    #
    # print(torch.__version__)
    #
    # print(torch.version.cuda)
    # print(torch.backends.cudnn.version())
