# ============================================
# Armor & Ball Detection with Hard-Gated Kalman Tracking
# ============================================
# 设计目标：
# 1. YOLO 负责“发现”，Kalman 负责“连续性”
# 2. 任何 ball 必须通过 Kalman gate，否则直接丢弃
# 3. 轨迹只来自一个来源（YOLO 或 KF 预测）
# 4. 删除所有历史残留逻辑，保证状态机唯一

import cv2
import numpy as np
import time
import argparse
import os
import sys
from openvino.runtime import Core
from filterpy.kalman import KalmanFilter

# ================== CONFIG ==================
MODEL_PATH = "best.onnx"
VIDEO_PATH = "12月19日 中科大 东南.mp4"
INPUT_SIZE = 640

ARMOR_ID = 0
BALL_ID = 1

ARMOR_CONF_THRES = 0.15
BALL_CONF_THRES = 0.25

KALMAN_GATE_DIST = 50.0     # 像素级 gate，超过即认为是新目标/干扰
MAX_MISSED_FRAMES = 5

# ================== UTILS ==================

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (nw, nh))
    pad_w = new_shape[1] - nw
    pad_h = new_shape[0] - nh
    top = pad_h // 2
    left = pad_w // 2
    img_padded = cv2.copyMakeBorder(
        img_resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=color
    )
    return img_padded, scale, left, top


def non_max_suppression(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def create_ball_kf():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    kf.P *= 100.0
    kf.R *= 5.0
    kf.Q *= 0.01
    return kf

# ================== MAIN ==================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=VIDEO_PATH)
    parser.add_argument('--model', default=MODEL_PATH)
    args = parser.parse_args()

    ie = Core()
    compiled_model = ie.compile_model(args.model, args.device.upper())
    output_layer = compiled_model.output(0)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("[ERROR] Cannot open video")
        return

    ball_kf = None
    missed_frames = 0
    ball_traj = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h0, w0 = frame.shape[:2]

        # ---------- KF Predict ----------
        predicted = None
        if ball_kf is not None:
            ball_kf.predict()
            predicted = (float(ball_kf.x[0]), float(ball_kf.x[1]))

        # ---------- YOLO Inference ----------
        img_lb, scale, pad_x, pad_y = letterbox(frame, (INPUT_SIZE, INPUT_SIZE))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]

        pred = compiled_model([img])[output_layer][0]

        balls = []
        armors = []

        for i in range(pred.shape[1]):
            cx, cy, w, h = pred[0:4, i]
            scores = pred[4:, i]
            cid = int(np.argmax(scores))
            conf = float(scores[cid])
            if cid == BALL_ID and conf >= BALL_CONF_THRES:
                cx = (cx - pad_x) / scale
                cy = (cy - pad_y) / scale
                balls.append((cx, cy, conf))
            elif cid == ARMOR_ID and conf >= ARMOR_CONF_THRES:
                cx = (cx - pad_x) / scale
                cy = (cy - pad_y) / scale
                armors.append((cx, cy, conf))

        # ---------- Select Armor ----------
        armor_center = None
        if armors:
            armor_center = max(armors, key=lambda x: x[2])[:2]
            cv2.circle(frame, (int(armor_center[0]), int(armor_center[1])), 5, (0,255,0), -1)

        # ---------- Ball Hard Gate ----------
        accepted_ball = None
        if balls:
            if predicted is None:
                accepted_ball = max(balls, key=lambda x: x[2])
            else:
                for bx, by, bc in balls:
                    if np.hypot(bx - predicted[0], by - predicted[1]) < KALMAN_GATE_DIST:
                        accepted_ball = (bx, by, bc)
                        break

        if accepted_ball is not None:
            bx, by, bc = accepted_ball
            if ball_kf is None:
                ball_kf = create_ball_kf()
                ball_kf.x = np.array([bx, by, 0, 0])
            else:
                ball_kf.update(np.array([bx, by]))
            missed_frames = 0
            ball_traj.append((bx, by))
        else:
            missed_frames += 1
            if missed_frames > MAX_MISSED_FRAMES:
                ball_kf = None
                ball_traj.clear()

        # ---------- Draw ----------
        if predicted is not None:
            cv2.circle(frame, (int(predicted[0]), int(predicted[1])), 4, (0,0,255), -1)

        for i in range(1, len(ball_traj)):
            cv2.line(frame,
                     (int(ball_traj[i-1][0]), int(ball_traj[i-1][1])),
                     (int(ball_traj[i][0]), int(ball_traj[i][1])),
                     (0,255,255), 2)

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
