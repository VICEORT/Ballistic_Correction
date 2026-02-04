"""
Armor detection runner

说明：
- 使用 OpenVINO 加载 YOLOv8 导出的 ONNX (`best.onnx`)。
- 主流程: 读取视频帧 -> 预处理 (letterbox、RGB、归一化) -> 推理 -> 后处理 (坐标映射、NMS、类别筛选) -> 绘制并在窗口显示。
- 不会把处理后的视频保存为文件，程序仅调用 `cv2.imshow` 弹窗显示结果。

如果需要离线保存结果，可在后续步骤中添加视频写出逻辑。
"""

import cv2
import numpy as np
from openvino.runtime import Core
from filterpy.kalman import KalmanFilter
import math
import argparse
import os
import time
import sys

# ================== CONFIG ==================
MODEL_PATH = "best.onnx"
VIDEO_PATH = "Video_20260203120532603.avi"  # 新的视频源路径

INPUT_SIZE = 640
# 调低阈值以减少漏检（模型输出置信度偏低时）
# 将 Armor 阈值进一步降低，方便排查是否为置信度偏低导致的漏检
ARMOR_CONF_THRES = 0.15
BALL_CONF_THRES  = 0.25

ARMOR_ID = 0
BALL_ID  = 1
# ============================================

# ---------------- 预处理 ----------------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    将输入图片按比例缩放并在两侧填充（保持纵横比），返回填充后的图片、缩放比例、pad x 和 pad y。

    这是 YOLO 常用的预处理：缩放到指定长宽，同时在短边做常数填充，
    以保证输入尺寸与模型一致且不发生纵横比变形。
    返回值： (img_padded, scale, pad_x_left, pad_y_top)
    """
    h, w = img.shape[:2]
    scale = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(h * scale), int(w * scale)

    img_resized = cv2.resize(img, (nw, nh))
    pad_w = new_shape[1] - nw
    pad_h = new_shape[0] - nh

    top = pad_h // 2
    left = pad_w // 2

    img_padded = cv2.copyMakeBorder(
        img_resized,
        top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT,
        value=color
    )
    return img_padded, scale, left, top

# ---------------- 图像增强：轻量化对比度增强 ----------------
def apply_simple_contrast(img):
    """简单的线性对比度/亮度调整，用于提升暗/低对比场景下的小物体可见性。

    该函数并不是必须的，只是做一个轻量增强，避免过强的处理影响检测结果。
    """
    alpha = 1.2  # 对比度系数
    beta = 10     # 亮度系数
    enhanced_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return enhanced_img

# ---------------- 优化装甲板框 ----------------
def optimize_armor_box(frame, mask_color_lower=(100, 150, 0), mask_color_upper=(130, 255, 255)):
    """
    简单颜色掩码优化：在给定 ROI 内用 HSV 颜色范围提取可能的装甲区域，
    通过形态学操作与轮廓拟合得到最小外接矩形。用于微调模型给出的粗框。

    返回： (optimized_boxes, mask)
    - optimized_boxes: list of 4-point boxes (np.int32)
    - mask: 二值掩码（调试用）
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, mask_color_lower, mask_color_upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    optimized_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            optimized_boxes.append(box)
    
    return optimized_boxes, mask

# ---------------- 卡尔曼滤波器 ----------------
def create_kalman_filter():
    kf = KalmanFilter(dim_x=7, dim_z=4)
    kf.F = np.eye(7)
    kf.H = np.zeros((4, 7))
    kf.H[:4, :4] = np.eye(4)
    kf.P *= 10
    kf.R = np.eye(4) * 10
    kf.Q = np.eye(7) * 0.01
    return kf


def create_simple_ball_kf():
    # state: [x, y, vx, vy]
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0
    kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
    kf.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
    # 初始协方差、测量噪声、过程噪声的经验值，可根据实际视频帧率与噪声调整
    kf.P *= 100.
    kf.R = np.eye(2) * 5.
    kf.Q = np.eye(4) * 0.01
    return kf


def create_ca_ball_kf():
    """
    恒定加速度（CA）卡尔曼滤波器，状态量为 [x, y, vx, vy, ax, ay]
    测量为 [x, y]
    """
    kf = KalmanFilter(dim_x=6, dim_z=2)
    dt = 1.0
    # 状态转移矩阵 F
    kf.F = np.array([
        [1, 0, dt, 0, 0.5 * dt * dt, 0],
        [0, 1, 0, dt, 0, 0.5 * dt * dt],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=float)
    # 测量矩阵 H
    kf.H = np.zeros((2, 6))
    kf.H[0, 0] = 1
    kf.H[1, 1] = 1
    kf.P *= 100.
    kf.R = np.eye(2) * 5.0
    kf.Q = np.eye(6) * 0.01
    return kf


def predict_via_poly(trajectory, steps_ahead=1):
    """
    使用二次多项式 (y = ax^2 + bx + c) 基于 trajectory 预测下一个位置。
    trajectory: list of (x, y, frame_idx)
    返回预测的 (x, y) 或 None
    """
    if len(trajectory) < 5:
        return None
    xs = np.array([p[0] for p in trajectory])
    ys = np.array([p[1] for p in trajectory])
    try:
        coeffs = np.polyfit(xs, ys, 2)
        # 以 x 增量作为 steps_ahead 估计下一个 x
        last_x = xs[-1]
        avg_dx = np.mean(np.diff(xs)) if len(xs) > 1 else 0
        pred_x = last_x + avg_dx * steps_ahead
        pred_y = np.polyval(coeffs, pred_x)
        return (pred_x, pred_y)
    except Exception:
        return None

# ---------------- 置信度计算 ----------------
def calculate_confidence(yolo_conf, kalman_conf, alpha=0.6):
    """
    基于YOLO检测和卡尔曼预测的置信度加权计算。
    """
    return alpha * yolo_conf + (1 - alpha) * kalman_conf

# ---------------- 匹配检测框和预测框 ----------------
def associate_detections_to_trackers(detections, trackers):
    cost_matrix = np.zeros((len(detections), len(trackers)))
    for i, det in enumerate(detections):
        for j, trk in enumerate(trackers):
            cost_matrix[i, j] = np.linalg.norm(det[:2] - trk[:2])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = list(zip(row_ind, col_ind))
    return matches


def non_max_suppression(boxes, scores, iou_threshold=0.45):
    """
    简单的基于面积和 IoU 的 NMS 实现。
    boxes: list of (x1,y1,x2,y2)
    scores: list of float
    返回保留框的索引列表。
    """
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
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


def draw_status_texts(frame, texts, x=10, y=30, line_height=22, font_scale=0.7):
    """
    按行绘制多条状态文本，避免重叠。
    texts: list of (text, color) 或 list of text (color 默认为绿)
    """
    for i, item in enumerate(texts):
        if isinstance(item, tuple):
            text, color = item
        else:
            text, color = item, (0, 255, 0)
        yi = y + i * line_height
        cv2.putText(frame, text, (x, yi), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

# ---------------- OpenVINO & 设备选择 ----------------
# 支持通过命令行参数 `--device cpu|gpu` 或环境变量 `OV_DEVICE` 选择运行设备。
parser = argparse.ArgumentParser()
parser.add_argument('--device', choices=['cpu', 'gpu', 'CPU', 'GPU'], default=os.getenv('OV_DEVICE', 'CPU'),
                    help='Device to run inference on (CPU or GPU).')
args, unknown = parser.parse_known_args()
device_req = args.device.upper()

ie = Core()
device_name = 'GPU' if device_req == 'GPU' else 'CPU'
try:
    compiled_model = ie.compile_model(MODEL_PATH, device_name)
    print(f"[INFO] Compiled model on device: {device_name}")
except Exception as e:
    # 不允许回退：如果用户指定的设备编译失败，直接退出并返回错误
    print(f"[ERROR] Failed to compile on {device_name}: {e}. Exiting (fallback disabled).")
    sys.exit(1)

output_layer = compiled_model.output(0)

# ---------------- 视频读取 ----------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("[ERROR] 无法打开视频源")
    exit()

# 变量用于保持上一次的输出结果
last_diff = None
trackers = []
ball_kf = None
frame_idx = 0
class_counts = {ARMOR_ID: 0, BALL_ID: 0}
start_time = time.time()
frame_times = []
# 状态变量：用于 Front/Back 判定
ball_was_below = False  # 记录弹丸是否曾经过装甲下边缘
frames_no_ball = 0      # 连续未检测到弹丸的帧数
back_reported = False   # 已报告 Back 状态，避免重复打印
last_horizontal = None  # 'true' or 'Fault'
last_position = None    # 'Front' or 'Back'
# 轨迹和预测相关
ball_trajectory = []    # list of (x, y, frame_idx)
predicted_center = None
disappearance_handled = False
DISAPPEAR_FRAMES = 3   # 连续未检测到球后视为消失
BOOST_RADIUS = 40      # 预测区域内提升置信度的像素半径
BOOST_FACTOR = 1.3     # 预测区域内置信度放大倍数（上限 1.0）
prev_trajectory = []   # 已完成的上一次轨迹（用于消失后显示），在新轨迹开始时清除
last_disappearance_point = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()

    h0, w0 = frame.shape[:2]

    # 先进行卡尔曼预测（如果已有 KF），以便在后续构建检测候选时提升预测区域内的候选置信度
    predicted_center = None
    if ball_kf is not None:
        try:
            # 预测状态（注意：不在此处做 update）
            ball_kf.predict()
            predicted_center = (float(ball_kf.x[0]), float(ball_kf.x[1]))
        except Exception:
            predicted_center = None

    # ---------------- 图像增强 / 预处理 ----------------
    # 1) 可选的轻量对比度增强（对弱对比或暗帧有帮助）
    enhanced_frame = apply_simple_contrast(frame)

    # 2) letterbox -> RGB -> 归一化 -> CHW -> NCHW 扩展
    # 注意：坐标映射时需要用到返回的 scale 和 pad 值
    img_lb, scale, pad_x, pad_y = letterbox(enhanced_frame, (INPUT_SIZE, INPUT_SIZE))
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    input_tensor = np.expand_dims(img_chw, axis=0)

    # ---------------- 推理 ----------------
    # 输入到 OpenVINO 编译模型并取第一个输出（YOLOv8 导出常见格式为 [1,6,N])
    output = compiled_model([input_tensor])[output_layer]
    pred = output[0]  # shape (6, num_preds) -> [cx,cy,w,h,cls0,cls1,...]

    # ---------------- 后处理：收集检测并做 NMS ----------------
    detections = {ARMOR_ID: [], BALL_ID: []}  # per-class list of (x1,y1,x2,y2,conf,center_x,center_y)

    for i in range(pred.shape[1]):
        cx, cy, w, h = pred[0:4, i]
        cls_scores = pred[4:, i]

        cls_id = int(np.argmax(cls_scores))
        conf = float(cls_scores[cls_id])

        if cls_id not in (ARMOR_ID, BALL_ID):
            continue

        # 过滤低置信度
        if (cls_id == ARMOR_ID and conf < ARMOR_CONF_THRES) or (cls_id == BALL_ID and conf < BALL_CONF_THRES):
            continue

        # 把模型坐标转换回原图坐标
        cx_img = (float(cx) - pad_x) / scale
        cy_img = (float(cy) - pad_y) / scale
        bw_img = float(w) / scale
        bh_img = float(h) / scale

        x1 = int(cx_img - bw_img / 2)
        y1 = int(cy_img - bh_img / 2)
        x2 = int(cx_img + bw_img / 2)
        y2 = int(cy_img + bh_img / 2)

        # 截断到图像范围
        x1 = max(0, min(w0 - 1, x1))
        y1 = max(0, min(h0 - 1, y1))
        x2 = max(0, min(w0 - 1, x2))
        y2 = max(0, min(h0 - 1, y2))

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        # 如果有预测中心，在预测半径内的候选给予更高权重，优先被选中
        if predicted_center is not None:
            dx = abs(center_x - predicted_center[0])
            dy = abs(center_y - predicted_center[1])
            if dx <= BOOST_RADIUS and dy <= BOOST_RADIUS:
                conf = min(1.0, conf * BOOST_FACTOR)
        detections[cls_id].append((x1, y1, x2, y2, conf, center_x, center_y))

    # 对每个类别做 NMS
    selected = {ARMOR_ID: [], BALL_ID: []}
    for cid in (ARMOR_ID, BALL_ID):
        boxes = [(d[0], d[1], d[2], d[3]) for d in detections[cid]]
        scores = [d[4] for d in detections[cid]]
        if boxes:
            keep_idx = non_max_suppression(boxes, scores, iou_threshold=0.4)
            for idx in keep_idx:
                selected[cid].append(detections[cid][idx])

    # 选取最可信装甲与弹丸（若存在）
    best_armor = None
    if selected[ARMOR_ID]:
        best_armor = max(selected[ARMOR_ID], key=lambda x: x[4])

    best_ball = None
    if selected[BALL_ID]:
        best_ball = max(selected[BALL_ID], key=lambda x: x[4])

    # ---------------- 绘制 Ball（优先使用检测框；卡尔曼仅在缺失时预测） ----------------
    frame_idx += 1

    if best_ball is not None:
        bx1, by1, bx2, by2, bconf, bcx, bcy = best_ball
        # 新的轨迹开始，清除上一次的已完成轨迹与消失点标记
        prev_trajectory.clear()
        last_disappearance_point = None
        # 增加计数用于诊断类别映射与频率
        class_counts[BALL_ID] += 1

        # 直接用检测框绘制（避免卡尔曼过度平滑导致落在轨迹上）
        cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 0, 255), 2)
        cv2.putText(frame, f"Ball {bconf:.2f}", (int(bx1), max(0, int(by1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 更新/初始化卡尔曼（使用恒定加速度模型 CA），不使用卡尔曼结果覆盖检测框显示
        if ball_kf is None:
            ball_kf = create_ca_ball_kf()
            # 初始化状态 [x,y,vx,vy,ax,ay]
            ball_kf.x = np.array([bcx, bcy, 0., 0., 0., 0.])
        else:
            # 已在帧开始做过 predict，这里直接 update
            try:
                ball_kf.update(np.array([bcx, bcy]))
            except Exception:
                pass
        # 保存轨迹点，用于多项式拟合与消失点判定
        ball_trajectory.append((float(bcx), float(bcy), frame_idx))
        if len(ball_trajectory) > 50:
            ball_trajectory.pop(0)
        disappearance_handled = False
        # 有检测时重置无球计数并清除 back 报告标志
        frames_no_ball = 0
        back_reported = False

    else:
        # 没有检测到弹丸，使用已在帧开始得到的预测位置绘制预测点/框
        if predicted_center is not None:
            pred_x, pred_y = predicted_center
            cv2.circle(frame, (int(pred_x), int(pred_y)), 4, (0, 0, 200), -1)
        # 连续未检测到弹丸帧计数
        frames_no_ball += 1

    # 统计装甲检测次数，便于判断类别是否混淆或置信度过低
    if best_armor is not None:
        class_counts[ARMOR_ID] += 1

    # 每 200 帧打印一次统计信息，帮助诊断 armor 是否被正确预测到
    if frame_idx % 200 == 0:
        print(f"[STAT] frame {frame_idx}: counts: armor={class_counts.get(ARMOR_ID,0)}, ball={class_counts.get(BALL_ID,0)}")

    # ---------------- 优化装甲板框 ----------------
    if best_armor is not None:
        ax1, ay1, ax2, ay2, aconf, acx, acy = best_armor

        # 对装甲板区域进行优化（先确保 ROI 在图像内）
        rx1, ry1, rx2, ry2 = max(0, ax1), max(0, ay1), min(w0 - 1, ax2), min(h0 - 1, ay2)
        if rx2 > rx1 and ry2 > ry1:
            roi = frame[ry1:ry2, rx1:rx2]
            optimized_boxes, mask = optimize_armor_box(roi)
        else:
            optimized_boxes, mask = [], None

        cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 255, 0), 2)
        cv2.putText(frame, f"Armor {aconf:.2f}", (ax1, max(0, ay1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        print(f"[OK] Armor conf = {aconf:.3f}")

        # 检查弹丸消失点判定：若连续若干帧未检测到球，使用轨迹最后点与装甲边界比较
        if frames_no_ball >= DISAPPEAR_FRAMES and (not disappearance_handled) and len(ball_trajectory) > 0:
            last_x, last_y, _ = ball_trajectory[-1]
            if last_y < ay1:
                last_position = 'Back'
            elif last_y > ay2:
                last_position = 'Front'
            # 保存当前完整轨迹以供消失点显示，并记录消失点
            prev_trajectory = ball_trajectory.copy()
            last_disappearance_point = (last_x, last_y)
            # 标记已处理，清空当前轨迹为下一次下落做准备
            disappearance_handled = True
            ball_trajectory.clear()

        # 计算弹丸与装甲板水平坐标差值（使用卡尔曼或检测位置）
        if best_ball is not None:
            _, _, _, _, bconf, bcx, bcy = best_ball
            ball_center = (bcx, bcy)
        elif ball_kf is not None:
            ball_center = (float(ball_kf.x[0]), float(ball_kf.x[1]))
        else:
            ball_center = None

        # 统一构建要显示的状态文本，按行绘制以避免重叠
        status_lines = []

        if ball_center is not None:
            armor_center = (acx, acy)
            horizontal_diff = ball_center[0] - armor_center[0]
            pred_conf = calculate_confidence(bconf if best_ball is not None else 0.0, aconf)
            # 仅在真实检测到 Ball 时更新 last_diff（不要使用卡尔曼预测点来计算 diff）
            if best_ball is not None:
                last_diff = horizontal_diff
            # 显示使用 last_diff（若尚未有检测到的 diff，则显示 N/A 或保留上次值）
            if last_diff is not None:
                status_lines.append((f"Diff: {last_diff:.2f}", (0, 255, 0)))
            else:
                status_lines.append(("Diff: N/A", (0, 255, 0)))
            status_lines.append((f"Ball Conf: {pred_conf:.2f}", (0, 255, 0)))

            # 仅在真实检测到 Ball 时更新 Horizontal 判定（判断投影是否在 armor 区间）
            if best_ball is not None:
                bx = ball_center[0]
                if ax1 <= bx <= ax2:
                    last_horizontal = 'true'
                else:
                    last_horizontal = 'Fault'
            # 位置判定（检测优先）
            if best_ball is not None:
                if bcy > ay2:
                    last_position = 'Front'
                    ball_was_below = True
                    back_reported = False
                else:
                    last_position = 'Back'
            else:
                # 使用卡尔曼位置作为近似
                if ball_center[1] > ay2:
                    last_position = 'Front'
                    ball_was_below = True
                    back_reported = False
                else:
                    last_position = 'Back'
        else:
            if last_diff is not None:
                status_lines.append((f"Diff: {last_diff:.2f}", (0, 255, 0)))
            else:
                status_lines.append(("No Ball detected", (0, 255, 0)))

        # 在顶部按行绘制水平判定和位置判定（使用保留的 last_* 值）
        if last_horizontal is not None:
            color_h = (0, 255, 0) if last_horizontal == 'true' else (0, 0, 255)
            status_lines.append((f"Horizontal: {last_horizontal}", color_h))

        # 如果球曾在装甲下方并且连续若干帧未检测到，则把 last_position 设为 Back
        if ball_was_below and frames_no_ball >= 3 and not back_reported:
            last_position = 'Back'
            back_reported = True

        if last_position is not None:
            color_p = (0, 255, 0) if last_position == 'Front' else (0, 0, 255)
            status_lines.append((f"Position: {last_position}", color_p))

        # 绘制所有状态行，避免重叠
        draw_status_texts(frame, status_lines, x=10, y=30, line_height=24, font_scale=0.7)

    else:
        print("[WARN] No Armor detected")

    # 显示视频
    cv2.imshow("Real-time Detection", frame)

    # 记录本帧处理耗时（含预处理/推理/后处理/绘制）
    t1 = time.time()
    frame_times.append(t1 - t0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 退出前打印性能统计
total_frames = len(frame_times)
total_time = sum(frame_times) if total_frames > 0 else 0.0
avg_fps = (total_frames / total_time) if total_time > 0 else 0.0
avg_frame_ms = (total_time / total_frames * 1000.0) if total_frames > 0 else 0.0
print(f"[PERF] Device={device_name} frames={total_frames} total_time={total_time:.3f}s avg_fps={avg_fps:.2f} avg_frame_ms={avg_frame_ms:.2f}ms")

cap.release()
cv2.destroyAllWindows()
