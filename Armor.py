import cv2
import numpy as np
from openvino.runtime import Core
from filterpy.kalman import KalmanFilter
import time
import argparse
import os
import sys

# ================== CONFIG ==================
MODEL_PATH = "best.onnx"
VIDEO_PATH = "12月19日 中石油 大工 2.mp4"

INPUT_SIZE = 640
ARMOR_CONF_THRES = 0.15
BALL_CONF_THRES = 0.25
ARMOR_ID = 0
BALL_ID = 1

# 可视化配置
VISUALIZE_ALL_DETECTIONS = False  # 关闭所有原始检测显示
VISUALIZE_LOW_CONF = 0.05
VISUALIZE_KALMAN = False  # 关闭卡尔曼预测显示
VISUALIZE_TRAJECTORY = True  # 保留轨迹显示

# 轨迹筛选配置
MAX_TRAJECTORY_ANGLE = 15  # 最大倾斜角度（度）
MAX_HORIZONTAL_MOVE = 20  # 最大横向移动像素
STAGNANT_FRAMES = 3  # 停滞帧数阈值
STAGNANT_RADIUS = 30  # 停滞半径（像素）
MIN_VERTICAL_SPEED = 2  # 最小垂直速度（像素/帧）
MAX_BALL_ARMOR_AREA_RATIO = 0.5  # ball面积不能超过armor面积的一半

VERT_DEVIATION_THRESH = 10
VERT_TOL_FRAMES = 3
HORIZ_DISCONNECT_THRESH = 50
PRED_TOLERANCE = 30
PRED_X_TOLERANCE = 40

# 跟踪器配置
TRACKER_TYPE = "CSRT"  # CSRT比KCF更精确，但速度稍慢
TRACKER_MAX_MISS = 5  # 跟踪器最大丢失帧数
TRACKER_CONFIDENCE_THRES = 0.5  # 跟踪置信度阈值
HYBRID_FUSION_WEIGHT = 0.7  # 混合跟踪权重（检测:0.7, 跟踪:0.3）

# ---------------- 图像增强 ----------------
def apply_simple_contrast(img):
    alpha = 1.2
    beta = 10
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

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
        img_resized, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT, value=color
    )
    return img_padded, scale, left, top

def optimize_armor_box(frame, mask_color_lower=(100, 150, 0), mask_color_upper=(130, 255, 255)):
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
            optimized_boxes.append(np.int32(box))
    return optimized_boxes, mask

def create_ca_ball_kf():
    kf = KalmanFilter(dim_x=6, dim_z=2)
    dt = 1.0
    kf.F = np.array([
        [1, 0, dt, 0, 0.5 * dt * dt, 0],
        [0, 1, 0, dt, 0, 0.5 * dt * dt],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ], dtype=float)
    kf.H = np.zeros((2, 6))
    kf.H[0, 0] = 1
    kf.H[1, 1] = 1
    kf.P *= 100.0
    kf.R = np.eye(2) * 5.0
    kf.Q = np.eye(6) * 0.01
    return kf

def calculate_confidence(yolo_conf, kalman_conf, alpha=0.6):
    return alpha * yolo_conf + (1 - alpha) * kalman_conf

def non_max_suppression(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
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
    for i, item in enumerate(texts):
        if isinstance(item, tuple):
            text, color = item
        else:
            text, color = item, (0, 255, 0)
        yi = y + i * line_height
        cv2.putText(frame, text, (x, yi), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

def is_trajectory_vertical(trajectory, max_angle_deg=15, min_points=3, min_displacement=5):
    if len(trajectory) < min_points:
        return True
    x0, y0, _ = trajectory[0]
    x1, y1, _ = trajectory[-1]
    dx = x1 - x0
    dy = y1 - y0
    if abs(dy) < min_displacement and abs(dx) < min_displacement:
        return True
    angle_rad = np.arctan2(dx, dy)
    angle_deg = abs(np.degrees(angle_rad))
    return angle_deg <= max_angle_deg

def is_trajectory_downward(trajectory, min_points=3):
    if len(trajectory) < min_points:
        return True
    ys = [p[1] for p in trajectory]
    dy_total = ys[-1] - ys[0]
    if dy_total <= 0:
        return False
    deltas = [ys[i+1] - ys[i] for i in range(len(ys)-1)]
    down_count = sum(1 for d in deltas if d >= 0)
    return down_count >= (len(deltas) / 2)

def check_trajectory_horizontal_movement(trajectory, max_horizontal_move=20):
    """检查轨迹是否有显著横向移动"""
    if len(trajectory) < 2:
        return True  # 太短，暂时通过
    
    xs = [p[0] for p in trajectory]
    
    # 检查任意两个连续点之间的横向移动
    for i in range(1, len(xs)):
        dx = abs(xs[i] - xs[i-1])
        if dx > max_horizontal_move:
            return False  # 横向移动过大
    
    # 检查总体横向偏移
    total_horizontal_drift = abs(xs[-1] - xs[0])
    if total_horizontal_drift > max_horizontal_move * 2:
        return False
    
    return True

def check_trajectory_stagnation(trajectory, stagnant_frames=3, stagnant_radius=30):
    """检查轨迹是否在相近位置停滞"""
    if len(trajectory) < stagnant_frames:
        return True  # 太短，暂时通过
    
    # 检查最近N帧是否在同一位置
    recent_points = trajectory[-stagnant_frames:]
    
    # 计算这些点的中心
    center_x = sum(p[0] for p in recent_points) / len(recent_points)
    center_y = sum(p[1] for p in recent_points) / len(recent_points)
    
    # 检查所有点是否都在半径范围内
    all_within_radius = all(
        np.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) <= stagnant_radius
        for p in recent_points
    )
    
    if all_within_radius:
        # 检查是否有足够的垂直移动
        ys = [p[1] for p in recent_points]
        vertical_movement = max(ys) - min(ys)
        if vertical_movement < MIN_VERTICAL_SPEED * (stagnant_frames - 1):
            return False  # 停滞了
    
    return True

def check_trajectory_angle_consistency(trajectory, max_angle_deg=15, check_window=5):
    """检查轨迹角度的连续性，防止突然转向"""
    if len(trajectory) < 2:
        return True
    
    # 检查每个小段的角度
    for i in range(1, len(trajectory)):
        x0, y0, _ = trajectory[max(0, i - check_window)]
        x1, y1, _ = trajectory[i]
        dx = x1 - x0
        dy = y1 - y0
        
        if abs(dy) >= 5:  # 只在有明显垂直移动时检查
            angle_rad = np.arctan2(dx, dy)
            angle_deg = abs(np.degrees(angle_rad))
            if angle_deg > max_angle_deg:
                return False
    
    return True

def validate_ball_trajectory(trajectory, new_point=None):
    """
    综合验证ball轨迹的有效性
    返回: (is_valid, reason)
    """
    if new_point is not None:
        test_trajectory = trajectory + [new_point]
    else:
        test_trajectory = trajectory
    
    if len(test_trajectory) < 2:
        return True, "Too short"
    
    # 1. 检查是否垂直
    if not is_trajectory_vertical(test_trajectory, max_angle_deg=MAX_TRAJECTORY_ANGLE):
        return False, "Not vertical"
    
    # 2. 检查是否向下
    if not is_trajectory_downward(test_trajectory):
        return False, "Not downward"
    
    # 3. 检查横向移动
    if not check_trajectory_horizontal_movement(test_trajectory, max_horizontal_move=MAX_HORIZONTAL_MOVE):
        return False, "Horizontal movement"
    
    # 4. 检查是否停滞
    if not check_trajectory_stagnation(test_trajectory, stagnant_frames=STAGNANT_FRAMES, stagnant_radius=STAGNANT_RADIUS):
        return False, "Stagnation detected"
    
    # 5. 检查角度连续性
    if not check_trajectory_angle_consistency(test_trajectory, max_angle_deg=MAX_TRAJECTORY_ANGLE):
        return False, "Angle inconsistency"
    
    return True, "Valid"

# ---------------- 增强的图像预处理 ----------------
def preprocess_for_tracking(frame, bbox):
    """为跟踪器准备ROI图像"""
    x, y, w, h = bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    roi = frame[y1:y2, x1:x2].copy()
    
    # 增强对比度，使目标更明显
    roi = cv2.convertScaleAbs(roi, alpha=1.3, beta=10)
    
    # 应用CLAHE增强局部对比度
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return roi

# ---------------- 混合跟踪器类 ----------------
class HybridBallTracker:
    def __init__(self, tracker_type="CSRT"):
        self.tracker = None
        self.tracker_type = tracker_type
        self.is_initialized = False
        self.miss_count = 0
        self.last_valid_bbox = None
        self.last_track_success = False
        self.tracker_confidence = 0.0
        
        # 用于评估跟踪质量的指标
        self.velocity_history = []
        self.size_history = []
        self.position_history = []
        
    def initialize(self, frame, bbox):
        """初始化跟踪器"""
        try:
            if self.tracker_type == "CSRT":
                self.tracker = cv2.TrackerCSRT_create()
            elif self.tracker_type == "KCF":
                self.tracker = cv2.TrackerKCF_create()
            elif self.tracker_type == "MOSSE":
                self.tracker = cv2.TrackerMOSSE_create()
            else:
                self.tracker = cv2.TrackerCSRT_create()
            
            success = self.tracker.init(frame, bbox)
            self.is_initialized = success
            self.miss_count = 0
            self.last_valid_bbox = bbox
            self.tracker_confidence = 1.0
            
            # 初始化历史记录
            x, y, w, h = bbox
            self.velocity_history = []
            self.size_history = [(w, h)]
            self.position_history = [(x + w/2, y + h/2)]
            
            return success
        except Exception as e:
            print(f"[WARNING] Tracker initialization failed: {e}")
            self.is_initialized = False
            return False
    
    def update(self, frame):
        """更新跟踪器"""
        if not self.is_initialized or self.tracker is None:
            return False, None, 0.0
        
        try:
            success, bbox = self.tracker.update(frame)
            
            if success:
                # 计算跟踪置信度（基于历史一致性）
                confidence = self.calculate_tracker_confidence(bbox)
                self.tracker_confidence = confidence
                
                # 检查跟踪质量
                if confidence < TRACKER_CONFIDENCE_THRES:
                    success = False
                    self.miss_count += 1
                else:
                    self.last_valid_bbox = bbox
                    self.miss_count = 0
                    self.last_track_success = True
                    
                    # 更新历史记录
                    x, y, w, h = bbox
                    self.size_history.append((w, h))
                    if len(self.size_history) > 10:
                        self.size_history.pop(0)
                    
                    center = (x + w/2, y + h/2)
                    self.position_history.append(center)
                    if len(self.position_history) > 10:
                        self.position_history.pop(0)
                    
                    if len(self.position_history) > 1:
                        last_center = self.position_history[-2]
                        vx = center[0] - last_center[0]
                        vy = center[1] - last_center[1]
                        self.velocity_history.append((vx, vy))
                        if len(self.velocity_history) > 5:
                            self.velocity_history.pop(0)
            else:
                self.miss_count += 1
                self.tracker_confidence = max(0.0, self.tracker_confidence - 0.2)
            
            return success, bbox, self.tracker_confidence if success else 0.0
            
        except Exception as e:
            print(f"[WARNING] Tracker update failed: {e}")
            self.miss_count += 1
            return False, None, 0.0
    
    def calculate_tracker_confidence(self, bbox):
        """计算跟踪置信度，基于运动一致性和尺寸稳定性"""
        x, y, w, h = bbox
        confidence = 1.0
        
        # 1. 检查bbox合理性
        if w <= 0 or h <= 0 or w > 200 or h > 200:
            return 0.0
        
        # 2. 检查尺寸稳定性
        if len(self.size_history) > 0:
            avg_w = np.mean([s[0] for s in self.size_history])
            avg_h = np.mean([s[1] for s in self.size_history])
            size_change = abs(w - avg_w)/avg_w + abs(h - avg_h)/avg_h
            confidence *= max(0, 1.0 - size_change)
        
        # 3. 检查运动一致性
        if len(self.velocity_history) >= 2:
            avg_vx = np.mean([v[0] for v in self.velocity_history])
            avg_vy = np.mean([v[1] for v in self.velocity_history])
            std_vx = np.std([v[0] for v in self.velocity_history])
            std_vy = np.std([v[1] for v in self.velocity_history])
            
            # 高运动方差可能表示跟踪不稳定
            motion_variance = (std_vx + std_vy) / max(1.0, abs(avg_vx) + abs(avg_vy))
            confidence *= max(0, 1.0 - motion_variance * 0.5)
        
        return max(0.0, min(1.0, confidence))
    
    def reset(self):
        """重置跟踪器"""
        self.tracker = None
        self.is_initialized = False
        self.miss_count = 0
        self.last_valid_bbox = None
        self.tracker_confidence = 0.0
        self.velocity_history.clear()
        self.size_history.clear()
        self.position_history.clear()
    
    def is_reliable(self):
        """检查跟踪器是否可靠"""
        return (self.is_initialized and 
                self.miss_count < TRACKER_MAX_MISS and 
                self.tracker_confidence > TRACKER_CONFIDENCE_THRES)

# ---------------- 主程序 ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'gpu', 'CPU', 'GPU'], default=os.getenv('OV_DEVICE', 'CPU'))
    parser.add_argument('--video', default=VIDEO_PATH)
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--tracker', default=TRACKER_TYPE, choices=['CSRT', 'KCF', 'MOSSE'])
    args = parser.parse_args()

    device_req = args.device.upper()
    device_name = 'GPU' if device_req == 'GPU' else 'CPU'
    tracker_type = args.tracker

    ie = Core()
    try:
        compiled_model = ie.compile_model(args.model, device_name)
        print(f"[INFO] Compiled model on device: {device_name}")
    except Exception as e:
        print(f"[ERROR] Failed to compile on {device_name}: {e}. Exiting.")
        sys.exit(1)

    output_layer = compiled_model.output(0)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("[ERROR] 无法打开视频源")
        return

    # 初始化混合跟踪器
    hybrid_tracker = HybridBallTracker(tracker_type=tracker_type)
    
    # 运行时状态
    last_diff = None
    ball_kf = None
    frame_idx = 0
    class_counts = {ARMOR_ID: 0, BALL_ID: 0}
    frame_times = []

    ball_was_below = False
    frames_no_ball = 0
    back_reported = False
    last_horizontal = None
    last_position = None

    ball_trajectory = []
    prev_trajectory = []
    last_disappearance_point = None
    disappearance_handled = False

    DISAPPEAR_FRAMES = 3
    BOOST_RADIUS = 40
    BOOST_FACTOR = 1.3

    # 新增：存储相对于armor的轨迹
    ball_relative_trajectory = []
    prev_relative_trajectory = []
    last_armor_center = None
    last_relative_disappearance_point = None

    # 纵向容忍计数：允许短时间小幅向上波动
    vertical_tolerant_count = 0
    
    # 跟踪状态
    track_bbox = None
    track_center = None
    using_tracker = False
    tracker_miss_frames = 0
    hybrid_confidence = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        h0, w0 = frame.shape[:2]

        # 预测中心
        predicted_center = None
        if ball_kf is not None:
            try:
                ball_kf.predict()
                predicted_center = (float(ball_kf.x[0]), float(ball_kf.x[1]))
            except Exception:
                predicted_center = None

        # 图像预处理
        enhanced_frame = apply_simple_contrast(frame)
        
        # 如果跟踪器正在工作，先进行跟踪更新
        track_detection = None
        if hybrid_tracker.is_initialized:
            track_success, track_bbox, track_conf = hybrid_tracker.update(frame)
            
            if track_success and track_conf > TRACKER_CONFIDENCE_THRES:
                # 转换跟踪框格式为检测格式
                x, y, w, h = track_bbox
                track_center = (x + w/2, y + h/2)
                track_detection = {
                    'bbox': (x, y, x+w, y+h),
                    'center': track_center,
                    'confidence': track_conf,
                    'source': 'tracker'
                }
                using_tracker = True
                tracker_miss_frames = 0
            else:
                tracker_miss_frames += 1
                using_tracker = False
                if tracker_miss_frames > TRACKER_MAX_MISS:
                    hybrid_tracker.reset()
        
        # YOLO检测
        img_lb, scale, pad_x, pad_y = letterbox(enhanced_frame, (INPUT_SIZE, INPUT_SIZE))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))
        input_tensor = np.expand_dims(img_chw, axis=0)

        output = compiled_model([input_tensor])[output_layer]
        pred = output[0]

        # 解析检测结果 - 保存所有检测用于可视化
        all_raw_detections = {ARMOR_ID: [], BALL_ID: []}
        detections = {ARMOR_ID: [], BALL_ID: []}
        
        for i in range(pred.shape[1]):
            cx, cy, w, h = pred[0:4, i]
            cls_scores = pred[4:, i]
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])
            
            if cls_id not in (ARMOR_ID, BALL_ID):
                continue
            
            # 计算图像坐标
            cx_img = (float(cx) - pad_x) / scale
            cy_img = (float(cy) - pad_y) / scale
            bw_img = float(w) / scale
            bh_img = float(h) / scale
            x1 = int(cx_img - bw_img / 2)
            y1 = int(cy_img - bh_img / 2)
            x2 = int(cx_img + bw_img / 2)
            y2 = int(cy_img + bh_img / 2)
            x1 = max(0, min(w0 - 1, x1))
            y1 = max(0, min(h0 - 1, y1))
            x2 = max(0, min(w0 - 1, x2))
            y2 = max(0, min(h0 - 1, y2))
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            
            # 保存所有原始检测（包括低置信度）
            if conf >= VISUALIZE_LOW_CONF:
                all_raw_detections[cls_id].append((x1, y1, x2, y2, conf, center_x, center_y))
            
            # 应用置信度阈值
            if (cls_id == ARMOR_ID and conf < ARMOR_CONF_THRES) or (cls_id == BALL_ID and conf < BALL_CONF_THRES):
                continue
            
            # 置信度增强
            original_conf = conf
            if predicted_center is not None:
                dx = abs(center_x - predicted_center[0])
                dy = abs(center_y - predicted_center[1])
                if dx <= BOOST_RADIUS and dy <= BOOST_RADIUS:
                    conf = min(1.0, conf * BOOST_FACTOR)
            
            detections[cls_id].append((x1, y1, x2, y2, conf, center_x, center_y, original_conf))

        # ============ 可视化所有原始检测 ============
        # 已关闭，不再显示低置信度检测

        # ============ 可视化卡尔曼预测 ============
        # 已关闭，不再显示卡尔曼预测

        # NMS过滤
        selected = {ARMOR_ID: [], BALL_ID: []}
        for cid in (ARMOR_ID, BALL_ID):
            boxes = [(d[0], d[1], d[2], d[3]) for d in detections[cid]]
            scores = [d[4] for d in detections[cid]]
            if boxes:
                keep_idx = non_max_suppression(boxes, scores, iou_threshold=0.4)
                for idx in keep_idx:
                    selected[cid].append(detections[cid][idx])

        # ============ 可视化NMS后的检测 ============
        # 不再显示置信度增强标注

        # 选择最佳检测
        best_armor = max(selected[ARMOR_ID], key=lambda x: x[4]) if selected[ARMOR_ID] else None
        yolo_ball = max(selected[BALL_ID], key=lambda x: x[4]) if selected[BALL_ID] else None
        
        # 轨迹过滤 - 使用增强的验证逻辑
        filtered_yolo_ball = None
        trajectory_filter_info = ""
        trajectory_filter_reason = ""
        
        if yolo_ball is not None:
            bx1, by1, bx2, by2, bconf, bcx, bcy, orig_conf = yolo_ball
            
            # 创建新的测试点
            new_point = (float(bcx), float(bcy), frame_idx + 1)
            
            # 综合验证轨迹
            is_valid, reason = validate_ball_trajectory(ball_trajectory, new_point)
            trajectory_filter_reason = reason
            
            # 额外检查1：与armor的水平距离
            horiz_check = True
            horiz_dist = 0
            if best_armor is not None:
                if len(best_armor) == 8:
                    _, _, _, _, _, acx, _, _ = best_armor
                else:
                    _, _, _, _, _, acx, _ = best_armor
                horiz_dist = abs(float(bcx) - float(acx))
                horiz_check = horiz_dist <= 500
            
            # 额外检查2：ball面积不能超过armor面积的一半
            area_check = True
            ball_area = (bx2 - bx1) * (by2 - by1)
            armor_area = 0
            
            if best_armor is not None:
                if len(best_armor) == 8:
                    ax1, ay1, ax2, ay2, _, _, _, _ = best_armor
                else:
                    ax1, ay1, ax2, ay2, _, _, _ = best_armor
                armor_area = (ax2 - ax1) * (ay2 - ay1)
                
                if armor_area > 0:
                    area_ratio = ball_area / armor_area
                    area_check = area_ratio <= MAX_BALL_ARMOR_AREA_RATIO
                    
                    # 如果面积检查失败，记录原因
                    if not area_check:
                        trajectory_filter_reason = f"Ball too large ({area_ratio:.2f}x armor)"
            
            # 组合所有检查结果
            final_valid = is_valid and horiz_check and area_check
            
            # 不再显示详细的过滤信息和拒绝原因
            
            if final_valid:
                filtered_yolo_ball = yolo_ball

        frame_idx += 1
        
        # 当前帧的armor中心点
        current_armor_center = None
        if best_armor is not None:
            # 修复：best_armor可能有7或8个元素
            if len(best_armor) == 8:
                _, _, _, _, _, acx, acy, _ = best_armor
            else:
                _, _, _, _, _, acx, acy = best_armor
            current_armor_center = (float(acx), float(acy))
            last_armor_center = current_armor_center
        
        # 融合检测和跟踪结果
        best_ball = None
        detection_source = "none"
        
        # 情况1: YOLO检测到ball
        if filtered_yolo_ball is not None:
            # filtered_yolo_ball有8个元素
            bx1, by1, bx2, by2, yolo_conf, bcx, bcy, orig_conf = filtered_yolo_ball
            
            # 情况1a: 同时有跟踪结果 -> 融合
            if track_detection is not None:
                tx1, ty1, tx2, ty2 = track_detection['bbox']
                tcx, tcy = track_detection['center']
                track_conf = track_detection['confidence']
                
                detection_center = np.array([float(bcx), float(bcy)])
                track_center_array = np.array([tcx, tcy])
                distance = np.linalg.norm(detection_center - track_center_array)
                
                # 可视化检测和跟踪的距离
                cv2.line(frame, (int(bcx), int(bcy)), (int(tcx), int(tcy)), (255, 255, 0), 1)
                mid_x, mid_y = (int(bcx) + int(tcx))//2, (int(bcy) + int(tcy))//2
                cv2.putText(frame, f"Dist:{distance:.1f}", (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                if distance < 100:
                    fusion_weight = HYBRID_FUSION_WEIGHT
                    fused_cx = fusion_weight * float(bcx) + (1 - fusion_weight) * tcx
                    fused_cy = fusion_weight * float(bcy) + (1 - fusion_weight) * tcy
                    fused_w = (bx2 - bx1) * fusion_weight + (tx2 - tx1) * (1 - fusion_weight)
                    fused_h = (by2 - by1) * fusion_weight + (ty2 - ty1) * (1 - fusion_weight)
                    fused_x1 = fused_cx - fused_w/2
                    fused_y1 = fused_cy - fused_h/2
                    
                    # best_ball只需要7个元素（不需要orig_conf）
                    best_ball = (int(fused_x1), int(fused_y1), 
                                int(fused_x1 + fused_w), int(fused_y1 + fused_h),
                                (yolo_conf + track_conf) / 2, fused_cx, fused_cy)
                    detection_source = "hybrid"
                    hybrid_confidence = (yolo_conf + track_conf) / 2
                    hybrid_tracker.initialize(frame, (int(fused_x1), int(fused_y1), int(fused_w), int(fused_h)))
                else:
                    # best_ball只需要7个元素
                    best_ball = (bx1, by1, bx2, by2, yolo_conf, bcx, bcy)
                    detection_source = "yolo"
                    hybrid_confidence = yolo_conf
                    hybrid_tracker.initialize(frame, (bx1, by1, bx2-bx1, by2-by1))
            else:
                # best_ball只需要7个元素
                best_ball = (bx1, by1, bx2, by2, yolo_conf, bcx, bcy)
                detection_source = "yolo"
                hybrid_confidence = yolo_conf
                hybrid_tracker.initialize(frame, (bx1, by1, bx2-bx1, by2-by1))
        
        # 情况2: YOLO未检测到ball，但有跟踪结果
        elif track_detection is not None:
            x1, y1, x2, y2 = track_detection['bbox']
            tcx, tcy = track_detection['center']
            track_conf = track_detection['confidence']
            
            temp_traj = ball_trajectory.copy()
            temp_traj.append((tcx, tcy, frame_idx))
            
            if (is_trajectory_vertical(temp_traj, max_angle_deg=15) and
                is_trajectory_downward(temp_traj)):
                best_ball = (int(x1), int(y1), int(x2), int(y2), track_conf, tcx, tcy)
                detection_source = "tracker"
                hybrid_confidence = track_conf * 0.8  # 跟踪结果置信度打折
        
        # ============ 可视化实际轨迹 ============
        # 不再显示绝对坐标系下的轨迹和轨迹点序号

        # 处理ball检测/跟踪结果
        if best_ball is not None:
            bx1, by1, bx2, by2, bconf, bcx, bcy = best_ball
            prev_trajectory.clear()
            prev_relative_trajectory.clear()
            last_disappearance_point = None
            last_relative_disappearance_point = None
            class_counts[BALL_ID] += 1
            
            # 绘制ball识别框（保留）
            if detection_source == "yolo":
                color = (0, 0, 255)  # 红色: YOLO检测
                label = f"Ball {bconf:.2f}"
            elif detection_source == "tracker":
                color = (255, 0, 0)  # 蓝色: 跟踪器
                label = f"Ball {bconf:.2f}"
            else:  # hybrid
                color = (0, 255, 255)  # 黄色: 融合结果
                label = f"Ball {bconf:.2f}"
            
            cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2)
            cv2.putText(frame, label, (int(bx1), max(0, int(by1) - 6)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 获取上一次轨迹点
            last_x = ball_trajectory[-1][0] if ball_trajectory else None
            last_y = ball_trajectory[-1][1] if ball_trajectory else None

            # 计算预测误差
            pred_err = None
            if predicted_center is not None:
                pred_err = np.linalg.norm(np.array([float(bcx) - predicted_center[0], 
                                                    float(bcy) - predicted_center[1]]))

            # 横向断开判定 - 使用更严格的阈值
            if last_x is not None and abs(float(bcx) - last_x) > MAX_HORIZONTAL_MOVE * 2:
                ball_kf = create_ca_ball_kf()
                ball_kf.x = np.array([float(bcx), float(bcy), 0.0, 0.0, 0.0, 0.0])
                ball_trajectory.clear()
                ball_relative_trajectory.clear()
                ball_trajectory.append((float(bcx), float(bcy), frame_idx))
                if current_armor_center is not None:
                    rel_x = float(bcx) - current_armor_center[0]
                    rel_y = float(bcy) - current_armor_center[1]
                    ball_relative_trajectory.append((rel_x, rel_y, frame_idx))
                vertical_tolerant_count = 0
            else:
                # 纵向检查
                dy = None
                if last_y is not None:
                    dy = float(bcy) - last_y
                vertical_stable = True
                if dy is not None:
                    if dy >= 0:
                        vertical_tolerant_count = 0
                        vertical_stable = True
                    else:
                        if abs(dy) <= VERT_DEVIATION_THRESH and vertical_tolerant_count < VERT_TOL_FRAMES:
                            vertical_tolerant_count += 1
                            vertical_stable = True
                        else:
                            vertical_stable = False

                # 若存在预测且预测与检测偏差很大，则认为是新目标
                if pred_err is not None and pred_err > PRED_TOLERANCE and abs(float(bcx) - predicted_center[0]) > PRED_X_TOLERANCE:
                    ball_kf = create_ca_ball_kf()
                    ball_kf.x = np.array([float(bcx), float(bcy), 0.0, 0.0, 0.0, 0.0])
                    ball_trajectory.clear()
                    ball_relative_trajectory.clear()
                    ball_trajectory.append((float(bcx), float(bcy), frame_idx))
                    if current_armor_center is not None:
                        rel_x = float(bcx) - current_armor_center[0]
                        rel_y = float(bcy) - current_armor_center[1]
                        ball_relative_trajectory.append((rel_x, rel_y, frame_idx))
                    vertical_tolerant_count = 0
                else:
                    # 使用卡尔曼更新
                    if ball_kf is None:
                        ball_kf = create_ca_ball_kf()
                        ball_kf.x = np.array([float(bcx), float(bcy), 0.0, 0.0, 0.0, 0.0])
                    else:
                        try:
                            prev_R = ball_kf.R.copy() if hasattr(ball_kf, 'R') else None
                            rx = 5.0
                            ry = 1.0 if vertical_stable else 5.0
                            ball_kf.R = np.diag([rx, ry])
                            ball_kf.update(np.array([float(bcx), float(bcy)]))
                            if prev_R is not None:
                                ball_kf.R = prev_R
                        except Exception:
                            pass

                    # 计算相对于armor的坐标并存储
                    if current_armor_center is not None:
                        rel_x = float(bcx) - current_armor_center[0]
                        rel_y = float(bcy) - current_armor_center[1]
                        ball_relative_trajectory.append((rel_x, rel_y, frame_idx))

                    # 轨迹追加 - 添加额外验证
                    if not ball_trajectory:
                        ball_trajectory.append((float(bcx), float(bcy), frame_idx))
                    else:
                        # 验证新点是否符合轨迹规则
                        new_point = (float(bcx), float(bcy), frame_idx)
                        is_valid, _ = validate_ball_trajectory(ball_trajectory, new_point)
                        
                        if is_valid:
                            ball_trajectory.append(new_point)
                        else:
                            # 新点不符合规则，重新开始轨迹
                            ball_trajectory.clear()
                            ball_trajectory.append(new_point)
                            ball_kf = create_ca_ball_kf()
                            ball_kf.x = np.array([float(bcx), float(bcy), 0.0, 0.0, 0.0, 0.0])

            # 限制轨迹长度
            if len(ball_trajectory) > 100:
                ball_trajectory.pop(0)
            if len(ball_relative_trajectory) > 100:
                ball_relative_trajectory.pop(0)
            
            disappearance_handled = False
            frames_no_ball = 0
            back_reported = False
        else:
            # 没有ball检测/跟踪结果
            if predicted_center is not None:
                pred_x, pred_y = predicted_center
                cv2.circle(frame, (int(pred_x), int(pred_y)), 4, (0, 0, 200), -1)
            frames_no_ball += 1
            
            # 如果跟踪器丢失太久，重置
            if tracker_miss_frames > TRACKER_MAX_MISS:
                hybrid_tracker.reset()

        # 统计
        if best_armor is not None:
            class_counts[ARMOR_ID] += 1

        if frame_idx % 200 == 0:
            print(f"[STAT] frame {frame_idx}: counts: armor={class_counts.get(ARMOR_ID,0)}, ball={class_counts.get(BALL_ID,0)}")
            print(f"[TRACK] Tracker status: initialized={hybrid_tracker.is_initialized}, "
                  f"confidence={hybrid_tracker.tracker_confidence:.2f}, "
                  f"miss_count={hybrid_tracker.miss_count}")

        # 处理armor检测
        if best_armor is not None:
            # 修复：统一处理armor元组
            if len(best_armor) == 8:
                ax1, ay1, ax2, ay2, aconf, acx, acy, _ = best_armor
            else:
                ax1, ay1, ax2, ay2, aconf, acx, acy = best_armor
            
            # 绘制armor识别框（保留）
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 255, 0), 2)
            cv2.putText(frame, f"Armor {aconf:.2f}", (ax1, max(0, ay1 - 8)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 检查ball是否消失 - 新逻辑：基于消失点判断Position
            if frames_no_ball >= DISAPPEAR_FRAMES and (not disappearance_handled) and len(ball_trajectory) > 0:
                last_x, last_y, _ = ball_trajectory[-1]
                
                # 新逻辑：根据消失点相对于armor下边缘的位置判断
                # 消失点在下边缘以上 -> Back
                # 消失点在下边缘以下 -> Front
                # 只有这两种情况才有效，否则为miss
                if last_y < ay2:
                    # 消失点在armor下边缘以上
                    last_position = 'Back'
                elif last_y >= ay2:
                    # 消失点在armor下边缘以下（包括下边缘）
                    last_position = 'Front'
                else:
                    # 这个else实际上不会被触发，但保留以防万一
                    last_position = 'Miss'
                
                prev_trajectory = ball_trajectory.copy()
                prev_relative_trajectory = ball_relative_trajectory.copy()
                last_disappearance_point = (last_x, last_y)
                
                # 计算相对消失点
                if last_armor_center is not None:
                    last_relative_disappearance_point = (last_x - last_armor_center[0], 
                                                         last_y - last_armor_center[1])
                
                disappearance_handled = True
                ball_trajectory.clear()
                ball_relative_trajectory.clear()

            # 状态信息显示（保留左上角文字）
            status_lines = []
            
            # 添加跟踪源信息
            if detection_source != "none":
                source_color = (0, 255, 0) if detection_source == "yolo" else (255, 255, 0) if detection_source == "hybrid" else (0, 200, 255)
                status_lines.append((f"Source: {detection_source}", source_color))
                status_lines.append((f"Conf: {hybrid_confidence:.2f}", source_color))
            
            # 添加跟踪器状态
            tracker_status = "Active" if hybrid_tracker.is_initialized else "Inactive"
            tracker_color = (0, 255, 0) if hybrid_tracker.is_initialized else (0, 0, 255)
            status_lines.append((f"Tracker: {tracker_status} (C:{hybrid_tracker.tracker_confidence:.2f})", tracker_color))
            
            if best_ball is not None:
                _, _, _, _, bconf, bcx, bcy = best_ball
                ball_center = (bcx, bcy)
            elif ball_kf is not None:
                ball_center = (float(ball_kf.x[0]), float(ball_kf.x[1]))
            else:
                ball_center = None

            if ball_center is not None:
                armor_center = (acx, acy)
                horizontal_diff = ball_center[0] - armor_center[0]
                if best_ball is not None:
                    last_diff = horizontal_diff
                if last_diff is not None:
                    status_lines.append((f"Diff: {last_diff:.2f}", (0, 255, 0)))
                else:
                    status_lines.append(("Diff: N/A", (0, 255, 0)))
                
                # Horizontal判断保持不变
                if best_ball is not None:
                    bx = ball_center[0]
                    last_horizontal = 'true' if (ax1 <= bx <= ax2) else 'Fault'
                
                # Position判断：实时显示ball位置，但不影响最终判断
                # 最终判断只在消失时才设定
                if best_ball is not None:
                    # 实时显示当前位置（仅供参考）
                    if bcy < ay2:
                        current_position_display = 'Above Bottom'
                    else:
                        current_position_display = 'Below Bottom'
            else:
                if last_diff is not None:
                    status_lines.append((f"Diff: {last_diff:.2f}", (0, 255, 0)))
                else:
                    status_lines.append(("No Ball detected", (0, 255, 0)))

            if last_horizontal is not None:
                color_h = (0, 255, 0) if last_horizontal == 'true' else (0, 0, 255)
                status_lines.append((f"Horizontal: {last_horizontal}", color_h))
            
            # Position显示：只显示有效判断结果
            if last_position is not None:
                if last_position in ['Front', 'Back']:
                    color_p = (0, 255, 0) if last_position == 'Front' else (0, 0, 255)
                else:
                    # Miss情况用灰色
                    color_p = (128, 128, 128)
                status_lines.append((f"Position: {last_position}", color_p))
            else:
                status_lines.append(("Position: Waiting", (200, 200, 200)))

            # 绘制左上角状态文字（保留）
            draw_status_texts(frame, status_lines, x=10, y=30, line_height=20, font_scale=0.6)
            
            # 绘制armor坐标系下的轨迹（保留）
            if current_armor_center is not None:
                # 绘制历史相对轨迹（灰色）
                if prev_relative_trajectory and len(prev_relative_trajectory) > 1:
                    for i in range(1, len(prev_relative_trajectory)):
                        rel_x1, rel_y1, _ = prev_relative_trajectory[i - 1]
                        rel_x2, rel_y2, _ = prev_relative_trajectory[i]
                        
                        # 将相对坐标转换为绝对坐标
                        abs_x1 = current_armor_center[0] + rel_x1
                        abs_y1 = current_armor_center[1] + rel_y1
                        abs_x2 = current_armor_center[0] + rel_x2
                        abs_y2 = current_armor_center[1] + rel_y2
                        
                        cv2.line(frame, (int(abs_x1), int(abs_y1)), (int(abs_x2), int(abs_y2)), (160, 160, 160), 2)
                    
                    # 绘制历史消失点（红色圆圈）
                    if last_relative_disappearance_point is not None:
                        rel_dx, rel_dy = last_relative_disappearance_point
                        abs_dx = current_armor_center[0] + rel_dx
                        abs_dy = current_armor_center[1] + rel_dy
                        
                        cv2.circle(frame, (int(abs_dx), int(abs_dy)), 6, (0, 0, 255), -1)
                        cv2.putText(frame, "Disappear", (int(abs_dx) + 8, int(abs_dy) - 8), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # 绘制当前相对轨迹（黄色）
                if len(ball_relative_trajectory) > 1:
                    for i in range(1, len(ball_relative_trajectory)):
                        rel_x1, rel_y1, _ = ball_relative_trajectory[i - 1]
                        rel_x2, rel_y2, _ = ball_relative_trajectory[i]
                        
                        # 将相对坐标转换为绝对坐标
                        abs_x1 = current_armor_center[0] + rel_x1
                        abs_y1 = current_armor_center[1] + rel_y1
                        abs_x2 = current_armor_center[0] + rel_x2
                        abs_y2 = current_armor_center[1] + rel_y2
                        
                        cv2.line(frame, (int(abs_x1), int(abs_y1)), (int(abs_x2), int(abs_y2)), (0, 255, 255), 2)
                    
                    # 绘制当前轨迹的最新点（绿色圆点）
                    if ball_relative_trajectory:
                        rel_lx, rel_ly, _ = ball_relative_trajectory[-1]
                        abs_lx = current_armor_center[0] + rel_lx
                        abs_ly = current_armor_center[1] + rel_ly
                        cv2.circle(frame, (int(abs_lx), int(abs_ly)), 4, (0, 255, 0), -1)

        # 显示帧率（保留）
        fps_text = f"FPS: {1/(time.time()-t0+1e-6):.1f}"
        cv2.putText(frame, fps_text, (w0 - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Armor Judge - Detection", frame)

        t1 = time.time()
        frame_times.append(t1 - t0)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # 按r键重置跟踪器
            hybrid_tracker.reset()
            print("[INFO] Tracker reset")

    # 性能统计
    total_frames = len(frame_times)
    total_time = sum(frame_times) if total_frames > 0 else 0.0
    avg_fps = (total_frames / total_time) if total_time > 0 else 0.0
    avg_frame_ms = (total_time / total_frames * 1000.0) if total_frames > 0 else 0.0
    print(f"\n[PERF] Device={device_name}, Tracker={tracker_type}")
    print(f"       Total frames: {total_frames}")
    print(f"       Total time: {total_time:.3f}s")
    print(f"       Average FPS: {avg_fps:.2f}")
    print(f"       Average frame time: {avg_frame_ms:.2f}ms")
    print(f"       Ball detections: {class_counts.get(BALL_ID, 0)}")
    print(f"       Armor detections: {class_counts.get(ARMOR_ID, 0)}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()