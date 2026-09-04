"""
TrafficGuard - Object Detection Pipeline (YOLO ONNX Engine)
Robust, lightweight detector supporting dynamic model input types (FP32/FP16),
class-aware NMS, aspect-ratio preserving letterboxing, and configurable thresholds.
Zero emojis, strict industrial coding standards.
"""
import os
import time
import urllib.request
import logging
import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("trafficguard.detector")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "yolov5n.onnx")
MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"

# Configurable detector defaults
DEFAULT_DETECTION_FPS = 10
DEFAULT_INPUT_SIZE = 640
DEFAULT_CONF_THRESHOLD = 0.20
DEFAULT_NMS_THRESHOLD = 0.40

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resizes and pads image while strictly preserving native aspect ratio."""
    shape = img.shape[:2]  # (height, width)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2.0, dh / 2.0

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


class YOLODetector:
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        conf_threshold=DEFAULT_CONF_THRESHOLD,
        nms_threshold=DEFAULT_NMS_THRESHOLD,
        input_size=DEFAULT_INPUT_SIZE,
        target_classes=None
    ):
        self.model_path = model_path
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.input_size = int(input_size)
        self.target_classes = target_classes or TARGET_CLASSES

        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_type = "tensor(float)"
        self.is_fp16 = False
        self.input_shape = [1, 3, self.input_size, self.input_size]

        self._ensure_model_exists()
        self._init_session()

    def _ensure_model_exists(self):
        if os.path.exists(self.model_path):
            return

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        logger.info("Downloading pre-trained YOLOv5n ONNX model weights...")
        try:
            req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(self.model_path, "wb") as f:
                f.write(resp.read())
            logger.info("Model weights successfully saved to %s", self.model_path)
        except Exception as e:
            logger.error("Failed to download YOLO model weights: %s", e)

    def _init_session(self):
        if not os.path.exists(self.model_path):
            logger.warning("YOLO model file not found at %s. Detector inactive.", self.model_path)
            return

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(4, os.cpu_count() or 2)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(self.model_path, opts, providers=["CPUExecutionProvider"])
        input_meta = self.session.get_inputs()[0]
        output_meta = self.session.get_outputs()[0]

        self.input_name = input_meta.name
        self.output_name = output_meta.name
        self.input_type = input_meta.type
        self.input_shape = input_meta.shape

        # Dynamically determine if the ONNX model expects float16 or float32
        self.is_fp16 = ("float16" in self.input_type.lower())

        logger.info(
            "YOLO ONNX initialized: input=%s (type=%s, shape=%s), output=%s, conf=%.2f, nms=%.2f",
            self.input_name, self.input_type, self.input_shape, self.output_name,
            self.conf_threshold, self.nms_threshold
        )

    def detect(self, frame):
        """
        Executes YOLO object detection with aspect-preserving letterboxing,
        model-consistent data type conversion, class-aware NMS, and boundary clamping.

        Returns:
            detections: list of dicts with keys:
                - bbox: tuple (x1, y1, x2, y2)
                - class_name: str
                - class_id: int
                - confidence: float
                - center: tuple (cx, cy)
                - area: float
            inference_ms: float latency in milliseconds
        """
        if self.session is None or frame is None:
            return [], 0.0

        h_orig, w_orig = frame.shape[:2]
        t_start = time.perf_counter()

        # 1. Aspect-ratio preserving letterbox
        img_padded, ratio, (dw, dh) = letterbox(frame, (self.input_size, self.input_size))
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)

        # 2. Match exact model input tensor dtype
        if self.is_fp16:
            img_norm = (img_rgb.astype(np.float16) / 255.0).transpose(2, 0, 1)
        else:
            img_norm = (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)

        input_tensor = np.expand_dims(img_norm, axis=0)

        # 3. ONNX inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        preds = outputs[0].astype(np.float32)  # Shape: (25200, 85)

        # 4. Filter predictions by objectness and class score
        boxes = []
        scores = []
        class_ids = []

        for pred in preds:
            obj_conf = pred[4]
            if obj_conf > self.conf_threshold:
                class_scores = pred[5:]
                cid = int(np.argmax(class_scores))
                score = float(obj_conf * class_scores[cid])
                if score > self.conf_threshold and cid in self.target_classes:
                    cx, cy, bw, bh = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
                    # Invert letterbox coordinates
                    x1 = (cx - bw / 2.0 - dw) / ratio
                    y1 = (cy - bh / 2.0 - dh) / ratio
                    w = bw / ratio
                    h = bh / ratio
                    boxes.append([int(x1), int(y1), int(w), int(h)])
                    scores.append(score)
                    class_ids.append(cid)

        if len(boxes) == 0:
            inference_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
            return [], inference_ms

        # 5. Category-Aware Non-Maximum Suppression (NMS)
        # Vehicles (car, motorcycle, bus, truck) form one NMS group to eliminate co-located
        # duplicate class predictions (e.g. car + bus on the same vehicle), while pedestrians
        # remain in an isolated group to prevent suppression in proximity.
        by_category = {}
        for idx, cid in enumerate(class_ids):
            cat = "person" if cid == 0 else "vehicle"
            by_category.setdefault(cat, []).append(idx)

        selected_indices = []
        for cat, group_indices in by_category.items():
            cls_boxes = [boxes[i] for i in group_indices]
            cls_scores = [scores[i] for i in group_indices]
            nms_res = cv2.dnn.NMSBoxes(cls_boxes, cls_scores, self.conf_threshold, self.nms_threshold)
            for k in nms_res:
                selected_indices.append(group_indices[int(k)])

        # Sub-box containment suppression for overlapping vehicle classes (IoM > 0.65)
        filtered_indices = []
        selected_indices.sort(key=lambda i: scores[i], reverse=True)
        for i in selected_indices:
            bi = boxes[i]
            ai = max(1, bi[2] * bi[3])
            ci = class_ids[i]
            suppress = False
            for prev in filtered_indices:
                cp = class_ids[prev]
                if ci != 0 and cp != 0:  # Both are vehicles
                    bp = boxes[prev]
                    ap = max(1, bp[2] * bp[3])
                    inter_w = max(0, min(bi[0] + bi[2], bp[0] + bp[2]) - max(bi[0], bp[0]))
                    inter_h = max(0, min(bi[1] + bi[3], bp[1] + bp[3]) - max(bi[1], bp[1]))
                    inter = inter_w * inter_h
                    if inter / min(ai, ap) > 0.65:
                        suppress = True
                        break
            if not suppress:
                filtered_indices.append(i)

        # 6. Assemble clamped detections
        detections = []
        for idx in filtered_indices:
            x, y, w, h = boxes[idx]
            x1 = max(0.0, min(float(w_orig), float(x)))
            y1 = max(0.0, min(float(h_orig), float(y)))
            x2 = max(0.0, min(float(w_orig), float(x + w)))
            y2 = max(0.0, min(float(h_orig), float(y + h)))

            bw_clamped = x2 - x1
            bh_clamped = y2 - y1

            # Reject microscopic noise and degraded artifacts
            if bw_clamped < 14.0 or bh_clamped < 10.0 or (bw_clamped * bh_clamped) < 180.0:
                continue

            cid = class_ids[idx]
            cname = self.target_classes.get(cid, "vehicle")
            conf = round(scores[idx], 3)
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            area = float(bw_clamped * bh_clamped)

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "class_name": cname,
                "class_id": cid,
                "confidence": conf,
                "center": center,
                "area": area
            })

        inference_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
        return detections, inference_ms
