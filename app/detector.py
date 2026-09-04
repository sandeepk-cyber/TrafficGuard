"""
TrafficGuard - Deep Learning Object Detector (YOLO FP16 ONNX Engine)
High-recall detection with aspect-ratio preserving letterbox padding and zero emojis.
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
    """Resizes and pads image while preserving native aspect ratio."""
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
    def __init__(self, model_path=DEFAULT_MODEL_PATH, conf_threshold=0.25, nms_threshold=0.35):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.session = None
        self.input_name = None
        self.output_name = None
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
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info("YOLO ONNX session active (Input: %s, Output: %s, Conf: %.2f)",
                    self.input_name, self.output_name, self.conf_threshold)

    def detect(self, frame):
        """
        Runs object detection on frame using letterbox padding and high-recall threshold.
        Returns:
            detections: list of dicts (bbox, class_name, class_id, confidence, center)
            inference_ms: float
        """
        if self.session is None or frame is None:
            return [], 0.0

        h_orig, w_orig = frame.shape[:2]
        t_start = time.perf_counter()

        # 1. Aspect-ratio preserving letterbox
        img_padded, ratio, (dw, dh) = letterbox(frame, (640, 640))
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_fp16 = (img_rgb.astype(np.float16) / 255.0).transpose(2, 0, 1)
        input_tensor = np.expand_dims(img_fp16, axis=0)

        # 2. Inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        preds = outputs[0].astype(np.float32)  # Shape: (25200, 85)

        # 3. Filter predictions with calibrated high-recall threshold (0.18)
        boxes, scores, class_ids = [], [], []

        for pred in preds:
            obj_conf = pred[4]
            if obj_conf > self.conf_threshold:
                class_scores = pred[5:]
                cid = int(np.argmax(class_scores))
                score = float(obj_conf * class_scores[cid])
                if score > self.conf_threshold and cid in TARGET_CLASSES:
                    cx, cy, bw, bh = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
                    # Invert letterbox padding
                    x1 = (cx - bw / 2.0 - dw) / ratio
                    y1 = (cy - bh / 2.0 - dh) / ratio
                    w = bw / ratio
                    h = bh / ratio
                    boxes.append([int(x1), int(y1), int(w), int(h)])
                    scores.append(score)
                    class_ids.append(cid)

        # 4. NMS filtering
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold)
        raw_candidates = []
        if len(indices) > 0:
            for i in indices:
                idx = int(i)
                x, y, w, h = boxes[idx]
                x1 = max(0, min(w_orig - 1, x))
                y1 = max(0, min(h_orig - 1, y))
                x2 = max(0, min(w_orig, x + w))
                y2 = max(0, min(h_orig, y + h))
                bw = x2 - x1
                bh = y2 - y1
                cid = class_ids[idx]
                cname = TARGET_CLASSES.get(cid, "vehicle")

                # Filter out uncalibrated deep-horizon noise
                if cname == "person":
                    if bh < 20 or bw < 8:
                        continue
                else:
                    if bh < 14 or bw < 22 or (bw * bh) < 280:
                        continue

                raw_candidates.append({
                    "bbox": (x1, y1, x2, y2),
                    "class_name": cname,
                    "class_id": cid,
                    "confidence": round(scores[idx], 2),
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    "area": float(bw * bh)
                })

        # 5. Containment / Soft-IoM Suppression
        # Suppress duplicate sub-components (e.g. wheels/trunks) enclosed inside larger bounding boxes
        sorted_candidates = sorted(raw_candidates, key=lambda d: d["confidence"], reverse=True)
        suppressed_indices = set()
        for i in range(len(sorted_candidates)):
            if i in suppressed_indices:
                continue
            b1 = sorted_candidates[i]["bbox"]
            a1 = sorted_candidates[i]["area"]
            for j in range(i + 1, len(sorted_candidates)):
                if j in suppressed_indices:
                    continue
                b2 = sorted_candidates[j]["bbox"]
                a2 = sorted_candidates[j]["area"]

                inter_w = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                inter_h = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                inter = inter_w * inter_h

                if inter > 0.0 and min(a1, a2) > 0.0:
                    containment = inter / min(a1, a2)
                    if containment > 0.65:
                        suppressed_indices.add(j)

        detections = [d for idx, d in enumerate(sorted_candidates) if idx not in suppressed_indices]
        inference_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
        return detections, inference_ms
