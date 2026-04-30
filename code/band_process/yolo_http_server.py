import argparse
import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import requests
from cv2_rolling_ball import subtract_background_rolling_ball
from ultralytics import YOLO

import utils

RESIZE_LENGTH = 2200
BASELINE_RESIZE_LENGTH = 1850


class BandDetector:
    def __init__(self, model_path: str, image_dirs: List[str] | None = None):
        self.model = YOLO(model_path, task="detect")
        self.lock = threading.Lock()
        self.image_dirs = [os.path.abspath(p) for p in (image_dirs or [])]

    @staticmethod
    def _decode_base64_image(image_base64: str) -> np.ndarray:
        if "," in image_base64 and image_base64.strip().startswith("data:"):
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("图片解码失败，请确认 image_base64 是否正确")
        return image

    @staticmethod
    def _decode_url_image(image_url: str, timeout_sec: float = 20.0) -> np.ndarray:
        response = requests.get(image_url, timeout=timeout_sec)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("图片解码失败，请确认 image_url 指向的是有效图片")
        return image

    def _decode_local_filename(self, filename: str) -> np.ndarray:
        if not self.image_dirs:
            raise ValueError("服务未配置 image_dirs，无法使用 filename 模式")
        if not filename or not isinstance(filename, str):
            raise ValueError("filename 不能为空")
        missing_candidates: List[str] = []
        for image_dir in self.image_dirs:
            # 仅允许在 image_dir 下读取，防止路径穿越。
            candidate = os.path.abspath(os.path.join(image_dir, filename))
            if os.path.commonpath([image_dir, candidate]) != image_dir:
                continue
            if not os.path.exists(candidate):
                missing_candidates.append(candidate)
                continue
            if not os.path.isfile(candidate):
                continue
            image = cv2.imdecode(np.fromfile(candidate, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"图片解码失败: {candidate}")
            return image

        raise ValueError(
            "图片不存在，请检查 filename 或可用目录。已搜索目录: "
            + " | ".join(self.image_dirs)
            + f"；filename={filename}"
        )

    @staticmethod
    def _normalize_boxes(result: np.ndarray, class_names: Dict[int, str]) -> List[Dict[str, Any]]:
        boxes: List[Dict[str, Any]] = []
        for box in result:
            boxes.append(
                {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "confidence": float(box[4]),
                    "class_id": int(box[5]),
                    "class_name": class_names.get(int(box[5]), str(int(box[5]))),
                }
            )
        return boxes

    def detect(
        self,
        image_base64: str | None = None,
        image_url: str | None = None,
        filename: str | None = None,
        conf_threshold: float = 0.3,
    ) -> Dict[str, Any]:
        if image_base64:
            original_image = self._decode_base64_image(image_base64)
        elif image_url:
            original_image = self._decode_url_image(image_url)
        elif filename:
            original_image = self._decode_local_filename(filename)
        else:
            raise ValueError("缺少图片输入，请提供 image_base64、image_url 或 filename")
        original_gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

        image = utils.crop_rotate_image(original_gray_image)
        if image is None:
            raise ValueError("图片预处理失败，未能提取有效条带区域")

        image = cv2.resize(image, (RESIZE_LENGTH, image.shape[0]), interpolation=cv2.INTER_AREA)
        raw_gray = image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        with self.lock:
            yolo_out = self.model(image_rgb.copy(), imgsz=1280, verbose=False)[0]

        result = yolo_out.boxes.cpu().numpy().data
        class_names = yolo_out.names if isinstance(yolo_out.names, dict) else {0: "digital", 1: "black"}
        all_boxes = self._normalize_boxes(result, class_names)
        filtered_boxes = [b for b in all_boxes if b["confidence"] >= conf_threshold]

        digital_result, black_result = utils.get_object_detection_result(result)
        digital_flag, digital_position_result = utils.recognize_digital(
            digital_result, black_result, image_rgb.shape[1]
        )
        black_position_result, _ = utils.get_position_result(digital_flag, black_result, image_rgb)

        baseline_index = utils.get_imt_baseline_begin_position(digital_position_result, black_position_result)

        if baseline_index != 0:
            image_baseline_right = image_rgb[:, baseline_index:]
            rate = BASELINE_RESIZE_LENGTH / image_baseline_right.shape[1]
            black_position_result_scaled = [
                (
                    (pos[0] - baseline_index) * rate + baseline_index,
                    (pos[1] - baseline_index) * rate + baseline_index,
                    pos[2],
                )
                for pos in black_position_result
            ]
        else:
            black_position_result_scaled = black_position_result

        image_subtract, _ = subtract_background_rolling_ball(
            raw_gray,
            150,
            light_background=True,
            use_paraboloid=False,
            do_presmooth=True,
        )
        image_subtract_normalize = image_subtract.copy()
        cv2.normalize(image_subtract, image_subtract_normalize, 0, 255, cv2.NORM_MINMAX)
        image_subtract_normalize_average_gray = utils.get_y_average_gray(image_subtract_normalize)

        tip, judge_point_distance = utils.get_imt_tip_and_distance(
            digital_position_result,
            black_position_result_scaled,
            image_subtract_normalize_average_gray,
        )

        detected_points: List[Dict[str, Any]] = []
        for point_symbol, (name, point) in zip(tip, judge_point_distance.items()):
            if point_symbol == "+":
                detected_points.append(
                    {
                        "name": name,
                        "range": [int(point[0]), int(point[1])],
                    }
                )

        prompt_text = ""
        if detected_points:
            point_names = "、".join([p["name"] for p in detected_points])
            prompt_text = f"该检测条带上包含{point_names}等点位，请结合知识库判断条带类型。"

        return {
            "digital_flag": int(digital_flag),
            "digital_position": None
            if digital_position_result is None
            else [
                float(digital_position_result[0]),
                float(digital_position_result[1]),
                float(digital_position_result[2]),
            ],
            "baseline_index": int(baseline_index),
            "boxes": filtered_boxes,
            "boxes_raw_count": int(len(all_boxes)),
            "boxes_filtered_count": int(len(filtered_boxes)),
            "detected_points": detected_points,
            "prompt_text": prompt_text,
        }


class RequestHandler(BaseHTTPRequestHandler):
    detector: BandDetector = None

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send_json(200, {"ok": True, "service": "band-yolo-http"})
            return
        self._send_json(404, {"ok": False, "error": "Not Found"})

    def do_POST(self) -> None:
        if self.path != "/detect":
            self._send_json(404, {"ok": False, "error": "Not Found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ValueError("请求体不能为空")

            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))

            image_base64 = payload.get("image_base64")
            image_url = payload.get("image_url")
            filename = payload.get("filename")
            if isinstance(filename, str):
                filename = os.path.basename(filename.strip().strip('"').strip("'"))
            if not image_base64 and not image_url and not filename:
                raise ValueError("缺少字段 image_base64、image_url 或 filename")

            conf_threshold = float(payload.get("conf_threshold", 0.3))
            result = self.detector.detect(
                image_base64=image_base64,
                image_url=image_url,
                filename=filename,
                conf_threshold=conf_threshold,
            )
            self._send_json(200, {"ok": True, "data": result})
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})


def build_server(host: str, port: int, model_path: str, image_dirs: List[str]) -> ThreadingHTTPServer:
    RequestHandler.detector = BandDetector(model_path=model_path, image_dirs=image_dirs)
    return ThreadingHTTPServer((host, port), RequestHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Band YOLO HTTP Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--model", type=str, default=None, help="box_best.pt 路径")
    parser.add_argument(
        "--image-dirs",
        type=str,
        default=(
            "dataset/label_box_image_dataset_det_yolov8/val,"
            "dataset/label_box_image_dataset_det_yolov8/train,"
            "dataset-small/val,"
            "dataset-small/train"
        ),
        help="filename 模式下的图片目录，支持多个目录，用逗号分隔；相对路径按项目根目录解析",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if args.model:
        model_path = args.model
    else:
        model_path = os.path.join(project_root, "box_best.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    image_dirs: List[str] = []
    for raw in args.image_dirs.split(","):
        item = raw.strip()
        if not item:
            continue
        if os.path.isabs(item):
            dir_path = item
        else:
            dir_path = str(Path(project_root) / item)
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"image_dir 不存在或不是目录: {dir_path}")
        image_dirs.append(dir_path)
    if not image_dirs:
        raise ValueError("至少需要一个有效的 image_dir")

    server = build_server(args.host, args.port, model_path, image_dirs=image_dirs)
    print(f"HTTP service started at http://{args.host}:{args.port}")
    print("image_dirs = " + " | ".join([os.path.abspath(p) for p in image_dirs]))
    print("POST /detect, GET /healthz")
    server.serve_forever()


if __name__ == "__main__":
    main()
