from ultralytics import YOLO
import os


class YOLOModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YOLOModel, cls).__new__(cls)
            cls._instance.model = YOLO(os.getenv("MODEL_PATH"))
        return cls._instance

    def predict(self, image, save=False, imgsz=1280, conf=0.3, iou=0.3, stream=False):
        results = self.model.predict(
            image,
            save=save,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            stream=stream
        )
        return results

    def process_results(self, results):
        """处理模型预测结果，将其转换为标准格式"""
        result = results[0]
        boxes = []
        for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                   result.boxes.conf.cpu().numpy(),
                                   result.boxes.cls.cpu().numpy()):
            boxes.append({
                "bbox": [float(coord) for coord in box],  # [x1, y1, x2, y2]
                "confidence": float(score),
                "class_id": int(cls),
                "class_name": self.model.names[int(cls)] if hasattr(self.model, "names") else str(int(cls))
            })
        return boxes
