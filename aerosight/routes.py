from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from .models import YOLOModel

router = APIRouter(prefix="/api")
model = YOLOModel()


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    接收上传的图像文件并进行预测
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model.predict(
        image,
        save=False,
        imgsz=1280,
        conf=0.3,
        iou=0.3,
        stream=False
    )
    boxes = model.process_results(results)

    return JSONResponse(content={"results": boxes})
