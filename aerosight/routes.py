# routes.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from PIL import Image
import io
from typing import List
from .models import YOLOModel
from .panorama import PanoramaStitcher

router = APIRouter(prefix="/api")
model = YOLOModel()
panorama_stitcher = PanoramaStitcher()


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    接收上传的图像文件并进行预测
    """
    try:
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

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像预测失败: {str(e)}")


@router.post("/panorama")
async def create_panorama(files: List[UploadFile] = File(...)):
    """
    将上传的多张图片拼接成全景图
    """
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="至少需要上传2张图片进行拼接")

    try:
        # 读取所有上传的图片
        images = []
        for file in files:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            images.append(image)

        # 拼接全景图
        success, panorama_bytes = panorama_stitcher.create_single_panorama(images)

        if success and panorama_bytes:
            return Response(
                content=panorama_bytes,
                media_type="image/jpeg",
                headers={"Content-Disposition": "attachment; filename=panorama.jpg"}
            )
        else:
            return JSONResponse(
                content={"success": False, "message": "图片拼接失败，请检查上传的图片是否合适"},
                status_code=400
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"全景图拼接失败: {str(e)}")
