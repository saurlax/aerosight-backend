import cv2
import numpy as np
from PIL import Image
import io
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PanoramaStitcher:
    def __init__(self):
        """初始化全景图拼接器"""
        self.stitcher = cv2.Stitcher.create()

    def stitch_images(self, images: List[np.ndarray]) -> Tuple[bool, Optional[np.ndarray]]:
        """
        拼接图片组
        """
        if len(images) < 2:
            return False, None

        try:
            # 调整图片大小以提高拼接成功率
            resized_images = []
            for img in images:
                height, width = img.shape[:2]
                if width > 1000:  # 如果图片太大，进行缩放
                    scale = 1000 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    resized_img = cv2.resize(img, (new_width, new_height))
                    resized_images.append(resized_img)
                else:
                    resized_images.append(img)

            status, pano = self.stitcher.stitch(resized_images)

            if status == cv2.STITCHER_OK:
                return True, pano
            else:
                logger.warning(f"拼接失败，状态码：{status}")
                return False, None

        except Exception as e:
            logger.error(f"拼接过程中发生错误：{str(e)}")
            return False, None

    def create_single_panorama(self, pil_images: List[Image.Image]) -> Tuple[bool, Optional[bytes]]:
        """
        将所有图片拼接成一张全景图
        """
        if len(pil_images) < 2:
            return False, None

        # 转换PIL图片为OpenCV格式
        cv_images = []
        for pil_img in pil_images:
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cv_images.append(cv_img)

        # 拼接所有图片
        success, panorama = self.stitch_images(cv_images)

        if success and panorama is not None:
            # 转换回PIL格式并编码为字节
            pano_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
            pano_pil = Image.fromarray(pano_rgb)

            # 转换为字节流
            img_byte_arr = io.BytesIO()
            pano_pil.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)

            return True, img_byte_arr.getvalue()

        return False, None
