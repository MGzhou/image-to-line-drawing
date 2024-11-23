#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/11/18 09:21:25
@Desc :None
'''

import cv2
import os
import time
import numpy as np
import onnxruntime as ort

class ImageToLine:
    def __init__(self, model_path):
        """初始化模型

        Args:
            model_path (str): 模型路径, 需要包括 line-drawings.onnx 与  line-relifer.onnx
        """
        self.line_drawings_path = os.path.join(model_path, "line-drawings.onnx")
        self.line_relifer_path = os.path.join(model_path, "line-relifer.onnx")
        if  not os.path.exists(self.line_drawings_path):
            raise ValueError(f"{self.line_drawings_path} not exists")
        if  not os.path.exists(self.line_relifer_path):
            raise ValueError(f"{self.line_relifer_path} not exists")

        # 加载模型
        self.line_drawings_session = ort.InferenceSession(self.line_drawings_path)
        self.line_relifer_session = ort.InferenceSession(self.line_relifer_path)

    def predict(self, image_path, save_path, enhance=True):
        """将彩色图片转换为线稿图

        Args:
            image_path (str): 图片路径
            save_path (str): 保存路径
            enhance (bool): 是否增强线稿图. Defaults to True.
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # 转换为RGB并归一化
        img = img.transpose(2, 0, 1).astype('float32')      # 转换为 (3, height, width) 格式
        feeds = {'input': np.expand_dims(img, axis=0)}
        # 运行推理
        output = self.line_drawings_session.run(None, feeds)[0]
        if enhance:
            output_image = np.transpose(output, (0, 2, 3, 1))
            output_image = output_image * 2 - 1       # 归一化到 [-1, 1]

            input_name = self.line_relifer_session.get_inputs()[0].name
            output_name = self.line_relifer_session.get_outputs()[0].name

            output = self.line_relifer_session.run([output_name], {input_name: output_image})[0]
            output_image = (output[0] + 1) / 2 * 255  # 将输出转换回 [0, 255]
            output_image = output_image.astype(np.uint8)
        else:
            output_image = output[0, 0]                             # 灰度图像，假设输出形状为 (1, 1, height, width)
            output_image = (output_image * 255).astype(np.uint8)    # 转换为 0-255 范围的 uint8 格式
        # 保存输出图像
        cv2.imwrite(save_path, output_image)


if __name__ == '__main__':
    start_time = time.time()
    # 输入图片
    path1 = r'data/remu.jpg'
    # 输出图片
    save1 = r'data/remu-line2.jpg'
    cts = ImageToLine(r'model')
    cts.predict(path1, save1)
    print(f"耗时 {int((time.time() - start_time))} 秒")

















