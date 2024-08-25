# YOLOv8-to-EXE

基于tkinter开发的用于yolov8模型推理的exe可执行程序，无需python环境即可运行模型推理

## 介绍

本项目是一个小Demo，实现的功能有**选择模型**、**批量识别**、**单张识别**、**保存结果图片**、**保存检测框类别和坐标**、**将检测目标抠图出来**、**切换上一张下一张图片**，模型文件夹里有官方的`yolov8s.pt`模型文件，还有一个我自己基于yolov8s模型训练的`staff-zone.pt`（钢琴五线谱区域识别）模型文件，config文件夹里有配置文件，通过修改配置信息可以改变程序默认的输出文件夹位置以及需要检测的置信度大小等等，二次开发本项目请先配置好虚拟环境，导入“requirement.txt”使用指令：
```bash
pip install -r requirement.txt
```
