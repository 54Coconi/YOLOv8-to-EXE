# YOLOv8-to-EXE

基于tkinter开发的用于yolov8模型推理的exe可执行程序，无需python环境即可运行模型推理

## 介绍

本项目是一个小Demo，实现的功能有**选择模型**、**批量识别**、**单张识别**、**保存结果图片**、**保存检测框类别和坐标**、**将检测目标抠图出来**、**切换上一张下一张图片**，图片支持鼠标滚轮滑动放大缩小、鼠标移动图片，模型文件夹里有官方的`yolov8s.pt`模型文件，还有一个我自己基于yolov8s模型训练的`staff-zone.pt`（钢琴五线谱区域识别）模型文件，config文件夹里有配置文件，通过修改配置信息可以改变程序默认的输出文件夹位置以及需要检测的置信度大小等等，二次开发本项目请先配置好虚拟环境，导入“requirement.txt”使用指令：
```bash
pip install -r requirement.txt
```
配置好环境后，先运行`main.py`，如果程序正常启动并且各个功能正常，则可以运行：
```bash
pyinstaller YOLOv8.spec
```
`YOLOv8.spec`是用于pyinstaller来打包程序的脚本文件，如果不了解该文件，建议不要修改任何内容。还有一个文件是`clear.py`，该文件是用于清理打包后的程序在运行时产生的临时文件，一般是以**`_MEI+随机数字`**命名的文件夹（在windows下大小在**4.33G**左右），所以如果程序**异常终止**、**崩溃**或**直接关闭“黑窗口”** 则会导致临时文件残留问题，鉴于该临时文件比较大，个人建议每次运行结束后都运行一次清理程序，当然该清理程序也可以打包后使用，请运行：
```bash
pyinstaller clear.spec
```
打包后的程序比较大，大概**2.4G左右**，当然你可以改用其它方案，这里不做赘述

## 程序界面
主窗口
![主窗口](https://github.com/54Coconi/picture-repo/blob/main/img/yolov8%20to%20exe.png)
检测界面
![检测界面](https://github.com/54Coconi/picture-repo/blob/main/img/yolov8%20to%20exe1.png)
检测目标抠出图片
![检测目标抠出图片](https://github.com/54Coconi/picture-repo/blob/main/img/cut%20img.png)


