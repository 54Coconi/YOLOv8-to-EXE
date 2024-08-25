import os
import shutil
import time
import json
import cv2

from ui import Win
from tkinter import filedialog, messagebox, NW

import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

# 定义默认配置
# MODEL_PATH = "./models"
# DEFAULT_MODEL = "./models/staff-zone.pt"
# DEFAULT_IMAGE_PATH = "./images"
#
# OUTPUT_PATH = "./output"
# CUT_IMAGE_PATH = "./cut_images"
#
# CONFIDENCE = 0.8

config = {}


class Controller:
    """
    控制器
    """
    ui: Win

    def __init__(self):
        self.model = None  # 模型
        self.image_paths = []  # 图片路径
        self.current_index = 0  # 当前图片索引
        self.detection_results = {}  # 检测结果字典
        self.categories = {}  # 类别字典
        self.image_on_canvas_left = None  # 保存左侧Canvas上的图像引用
        self.image_on_canvas_right = None  # 保存右侧Canvas上的图像引用
        self.scale_factor = 1.0  # 初始缩放比例
        self.image_origin_size = None  # 图片原始尺寸
        self.offset_x_left = 0  # 左画布图像的水平偏移
        self.offset_y_left = 0  # 左画布图像的垂直偏移
        self.offset_x_right = 0  # 右画布图像的水平偏移
        self.offset_y_right = 0  # 右画布图像的垂直偏移
        self.drag_data_left = {"x": 0, "y": 0}  # 左画布拖动数据
        self.drag_data_right = {"x": 0, "y": 0}  # 右画布拖动数据

    def init(self, ui):
        """
        初始化UI
        :param ui:
        """
        self.ui = ui
        # self.ui.tk_button_select_model
        # 绑定窗口大小变化事件到Canvas
        self.ui.tk_canvas_left_canvas.bind("<Configure>", self.on_canvas_resize)
        self.ui.tk_canvas_right_canvas.bind("<Configure>", self.on_canvas_resize)

        # 绑定鼠标滚轮事件
        self.ui.tk_canvas_left_canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.ui.tk_canvas_right_canvas.bind("<MouseWheel>", self.on_mouse_wheel)

        # 绑定鼠标拖动事件
        self.ui.tk_canvas_left_canvas.bind("<Button-1>", self.on_drag_start_left)
        self.ui.tk_canvas_left_canvas.bind("<B1-Motion>", self.on_drag_motion_left)
        self.ui.tk_canvas_left_canvas.bind("<ButtonRelease-1>", self.on_drag_stop_left)

        self.ui.tk_canvas_right_canvas.bind("<Button-1>", self.on_drag_start_right)
        self.ui.tk_canvas_right_canvas.bind("<B1-Motion>", self.on_drag_motion_right)
        self.ui.tk_canvas_right_canvas.bind("<ButtonRelease-1>", self.on_drag_stop_right)

        self.creat_default_config()  # 创建默认配置
        self.load_config()  # 加载配置
        self.del_output_cut()  # 清空输出目录(图片结果输出目录、目标抠出目录)
        self.load_model(config["default_model"])  # 加载默认模型

    @staticmethod
    def creat_default_config():
        """
        创建默认配置
        """
        config_path = "./config/config.json"
        if not os.path.exists(config_path):
            print("[INFO] - 配置文件不存在，创建默认配置")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            _config = {"model_path": "./models",
                       "default_model": "./models/staff-zone.pt",
                       "default_image_path": "./images",
                       "output_path": "./output/result_images",
                       "cut_image_path": "./output/cut_images",
                       "confidence": 0.8
                       }

            with open(config_path, 'w') as f:
                json.dump(_config, f, indent=4)

    @staticmethod
    def load_config():
        """
        加载配置文件
        """
        config_path = "./config/config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                global config
                config = json.load(f)
                print(f"[INFO] - 配置文件加载成功，配置信息为：{config}")
        else:
            print("[INFO] - 配置文件不存在，请检查")

    def load_model(self, model_path):
        """
        加载模型
        :param model_path: 模型路径
        """
        # model_path = self.ui.tk_input_model_path.get()
        if model_path:
            # Clear previous model
            self.model = None

            # Load categories from a file
            model_base_name = os.path.splitext(os.path.basename(model_path))[0]
            categories_file_path = os.path.join(os.path.dirname(model_path), model_base_name + ".txt")

            if not os.path.exists(categories_file_path):
                print("[WARNING] - 类别文件不存在。请确保模型和同名类别文件在同一目录下！")
                messagebox.showwarning("警告", "类别文件不存在。请确保模型和同名类别文件在同一目录下！")
                return

            # 显示模型路径
            self.ui.tk_input_model_path.delete(0, "end")
            self.ui.tk_input_model_path.insert(0, model_path)
            # 加载类别
            self.categories = self.load_categories(categories_file_path)

            start_load = time.time()
            # 加载模型
            self.model = YOLO(model_path)
            print(f"[INFO] - 模型 '{model_path}' 加载成功，耗时{time.time() - start_load}")

            # 清除之前的检测结果
            self.detection_results = {}
            # 清空图片路径和索引
            self.current_index = 0
            self.image_paths = []
            # 清空左侧画布的内容
            self.ui.tk_canvas_left_canvas.delete("all")
            # 清空右侧画布的内容
            self.ui.tk_canvas_right_canvas.delete("all")

    @staticmethod
    def load_categories(file_path):
        """
        从指定文件中加载类别信息。

        该方法通过读取一个包含类别的文件，来生成一个类别字典。文件中的每一行代表一个类别，
        行的索引作为键，类别名称作为值存入字典中。

        参数:
        file_path (str): 包含类别名称的文件路径。

        返回:
        dict: 一个字典，键为类别在文件中的行索引，值为类别名称。
        """
        # 初始化一个空字典，用于存储类别数据
        categories = {}

        # 打开文件，准备读取类别信息
        with open(file_path, 'r', encoding='utf-8') as file:
            # 遍历文件的每一行，从0开始为行索引
            for index, line in enumerate(file):
                # 去除行首行尾的空白字符，得到类别名称
                category_name = line.strip()
                # 将行索引和类别名称添加到字典中
                categories[index] = category_name
        print(f"[INFO] - 类别文件 '{file_path}' 加载成功")
        # 返回包含所有类别的字典
        return categories

    def check_model_loaded(self):
        """
        检查模型是否已加载
        :return: 布尔值，模型已加载返回 True，否则返回 False
        """
        if self.model is None:
            messagebox.showwarning("模型未加载", "请先加载模型！")
            return False
        return True

    def display_image(self, img, canvas, is_left=True):
        """
        在指定的画布上显示图像
        :param img:
        :param canvas:
        :param is_left:
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 获取Canvas的尺寸
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        # 获取图像的原始尺寸
        img_width, img_height = img.size
        if max(img_width, img_height) == img_height:
            img_height = canvas_height
            img_width = int(img_height * (img.width / img.height))
        else:
            img_width = canvas_width
            img_height = int(img_width * (img.height / img.width))

        # 计算缩放后的尺寸
        new_width = int(img_width * self.scale_factor)
        new_height = int(img_height * self.scale_factor)

        # 缩放图像
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img)

        # 计算图像显示的位置
        if is_left:
            x_offset = (canvas_width - new_width) // 2 + self.offset_x_left
            y_offset = (canvas_height - new_height) // 2 + self.offset_y_left
        else:
            x_offset = (canvas_width - new_width) // 2 + self.offset_x_right
            y_offset = (canvas_height - new_height) // 2 + self.offset_y_right

        # 清除之前的图像（如果有的话）
        if is_left:
            if self.image_on_canvas_left is not None:
                canvas.delete(self.image_on_canvas_left)
            self.image_on_canvas_left = canvas.create_image(x_offset, y_offset, anchor='nw', image=img_tk)
            canvas.image = img_tk  # 保持引用
        else:
            if self.image_on_canvas_right is not None:
                canvas.delete(self.image_on_canvas_right)
            self.image_on_canvas_right = canvas.create_image(x_offset, y_offset, anchor='nw', image=img_tk)
            canvas.image = img_tk  # 保持引用

    # ===================== 处理画布尺寸变化 =====================

    def on_canvas_resize(self, event):
        """
        处理Canvas尺寸变化
        :param event:
        """
        # 在Canvas尺寸变化时重新显示图像，以保持自适应
        if event.widget == self.ui.tk_canvas_left_canvas and self.image_paths:
            image_path = self.image_paths[self.current_index]
            self.display_image(Image.open(image_path), self.ui.tk_canvas_left_canvas, is_left=True)
        elif event.widget == self.ui.tk_canvas_right_canvas and self.image_paths:
            image_path = self.image_paths[self.current_index]
            img = cv2.imread(image_path)
            img_with_boxes = self.draw_boxes(img.copy(), self.detection_results[image_path])
            self.display_image(img_with_boxes, self.ui.tk_canvas_right_canvas, is_left=False)

    # ==================== 处理鼠标滚轮事件 ====================

    def on_mouse_wheel(self, event):
        """
        处理鼠标滚轮事件，根据滚动方向调整缩放比例，并更新图像显示
        """
        # 设置缩放限制，最大放大3倍，最小缩小到0.5倍
        if event.delta > 0:
            if self.scale_factor < 3.0:
                self.scale_factor *= 1.1
        elif event.delta < 0:
            if self.scale_factor > 0.5:
                self.scale_factor /= 1.1

        # 重新显示图像以应用新的缩放比例
        if event.widget == self.ui.tk_canvas_left_canvas:
            image_path = self.image_paths[self.current_index]
            self.display_image(Image.open(image_path), self.ui.tk_canvas_left_canvas, is_left=True)
        elif event.widget == self.ui.tk_canvas_right_canvas:
            image_path = self.image_paths[self.current_index]
            img = cv2.imread(image_path)
            img_with_boxes = self.draw_boxes(img.copy(), self.detection_results[image_path])
            self.display_image(img_with_boxes, self.ui.tk_canvas_right_canvas, is_left=False)

    # =================== 处理左侧画布移动事件 ====================

    def on_drag_start_left(self, event):
        """
        处理左画布鼠标拖动开始事件
        """
        # 记录拖动开始位置
        self.drag_data_left["x"] = event.x
        self.drag_data_left["y"] = event.y

    def on_drag_motion_left(self, event):
        """
        处理左画布鼠标拖动事件
        """
        # 计算拖动的偏移量
        delta_x = event.x - self.drag_data_left["x"]
        delta_y = event.y - self.drag_data_left["y"]

        # 更新偏移量
        self.offset_x_left += delta_x
        self.offset_y_left += delta_y

        # 更新拖动数据
        self.drag_data_left["x"] = event.x
        self.drag_data_left["y"] = event.y

        # 重新显示左画布的图像
        self.update_on_canvas(self.ui.tk_canvas_left_canvas, is_left=True)

    def on_drag_stop_left(self, event):
        """
        处理左画布鼠标拖动结束事件
        """
        # 结束拖动操作
        self.drag_data_left = {"x": 0, "y": 0}

    # =================== 处理右侧画布移动事件 ====================

    def on_drag_start_right(self, event):
        """
        处理右画布鼠标拖动开始事件
        """
        # 记录拖动开始位置
        self.drag_data_right["x"] = event.x
        self.drag_data_right["y"] = event.y

    def on_drag_motion_right(self, event):
        """
        处理右画布鼠标拖动事件
        """
        # 计算拖动的偏移量
        delta_x = event.x - self.drag_data_right["x"]
        delta_y = event.y - self.drag_data_right["y"]

        # 更新偏移量
        self.offset_x_right += delta_x
        self.offset_y_right += delta_y

        # 更新拖动数据
        self.drag_data_right["x"] = event.x
        self.drag_data_right["y"] = event.y

        # 重新显示右画布的图像
        self.update_on_canvas(self.ui.tk_canvas_right_canvas, is_left=False)

    def on_drag_stop_right(self, event):
        """
        处理右画布鼠标拖动结束事件
        """
        # 结束拖动操作
        self.drag_data_right = {"x": 0, "y": 0}

    # =================== 更新画布 ====================

    def update_on_canvas(self, canvas, is_left=True):
        """
        在指定的画布上显示图像
        :param canvas: 目标Canvas
        :param is_left: 是否是左画布
        """
        if is_left:
            image_path = self.image_paths[self.current_index]
            img = Image.open(image_path)
            self.display_image(img, canvas, is_left=True)
        else:
            image_path = self.image_paths[self.current_index]
            img = cv2.imread(image_path)
            img_with_boxes = self.draw_boxes(img.copy(), self.detection_results.get(image_path, []))
            self.display_image(img_with_boxes, canvas, is_left=False)

    def show_image_and_detection(self, is_batched=False):
        """
        显示图像并检测
        :param is_batched: 默认为 False
        """
        if self.image_paths:
            image_path = self.image_paths[self.current_index]
            img = cv2.imread(image_path)

            # 显示原始图像在左侧Canvas上
            self.display_image(Image.open(image_path), self.ui.tk_canvas_left_canvas, is_left=True)

            if is_batched:
                # 使用已有的检测结果
                detected_boxes = self.detection_results.get(image_path, [])
            else:
                # 执行新的目标检测
                start_predict = time.time()
                results = self.model.predict(img, verbose=False)
                print(f"[INFO] - 检测耗时: {time.time() - start_predict}")
                detected_boxes = []

                if results and hasattr(results[0], 'boxes'):
                    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()  # 获取检测框的坐标
                    classes = results[0].boxes.cls.cpu().numpy()  # 获取检测框的类
                    confidences = results[0].boxes.conf.cpu().numpy()  # 获取检测框的置信度

                    for box, cls, conf in zip(boxes_xyxy, classes, confidences):
                        detected_boxes.append(list(box) + [int(cls), float(conf)])

                # 更新检测结果
                self.detection_results[image_path] = detected_boxes

            # 绘制检测框并显示在右侧Canvas上
            img_with_boxes = self.draw_boxes(img.copy(), detected_boxes)
            self.display_image(img_with_boxes, self.ui.tk_canvas_right_canvas, is_left=False)

    def draw_boxes(self, img, detected_boxes):
        """
        绘制检测框
        :param img:
        :param detected_boxes:
        :return:
        """
        # print("[INFO] - 绘制检测框")
        # print("[DEBUG] - detected_boxes: ", detected_boxes)

        # 获取图像尺寸
        height, width = img.shape[:2]
        # 定义检测框的基础厚度
        base_thickness = 1.5
        # 根据图像尺寸调整厚度
        thickness_ratio = int(width / 640 + height / 640)  # 厚度比
        thickness = int(base_thickness * thickness_ratio)

        for box in detected_boxes:
            # print("[DEBUG] - box: ", box)
            x1, y1, x2, y2, cls, conf = box
            if conf >= config['confidence']:
                # 使用类别数字查找对应的类别名称
                category_name = self.categories.get(cls, 'Unknown')  # 默认为 'Unknown'
                label = f"{category_name} {conf:.2f}"
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness)
                cv2.putText(img, label, (int(x1), int(y1) - thickness), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0),
                            thickness)
        return img

    @staticmethod
    def del_output_cut():
        """删除输出目录，删除裁剪目录"""
        # 清空结果输出目录
        if os.path.exists(config['output_path']):
            for file in os.listdir(config['output_path']):
                file_path = os.path.join(config['output_path'], file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"[INFO] - 已删除文件 {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"[INFO] - 已删除目录 {file_path}")
        else:
            print("[ERROR] - 输出目录不存在!!!")
            messagebox.showwarning("警告", f"输出目录 '{config['output_path']}' 不存在！\n将重新创建输出目录")
            os.makedirs(config['output_path'])
            print("[INFO] - 输出目录已创建")

        # 清空目标裁剪目录
        if os.path.exists(config['cut_image_path']):
            for file in os.listdir(config['cut_image_path']):
                file_path = os.path.join(config['cut_image_path'], file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"[INFO] - 已删除文件 {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"[INFO] - 已删除目录 {file_path}")
        else:
            print("[ERROR] - 裁剪目录不存在!!!")
            messagebox.showwarning("警告", f"裁剪目录 '{config['cut_image_path']}' 不存在！\n将重新创建裁剪目录")
            os.makedirs(config['cut_image_path'])
            print("[INFO] - 裁剪目录已创建")

    # ====================================================================================
    #                                  按钮绑定的方法
    # ====================================================================================

    def select_models(self, evt):
        print("\n[INFO] - 点击按钮 -> <选择模型>")
        model_path = filedialog.askopenfilename(filetypes=[("模型文件", "*.pt;*.onnx")],
                                                initialdir=config["model_path"],
                                                title="选择模型文件")
        print("[INFO] - 模型路径为:", model_path)
        if model_path:
            self.load_model(model_path)

    def batch_recognition(self, evt):
        print("\n[INFO] - 点击按钮 -> <批量识别>")
        if not self.check_model_loaded():
            return

        folder_path = filedialog.askdirectory(initialdir=config["default_image_path"], title="选择文件夹")
        if folder_path and os.path.isdir(folder_path):
            self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                                if f.endswith(('.png', '.jpg', '.jpeg'))]
            self.current_index = 0
            self.detection_results = {}

            # 批量处理所有图像并保存结果
            start_batch = time.time()
            for image_path in self.image_paths:
                img = cv2.imread(image_path)
                results = self.model.predict(img, verbose=False)
                detected_boxes = []

                if results and hasattr(results[0], 'boxes'):
                    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()  # 获取检测框的坐标
                    classes = results[0].boxes.cls.cpu().numpy()  # 获取检测框的类
                    confidences = results[0].boxes.conf.cpu().numpy()  # 获取检测框的置信度

                    for box, cls, conf in zip(boxes_xyxy, classes, confidences):
                        detected_boxes.append(list(box) + [int(cls), float(conf)])

                self.detection_results[image_path] = detected_boxes
            print("[INFO] - 批量处理耗时: ", time.time() - start_batch)

            # [DEBUG] 显示检测结果
            i = 0
            for image_path, detected_boxes in self.detection_results.items():
                print(f"[INFO] - 第 {i + 1} 张图片 '{image_path}' 检测结果: {detected_boxes}")
                i += 1

            # 处理完成后显示第一张图像和检测结果
            self.show_image_and_detection(is_batched=True)

    def single_recognition(self, evt):
        print("\n[INFO] - 点击按钮 -> <单张识别>")
        if not self.check_model_loaded():
            return
        image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")],
                                                initialdir=config["default_image_path"],
                                                title="选择图像文件")
        if image_path and os.path.isfile(image_path):
            self.image_paths = [image_path]
            self.current_index = 0
            self.detection_results = {}
            self.show_image_and_detection()

    def save_result_image(self, evt):
        print("\n[INFO] - 点击按钮 -> <保存结果>")
        if not self.detection_results:
            print("[WARNING] - 请先进行图像检测后再保存结果图片！")
            return

        if self.image_paths:
            save_dir = config['output_path']
            if save_dir and os.path.exists(save_dir):
                for image_path in self.image_paths:
                    img = cv2.imread(image_path)
                    detected_boxes = self.detection_results.get(image_path, [])
                    img_with_boxes = self.draw_boxes(img, detected_boxes)
                    save_path = os.path.join(save_dir, os.path.basename(image_path))
                    cv2.imwrite(save_path, img_with_boxes)
                    print(f"[INFO] - 图片 '{image_path}' 的检测结果图像已保存")
                messagebox.showinfo("提示", f"检测结果图片已保存到 '{save_dir}' 目录")
            else:
                print("[ERROR] - 保存目录不存在，请检查输出路径！")

    def save_detection_coords(self, evt):
        print("\n[INFO] - 点击按钮 -> <保存坐标>")
        if not self.detection_results:
            print("[WARNING] - 请先进行图像检测后再保存结果坐标！")
            return

        output_dir = config['output_path']
        if output_dir and os.path.exists(output_dir):
            for image_path, boxes in self.detection_results.items():
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                txt_path = os.path.join(output_dir, base_name + '.txt')
                with open(txt_path, 'w') as f:
                    for box in boxes:
                        conf = box[5] if len(box) > 5 else 1.0
                        if conf >= config['confidence']:
                            x1, y1, x2, y2 = map(int, box[:4])
                            cls = int(box[4]) if len(box) > 4 else 0
                            f.write(f"{cls} {x1} {y1} {x2} {y2}\n")
                print(f"[INFO] - 图片 '{image_path}' 的坐标已保存到 {txt_path}")
            messagebox.showinfo("提示", f"检测结果坐标已保存到 '{output_dir}' 目录")
        else:
            messagebox.showwarning("警告", "保存目录不存在，请检查输出路径！")
            print("[ERROR] - 保存目录不存在，请检查输出路径！")

    def crop_detected_parts(self, evt):
        print("\n[INFO] - 点击按钮 -> <抠出目标>")
        if not self.detection_results:
            print("[WARNING] - 请先进行图像检测后再抠图！")
            return

        output_dir = config['cut_image_path']
        if output_dir and os.path.exists(output_dir):
            for image_path, boxes in self.detection_results.items():
                img = cv2.imread(image_path)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                for j, box in enumerate(boxes):
                    conf = box[5] if len(box) > 5 else 1.0
                    if conf >= config['confidence']:
                        x1, y1, x2, y2 = map(int, box[:4])
                        cropped_img = img[y1:y2, x1:x2]
                        crop_name = f"{base_name}_part_{j}.png"
                        cv2.imwrite(os.path.join(output_dir, crop_name), cropped_img)
                    else:
                        print(f"[INFO] - 当前检测结果置信度低于 {config['confidence']}，无需抠图")
                print(f"[INFO] - 图片 {base_name}.png 抠图结果已保存到 {output_dir}")
            messagebox.showinfo("提示", f"抠图结果已保存到 '{output_dir}' 目录")
        else:
            messagebox.showwarning("警告", "保存抠出图片的目录不存在，请检查配置文件！")
            print("[ERROR] - 保存目录不存在，请检查配置文件！")

    def show_prev_image(self, evt):
        print("\n[INFO] - 点击按钮 -> <上一张>")
        # 重置数据
        self.scale_factor = 1.0  # 缩放比例重置
        self.offset_x_left = 0  # 左画布图像的水平偏移
        self.offset_y_left = 0  # 左画布图像的垂直偏移
        self.offset_x_right = 0  # 右画布图像的水平偏移
        self.offset_y_right = 0  # 右画布图像的垂直偏移

        if self.image_paths:
            self.current_index = (self.current_index - 1) % len(self.image_paths)
            self.show_image_and_detection(is_batched=True)

    def show_next_image(self, evt):
        print("\n[INFO] - 点击按钮 -> <下一张>")
        # 重置数据
        self.scale_factor = 1.0  # 缩放比例重置
        self.offset_x_left = 0  # 左画布图像的水平偏移
        self.offset_y_left = 0  # 左画布图像的垂直偏移
        self.offset_x_right = 0  # 右画布图像的水平偏移
        self.offset_y_right = 0  # 右画布图像的垂直偏移

        if self.image_paths:
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self.show_image_and_detection(is_batched=True)
