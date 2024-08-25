"""
本代码由[Tkinter布局助手]生成
官网:https://www.pytk.net
QQ交流群:905019785
在线反馈:https://support.qq.com/product/618914
"""
import random
from tkinter import *
from tkinter.ttk import *


class WinGUI(Tk):
    def __init__(self):
        super().__init__()
        self.__win()
        self.tk_button_select_model = self.__tk_button_select_model(self)
        self.tk_button_batch = self.__tk_button_batch(self)
        self.tk_input_model_path = self.__tk_input_model_path(self)
        self.tk_button_single = self.__tk_button_single(self)
        self.tk_button_save_images = self.__tk_button_save_images(self)
        self.tk_button_save_coords = self.__tk_button_save_coords(self)
        self.tk_button_crop = self.__tk_button_crop(self)
        self.tk_button_prev = self.__tk_button_prev(self)
        self.tk_button_next = self.__tk_button_next(self)
        self.tk_frame_main_layout = self.__tk_frame_main_layout(self)
        self.tk_frame_left_layout = self.__tk_frame_left_layout(self.tk_frame_main_layout)
        self.tk_canvas_left_canvas = self.__tk_canvas_left_canvas(self.tk_frame_left_layout)
        self.tk_frame_right_layout = self.__tk_frame_right_layout(self.tk_frame_main_layout)
        self.tk_canvas_right_canvas = self.__tk_canvas_right_canvas(self.tk_frame_right_layout)

    def __win(self):
        self.title("YOLO 模型推理")
        # 设置窗口大小、居中
        width = 960
        height = 600
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)

        self.minsize(width=width, height=height)

    def scrollbar_autohide(self, vbar, hbar, widget):
        """自动隐藏滚动条"""

        def show():
            if vbar: vbar.lift(widget)
            if hbar: hbar.lift(widget)

        def hide():
            if vbar: vbar.lower(widget)
            if hbar: hbar.lower(widget)

        hide()
        widget.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Enter>", lambda e: show())
        if vbar: vbar.bind("<Leave>", lambda e: hide())
        if hbar: hbar.bind("<Enter>", lambda e: show())
        if hbar: hbar.bind("<Leave>", lambda e: hide())
        widget.bind("<Leave>", lambda e: hide())

    def v_scrollbar(self, vbar, widget, x, y, w, h, pw, ph):
        widget.configure(yscrollcommand=vbar.set)
        vbar.config(command=widget.yview)
        vbar.place(relx=(w + x) / pw, rely=y / ph, relheight=h / ph, anchor='ne')

    def h_scrollbar(self, hbar, widget, x, y, w, h, pw, ph):
        widget.configure(xscrollcommand=hbar.set)
        hbar.config(command=widget.xview)
        hbar.place(relx=x / pw, rely=(y + h) / ph, relwidth=w / pw, anchor='sw')

    def create_bar(self, master, widget, is_vbar, is_hbar, x, y, w, h, pw, ph):
        vbar, hbar = None, None
        if is_vbar:
            vbar = Scrollbar(master)
            self.v_scrollbar(vbar, widget, x, y, w, h, pw, ph)
        if is_hbar:
            hbar = Scrollbar(master, orient="horizontal")
            self.h_scrollbar(hbar, widget, x, y, w, h, pw, ph)
        self.scrollbar_autohide(vbar, hbar, widget)

    def __tk_button_select_model(self, parent):
        btn = Button(parent, text="选择模型", takefocus=False, )
        btn.place(relx=0.0052, rely=0.9333, relwidth=0.0625, relheight=0.0500)
        return btn

    def __tk_button_batch(self, parent):
        btn = Button(parent, text="批量识别", takefocus=False, )
        btn.place(relx=0.0781, rely=0.9333, relwidth=0.0625, relheight=0.0500)
        return btn

    def __tk_input_model_path(self, parent):
        ipt = Entry(parent, )
        ipt.place(relx=0.5677, rely=0.9333, relwidth=0.4240, relheight=0.0500)
        return ipt

    def __tk_button_single(self, parent):
        btn = Button(parent, text="单张识别", takefocus=False, )
        btn.place(relx=0.1510, rely=0.9333, relwidth=0.0625, relheight=0.0500)
        return btn

    def __tk_button_save_images(self, parent):
        btn = Button(parent, text="保存结果", takefocus=False, )
        btn.place(relx=0.2240, rely=0.9333, relwidth=0.0625, relheight=0.0500)
        return btn

    def __tk_button_save_coords(self, parent):
        btn = Button(parent, text="保存坐标", takefocus=False, )
        btn.place(relx=0.2969, rely=0.9333, relwidth=0.0625, relheight=0.0500)
        return btn

    def __tk_button_crop(self, parent):
        btn = Button(parent, text="抠出目标", takefocus=False, )
        btn.place(relx=0.3698, rely=0.9333, relwidth=0.0625, relheight=0.0500)
        return btn

    def __tk_button_prev(self, parent):
        btn = Button(parent, text="上一张", takefocus=False, )
        btn.place(relx=0.4427, rely=0.9333, relwidth=0.0521, relheight=0.0500)
        return btn

    def __tk_button_next(self, parent):
        btn = Button(parent, text="下一张", takefocus=False, )
        btn.place(relx=0.5052, rely=0.9333, relwidth=0.0521, relheight=0.0500)
        return btn

    def __tk_frame_main_layout(self, parent):
        frame = Frame(parent, )
        frame.place(relx=0.0000, rely=0.0000, relwidth=1.0000, relheight=0.9150)
        return frame

    def __tk_frame_left_layout(self, parent):
        frame = Frame(parent, )
        frame.place(relx=0.0000, rely=0.0000, relwidth=0.5000, relheight=1.0018)
        return frame

    def __tk_canvas_left_canvas(self, parent):
        canvas = Canvas(parent, bg="#aaa")
        canvas.place(relx=0.0042, rely=0.0036, relwidth=0.9917, relheight=0.9945)
        return canvas

    def __tk_frame_right_layout(self, parent):
        frame = Frame(parent, )
        frame.place(relx=0.5000, rely=0.0000, relwidth=0.5000, relheight=1.0018)
        return frame

    def __tk_canvas_right_canvas(self, parent):
        canvas = Canvas(parent, bg="#aaa")
        canvas.place(relx=0.0042, rely=0.0036, relwidth=0.9917, relheight=0.9964)
        return canvas


class Win(WinGUI):
    def __init__(self, controller):
        self.ctl = controller
        super().__init__()
        self.__event_bind()
        self.__style_config()
        self.ctl.init(self)

    def __event_bind(self):
        self.tk_button_select_model.bind('<Button>', self.ctl.select_models)
        self.tk_button_batch.bind('<Button>', self.ctl.batch_recognition)
        self.tk_button_single.bind('<Button>', self.ctl.single_recognition)
        self.tk_button_save_images.bind('<Button>', self.ctl.save_result_image)
        self.tk_button_save_coords.bind('<Button>', self.ctl.save_detection_coords)
        self.tk_button_crop.bind('<Button>', self.ctl.crop_detected_parts)
        self.tk_button_prev.bind('<Button>', self.ctl.show_prev_image)
        self.tk_button_next.bind('<Button>', self.ctl.show_next_image)
        pass

    def __style_config(self):
        pass


if __name__ == "__main__":
    win = WinGUI()
    win.mainloop()
