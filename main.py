"""
入口程序
"""
import os
import shutil
import tempfile

# 导入布局文件
from ui import Win as MainWin
# 导入窗口控制器
from control import Controller as MainUIController


def close_clear_temp():
    """
    关闭程序时清除临时文件
    """
    print("关闭程序")
    temp_dir = tempfile.gettempdir()
    for item in os.listdir(temp_dir):
        if item.startswith("_MEI"):
            shutil.rmtree(os.path.join(temp_dir, item))
            print("\n################ 关闭 -- 临时文件已删除 ################\n")
        app.quit()


# 将窗口控制器传递给UI
app = MainWin(MainUIController())

if __name__ == "__main__":
    # app.ctl.creat_default_config()  # 创建默认配置文件
    # app.ctl.load_config()  # 加载配置文件
    # app.ctl.del_output_cut()  # 删除裁剪和输出文件夹
    app.protocol("WM_DELETE_WINDOW", close_clear_temp)  # 窗口关闭时清除临时文件
    app.mainloop()
