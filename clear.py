import os
import shutil
import tempfile


def clear_temp(is_exit=True):
    """
    清除临时文件
    """
    flag = is_exit
    print("\n启动清理程序\n")
    temp_dir = tempfile.gettempdir()
    if os.path.exists(temp_dir):
        print("存在临时目录：", temp_dir)
        for item in os.listdir(temp_dir):
            if item.startswith("_MEI"):
                print("正在删除临时目录下以 '_MEI' 开头的文件夹：", item)
                shutil.rmtree(os.path.join(temp_dir, item))
                flag = False

        if flag:
            print("临时目录下无以 '_MEI' 开头的文件夹，无需清理")
    else:
        print("临时目录不存在")

    input("按回车键退出或关闭程序。。。")


if __name__ == '__main__':
    clear_temp()
