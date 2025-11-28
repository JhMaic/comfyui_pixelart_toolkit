"""Top-level package for comfyui_pixelart_toolkit."""

__author__ = """Justin Chan"""
__email__ = "owcin.cjh@gmail.com"
__version__ = "0.0.1"

import glob
import importlib.util
import inspect
import os
import subprocess
import sys

REQUIRED_PACKAGES = []


def is_installed(package_name):
    try:
        return importlib.util.find_spec(package_name) is not None
    except ImportError:
        return False


def install_requirements():
    # 获取 requirements.txt 的绝对路径
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

    if not os.path.exists(req_path):
        print(f"## [MyNode] Warning: requirements.txt not found at {req_path}")
        return

    print(f"## [MyNode] Installing dependencies from {req_path}...")
    try:
        # 使用 sys.executable 确保安装到 ComfyUI 当前运行的 Python 环境中
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
        print("## [MyNode] Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"## [MyNode] Error installing dependencies: {e}")


# 2. 检查逻辑：只有当缺少包时，才去跑 requirements.txt
#    这能避免每次启动 ComfyUI 都卡顿几秒去检查 pip
need_install = False
for package in REQUIRED_PACKAGES:
    if not is_installed(package):
        need_install = True
        break

if need_install:
    install_requirements()

# 这里设置你的节点存放的子目录名
NODE_DIR_NAME = "./src/comfyui_pixelart_toolkit"

# 最终导出的映射字典
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def load_nodes():
    # 获取当前文件所在目录
    current_dir = os.path.dirname(__file__)

    # 获取当前包名 (即 comfyui_pixelart_toolkit)
    package_name = __name__

    # 扫描当前目录下的所有 .py 文件
    # 使用 glob 可以更方便地过滤
    py_files = glob.glob(os.path.join(current_dir, "*.py"))

    for file_path in py_files:
        filename = os.path.basename(file_path)

        # 跳过 __init__.py 和其他非节点文件
        if filename.startswith("__") or filename.startswith("."):
            continue

        module_name = filename[:-3]  # 去掉 .py

        try:

            module = importlib.import_module(f".{module_name}", package=package_name)

            # 扫描模块中的类
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if (
                    hasattr(cls, "INPUT_TYPES")
                    and hasattr(cls, "FUNCTION")
                    and (
                        hasattr(cls, "Title")
                        or hasattr(cls, "TITLE")
                        or hasattr(cls, "title")
                    )
                ):

                    # 确定节点 ID
                    node_key = name

                    if node_key in NODE_CLASS_MAPPINGS:
                        continue

                    NODE_CLASS_MAPPINGS[node_key] = cls

                    # 处理显示名称
                    if hasattr(cls, "TITLE"):
                        NODE_DISPLAY_NAME_MAPPINGS[node_key] = cls.TITLE
                    elif hasattr(cls, "title"):
                        NODE_DISPLAY_NAME_MAPPINGS[node_key] = cls.title
                    else:
                        NODE_DISPLAY_NAME_MAPPINGS[node_key] = node_key

                    # print(f"Loaded Node: {node_key}")

        except Exception as e:
            # 打印详细错误方便调试
            print(f"Error loading module {filename}: {e}")


# 执行加载
load_nodes()

# 导出 ComfyUI 需要的变量
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
