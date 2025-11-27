"""Top-level package for comfyui_pixelart_utils."""

__author__ = """Justin Chan"""
__email__ = "owcin.cjh@gmail.com"
__version__ = "0.0.1"

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
NODE_DIR_NAME = "./src/comfyui_pixelart_utils"

# 最终导出的映射字典
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def load_nodes():
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接节点目录路径
    nodes_path = os.path.join(current_dir, NODE_DIR_NAME)

    # 检查目录是否存在
    if not os.path.exists(nodes_path):
        print(f"Skipping node loading: {nodes_path} does not exist.")
        return

    # 遍历目录下的所有文件
    for filename in os.listdir(nodes_path):
        if filename.endswith(".py") and not filename.startswith("__"):
            file_path = os.path.join(nodes_path, filename)
            module_name = filename[:-3]  # 去掉 .py 后缀

            try:
                # 动态导入模块
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # 扫描模块中的类
                for name, cls in inspect.getmembers(module, inspect.isclass):
                    # 关键逻辑：如何判断一个类是不是 ComfyUI 节点？
                    # 通常节点类必须包含 INPUT_TYPES 方法和 FUNCTION 属性
                    if hasattr(cls, "INPUT_TYPES") and hasattr(cls, "FUNCTION"):

                        # 1. 确定节点 ID (Key)
                        # 如果类里定义了 OUTPUT_NODE = True 等属性也可以在这里判断
                        # 优先使用类中定义的别名，否则使用类名
                        node_key = name

                        # 2. 注册到 CLASS MAPPINGS
                        if node_key in NODE_CLASS_MAPPINGS:
                            print(
                                f"Warning: Duplicate node name '{node_key}' found in {filename}, skipping."
                            )
                            continue

                        NODE_CLASS_MAPPINGS[node_key] = cls

                        # 3. 注册到 DISPLAY NAME MAPPINGS
                        # 你的示例中使用了 .title 属性，这里做兼容处理
                        if hasattr(cls, "title"):
                            NODE_DISPLAY_NAME_MAPPINGS[node_key] = cls.title
                        elif hasattr(cls, "TITLE"):  # ComfyUI 社区有时也用 TITLE
                            NODE_DISPLAY_NAME_MAPPINGS[node_key] = cls.TITLE
                        else:
                            # 如果没有定义标题，使用类名并插入空格（可选）
                            NODE_DISPLAY_NAME_MAPPINGS[node_key] = node_key

                        print(f"Loaded node: {node_key}")

            except Exception as e:
                print(f"Error loading module {filename}: {e}")


# 执行加载
load_nodes()

# 导出 ComfyUI 需要的变量
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
