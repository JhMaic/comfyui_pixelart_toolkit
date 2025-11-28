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
    # Get absolute path to requirements.txt
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

    if not os.path.exists(req_path):
        print(f"## [MyNode] Warning: requirements.txt not found at {req_path}")
        return

    print(f"## [MyNode] Installing dependencies from {req_path}...")
    try:
        # Use sys.executable to ensure installation to ComfyUI's currently running Python environment
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
        print("## [MyNode] Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"## [MyNode] Error installing dependencies: {e}")


# 2. Check logic: Only run requirements.txt when packages are missing
#    This avoids a few seconds of lag on every ComfyUI startup to check pip
need_install = False
for package in REQUIRED_PACKAGES:
    if not is_installed(package):
        need_install = True
        break

if need_install:
    install_requirements()

# Set your node storage subdirectory name here
NODE_DIR_NAME = "./src/comfyui_pixelart_toolkit"

# Final exported mapping dictionaries
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def load_nodes():
    # Get current file directory
    current_dir = os.path.dirname(__file__)

    # Get current package name (i.e., comfyui_pixelart_toolkit)
    package_name = __name__

    # Scan all .py files in current directory
    # Using glob for easier filtering
    py_files = glob.glob(os.path.join(current_dir, "*.py"))

    for file_path in py_files:
        filename = os.path.basename(file_path)

        # Skip __init__.py and other non-node files
        if filename.startswith("__") or filename.startswith("."):
            continue

        module_name = filename[:-3]  # Remove .py

        try:

            module = importlib.import_module(f".{module_name}", package=package_name)

            # Scan classes in module
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

                    # Determine node ID
                    node_key = name

                    if node_key in NODE_CLASS_MAPPINGS:
                        continue

                    NODE_CLASS_MAPPINGS[node_key] = cls

                    # Handle display name
                    if hasattr(cls, "TITLE"):
                        NODE_DISPLAY_NAME_MAPPINGS[node_key] = cls.TITLE
                    elif hasattr(cls, "title"):
                        NODE_DISPLAY_NAME_MAPPINGS[node_key] = cls.title
                    else:
                        NODE_DISPLAY_NAME_MAPPINGS[node_key] = node_key

                    # print(f"Loaded Node: {node_key}")

        except Exception as e:
            # Print detailed error for debugging
            print(f"Error loading module {filename}: {e}")


# Execute loading
load_nodes()

# Export variables needed by ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
