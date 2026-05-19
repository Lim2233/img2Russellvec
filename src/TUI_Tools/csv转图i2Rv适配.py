# -*- coding: utf-8 -*-
"""
    浪兮 csv转图i2Rv适配【AppID:110007】
    功能：48x48灰度CSV → 224x224RGB图 + CSV路径替换 + 自动分块
    路径：完全复用 110005 工具目录 | 支持自定义双路径
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 规范1：兼容导入目录构建模块
try:
    import 浪兮工具包.辅助工具.目录构建 as lxpb
except:
    import 目录构建 as lxpb

import csv
import glob
import numpy as np
from PIL import Image
from torchvision import transforms

# ===================== 【全局配置】严格遵守项目规范 =====================
TOOL_NAME = "csv转图i2Rv适配"
# 核心规范：完全借用110005的文件夹路径
SHARE_TOOL_ID = "110005"
BASE_RES_DIR = lxpb.R_DIR
# 共享路径：图片资源/110005
SHARE_TOOL_ROOT = os.path.join(BASE_RES_DIR, "图片资源", SHARE_TOOL_ID)
DEFAULT_INPUT_DIR = os.path.join(SHARE_TOOL_ROOT, "csv")  # 默认CSV目录
DEFAULT_OUTPUT_DIR = os.path.join(SHARE_TOOL_ROOT, "i2Rv输出")  # 默认输出目录

# 自定义路径变量
CUSTOM_INPUT_DIR = None
CUSTOM_OUTPUT_DIR = None

# 固定参数（原始代码硬编码）
IMAGE_SIZE_ORIGIN = 48
IMAGE_SIZE_TARGET = 224
CHUNK_SIZE = 1000  # 分块行数

# 自动创建共享目录
os.makedirs(DEFAULT_INPUT_DIR, exist_ok=True)
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)


# ===================== 【UI样式】项目统一风格 =====================
class Colors:
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"


ICONS = {
    "success": "✅", "folder": "📂", "file": "📄",
    "exit": "🚪", "error": "❌", "path": "🗂️"
}


def print_divider(char="=", length=60):
    print(Colors.BLUE + char * length + Colors.RESET)


def print_menu_title(title, subtitle=""):
    print(f"\n{Colors.GREEN}{title}{Colors.RESET} | {subtitle}")
    print_divider("=")


def print_operation_bar():
    print_divider("=")
    print(f"[0] 退出 | [p]自定义双路径 | 输入数字选择CSV")
    print_divider("=")
    print("请输入指令：", end="")


def log(msg, color=Colors.RESET):
    print(f"{color}{msg}{Colors.RESET}")


# ===================== 【核心功能】1:1复刻原始代码 =====================
def process_csv(input_path, output_dir, resize_transform, global_counter):
    """处理单个CSV文件，原始逻辑无修改"""
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        data_rows = list(reader)

    total_rows = len(data_rows)
    base_name = os.path.splitext(os.path.basename(input_path))[0] + "_processed"
    new_header = header.copy()
    new_header[1] = "image_path"

    def write_chunk(rows, part_label=""):
        nonlocal global_counter
        csv_name = f"{base_name}{part_label}.csv"
        csv_path = os.path.join(output_dir, csv_name)

        with open(csv_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(new_header)

            for row in rows:
                # 48x48灰度转RGB + 缩放224x224
                gray_vals = np.array(row[1].strip().split(), dtype=np.uint8).reshape(48, 48)
                img = Image.fromarray(gray_vals, mode="L").convert("RGB")
                img_resized = resize_transform(img)

                # 全局6位编号保存图片
                img_filename = f"{global_counter:06d}.png"
                img_path = os.path.join(img_dir, img_filename)
                img_resized.save(img_path)

                # 写入相对路径
                row[1] = os.path.join("img", img_filename)
                writer.writerow(row)
                global_counter += 1

        log(f"✅ 已保存: {csv_path} (共 {len(rows)} 行)", Colors.GREEN)

    # 自动分块逻辑
    if total_rows <= CHUNK_SIZE:
        write_chunk(data_rows)
    else:
        for part_num, i in enumerate(range(0, total_rows, CHUNK_SIZE), start=1):
            chunk = data_rows[i:i + CHUNK_SIZE]
            write_chunk(chunk, f"_part{part_num}")

    return global_counter


# ===================== 【自定义路径】双路径配置 =====================
def set_custom_path():
    global CUSTOM_INPUT_DIR, CUSTOM_OUTPUT_DIR
    log("\n" + "=" * 55, Colors.BLUE)
    log(f"{ICONS['path']} 双路径自定义（输入绝对路径）", Colors.YELLOW)
    log("💡 回车 = 恢复默认路径", Colors.YELLOW)
    log("=" * 55, Colors.BLUE)

    # 自定义CSV输入目录
    input_path = input(f"\n输入 CSV文件夹 绝对路径\n默认：{DEFAULT_INPUT_DIR}\n> ").strip()
    if input_path:
        if os.path.isabs(input_path):
            CUSTOM_INPUT_DIR = input_path
            os.makedirs(CUSTOM_INPUT_DIR, exist_ok=True)
            log(f"{ICONS['success']} 输入目录已设置：{CUSTOM_INPUT_DIR}", Colors.GREEN)
        else:
            log(f"{ICONS['error']} 非绝对路径，使用默认", Colors.RED)
    else:
        CUSTOM_INPUT_DIR = None
        log(f"{ICONS['success']} 输入目录恢复默认", Colors.GREEN)

    # 自定义输出目录
    output_path = input(f"\n输入 输出文件夹 绝对路径\n默认：{DEFAULT_OUTPUT_DIR}\n> ").strip()
    if output_path:
        if os.path.isabs(output_path):
            CUSTOM_OUTPUT_DIR = output_path
            os.makedirs(CUSTOM_OUTPUT_DIR, exist_ok=True)
            log(f"{ICONS['success']} 输出目录已设置：{CUSTOM_OUTPUT_DIR}", Colors.GREEN)
        else:
            log(f"{ICONS['error']} 非绝对路径，使用默认", Colors.RED)
    else:
        CUSTOM_OUTPUT_DIR = None
        log(f"{ICONS['success']} 输出目录恢复默认", Colors.GREEN)

    input("\n按回车返回菜单...")


# ===================== 【获取当前生效路径】 =====================
def get_active_paths():
    input_dir = CUSTOM_INPUT_DIR if CUSTOM_INPUT_DIR else DEFAULT_INPUT_DIR
    output_dir = CUSTOM_OUTPUT_DIR if CUSTOM_OUTPUT_DIR else DEFAULT_OUTPUT_DIR
    return input_dir, output_dir


# ===================== 【主菜单】常驻控制台 =====================
def main_menu():
    # 图片缩放预处理
    resize_transform = transforms.Resize((IMAGE_SIZE_TARGET, IMAGE_SIZE_TARGET))

    while True:
        input_dir, output_dir = get_active_paths()
        print_menu_title(TOOL_NAME, "48x48→224x224 RGB | 自动分块")
        log(f"📂 当前CSV目录：{input_dir}", Colors.BLUE)
        log(f"📂 当前输出目录：{output_dir}", Colors.BLUE)

        # 扫描CSV文件
        csv_list = []
        if os.path.exists(input_dir):
            csv_list = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])

        if csv_list:
            log(f"\n{ICONS['file']} 可用CSV文件：", Colors.YELLOW)
            for i, f in enumerate(csv_list, 1):
                log(f"[{i}] {f}")
        else:
            log(f"\n{ICONS['error']} 目录内无CSV文件！", Colors.RED)

        print_operation_bar()
        choice = input().strip().lower()

        # 0 退出工具
        if choice == "0":
            log(f"{ICONS['exit']} 已退出工具", Colors.YELLOW)
            break

        # p 自定义路径
        if choice == "p":
            set_custom_path()
            continue

        # 数字选择 批量处理CSV
        if choice.isdigit() and csv_list:
            idx = int(choice) - 1
            if 0 <= idx < len(csv_list):
                file_path = os.path.join(input_dir, csv_list[idx])
                log(f"\n🔄 正在处理：{csv_list[idx]}", Colors.BLUE)
                # 全局计数器归零
                global_counter = 0
                process_csv(file_path, output_dir, resize_transform, global_counter)
                log(f"\n{ICONS['success']} 文件处理完成！", Colors.GREEN)
            else:
                log(f"{ICONS['error']} 序号无效", Colors.RED)
        else:
            log(f"{ICONS['error']} 指令无效", Colors.RED)

        input("\n按回车继续...")


# ===================== 【项目启动接口】 =====================
def run_app(run_app_main, *info):
    # 规范2：不打印启动器文字，由主程序打印
    input_dir, output_dir = get_active_paths()

    # 接口调用模式
    if len(info) >= 1 and info[0]:
        resize_transform = transforms.Resize((IMAGE_SIZE_TARGET, IMAGE_SIZE_TARGET))
        csv_path = info[0]
        if os.path.exists(csv_path):
            global_counter = 0
            process_csv(csv_path, output_dir, resize_transform, global_counter)
        else:
            log(f"{ICONS['error']} 文件不存在", Colors.RED)
        return

    # 菜单模式
    main_menu()


if __name__ == "__main__":
    run_app(None)