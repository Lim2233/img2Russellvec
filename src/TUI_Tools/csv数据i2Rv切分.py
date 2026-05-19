# -*- coding: utf-8 -*-
"""
    浪兮 csv数据i2Rv切分【AppID:110008】
    功能：按CSV第三列分组切分 | 联动110007数据包结构
    规则：自动沿用源数据包同名目录 | 同名预警覆盖 | 层级统一对齐
    路径：完全复用 110005 工具目录
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 规范：兼容导入目录构建模块
try:
    import 浪兮工具包.辅助工具.目录构建 as lxpb
except:
    import 目录构建 as lxpb

import pandas as pd
from pathlib import Path

# ===================== 【全局配置】统一规范 =====================
TOOL_NAME = "csv数据i2Rv切分"
# 核心：复用110005文件夹路径
SHARE_TOOL_ID = "110005"
BASE_RES_DIR = lxpb.R_DIR
SHARE_TOOL_ROOT = os.path.join(BASE_RES_DIR, "图片资源", SHARE_TOOL_ID)
# 默认路径：输入=110007输出数据包目录，输出=切分结果根目录
DEFAULT_INPUT_DIR = os.path.join(SHARE_TOOL_ROOT, "i2Rv输出")
DEFAULT_OUTPUT_DIR = os.path.join(SHARE_TOOL_ROOT, "i2Rv切分结果")

# 自定义路径变量
CUSTOM_INPUT_DIR = None
CUSTOM_OUTPUT_DIR = None

# 自动创建默认目录
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
    "exit": "🚪", "error": "❌", "path": "🗂️", "warn": "⚠️"
}


def print_divider(char="=", length=60):
    print(Colors.BLUE + char * length + Colors.RESET)


def print_menu_title(title, subtitle=""):
    print(f"\n{Colors.GREEN}{title}{Colors.RESET} | {subtitle}")
    print_divider("=")


def print_operation_bar():
    print_divider("=")
    print(f"[0] 退出 | [p]自定义双路径 | 输入数字选择数据包")
    print_divider("=")
    print("请输入指令：", end="")


def log(msg, color=Colors.RESET):
    print(f"{color}{msg}{Colors.RESET}")


# ===================== 【核心功能】复刻原始切分逻辑 + 数据包层级对齐 =====================
def split_csv_by_3rd_column(source_package_path, output_root_path):
    """
    自动沿用源数据包名称创建目标数据包
    存在同名则弹出覆盖确认
    """
    # 获取源数据包名称
    package_name = os.path.basename(source_package_path)
    # 生成目标同名数据包路径
    target_package_path = os.path.join(output_root_path, package_name)

    # 同名数据包存在 弹出覆盖预警
    if os.path.exists(target_package_path):
        log(f"{ICONS['warn']} 检测到切分结果中已存在同名数据包【{package_name}】", Colors.YELLOW)
        confirm = input("是否直接覆盖原有全部数据？(y/n)：").strip().lower()
        if confirm != "y":
            log(f"{ICONS['error']} 已取消本次切分操作", Colors.RED)
            return
        # 确认覆盖 清空旧目录
        import shutil
        shutil.rmtree(target_package_path)
        log(f"{ICONS['success']} 已清空旧数据包，开始重新生成", Colors.GREEN)

    # 创建全新目标数据包目录
    os.makedirs(target_package_path, exist_ok=True)
    input_path = Path(source_package_path)

    # 获取当前数据包内所有CSV文件
    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        log(f"❌ 当前数据包内未找到CSV文件！", Colors.RED)
        return

    log(f"\n🔍 找到 {len(csv_files)} 个CSV文件，开始分组切分...", Colors.BLUE)

    # 遍历处理每个CSV
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
            # 取第三列作为分组依据
            group_col_name = df.columns[2]
            log(f"📄 处理文件：{csv_path.name} | 分组依据列：{group_col_name}", Colors.BLUE)

            # 按列分组
            for category, group_data in df.groupby(group_col_name):
                # 过滤路径非法字符
                safe_category = str(category).replace("/", "_").replace("\\", "_").replace(":", "_")
                # 分类子文件夹
                category_dir = os.path.join(target_package_path, safe_category)
                os.makedirs(category_dir, exist_ok=True)
                # 输出文件名
                out_file_name = f"{csv_path.stem}_{safe_category}{csv_path.suffix}"
                save_path = os.path.join(category_dir, out_file_name)
                # 保存拆分后表格
                group_data.to_csv(save_path, index=False, encoding="utf-8")

            log(f"✅ {csv_path.name} 切分完成", Colors.GREEN)
        except Exception as e:
            log(f"❌ 处理失败 {csv_path.name}：{str(e)}", Colors.RED)

    log(f"\n🎉 数据包【{package_name}】全部切分完成！", Colors.GREEN)
    log(f"📂 存放路径：{target_package_path}", Colors.BLUE)


# ===================== 【自定义路径】双路径配置 =====================
def set_custom_path():
    global CUSTOM_INPUT_DIR, CUSTOM_OUTPUT_DIR
    log("\n" + "=" * 55, Colors.BLUE)
    log(f"{ICONS['path']} 双路径自定义（输入绝对路径）", Colors.YELLOW)
    log("💡 回车 = 恢复默认路径", Colors.YELLOW)
    log("=" * 55, Colors.BLUE)

    # 自定义输入目录
    input_path = input(f"\n输入 待处理数据包根目录 绝对路径\n默认：{DEFAULT_INPUT_DIR}\n> ").strip()
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
    output_path = input(f"\n输入 切分结果根目录 绝对路径\n默认：{DEFAULT_OUTPUT_DIR}\n> ").strip()
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


# ===================== 【主菜单】选择数据包处理 =====================
def main_menu():
    while True:
        input_dir, output_dir = get_active_paths()
        print_menu_title(TOOL_NAME, "按第三列切分CSV | 同名数据包自动对齐")
        log(f"📂 源数据包根目录：{input_dir}", Colors.BLUE)
        log(f"📂 切分结果根目录：{output_dir}", Colors.BLUE)

        # 扫描所有一级数据包文件夹
        package_list = []
        if os.path.exists(input_dir):
            package_list = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

        if package_list:
            log(f"\n{ICONS['folder']} 可选择的源数据包：", Colors.YELLOW)
            for i, pkg in enumerate(package_list, 1):
                log(f"[{i}] {pkg}")
        else:
            log(f"\n{ICONS['error']} 暂无可用源数据包！请先用110007生成", Colors.RED)

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

        # 数字选择数据包执行切分
        if choice.isdigit() and package_list:
            idx = int(choice) - 1
            if 0 <= idx < len(package_list):
                target_package = package_list[idx]
                source_full_path = os.path.join(input_dir, target_package)
                log(f"\n🚀 开始处理源数据包：{target_package}", Colors.BLUE)
                split_csv_by_3rd_column(source_full_path, output_dir)
            else:
                log(f"{ICONS['error']} 序号无效", Colors.RED)
        else:
            log(f"{ICONS['error']} 指令无效", Colors.RED)

        input("\n按回车继续...")


# ===================== 【项目启动接口】遵守规范 =====================
def run_app(run_app_main, *info):
    # 规范：不打印启动器文字，由主程序打印
    input_dir, output_dir = get_active_paths()

    # 接口调用模式
    if len(info) >= 1 and info[0]:
        source_pkg_path = info[0]
        if os.path.isdir(source_pkg_path):
            split_csv_by_3rd_column(source_pkg_path, output_dir)
        else:
            log(f"{ICONS['error']} 数据包路径不存在", Colors.RED)
        return

    # 菜单模式
    main_menu()


if __name__ == "__main__":
    run_app(None)