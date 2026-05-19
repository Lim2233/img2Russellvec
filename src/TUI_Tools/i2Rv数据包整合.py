# -*- coding: utf-8 -*-
"""
    浪兮 i2Rv数据包整合【AppID:110009】
    功能：全自动合并i2Rv输出+切分结果数据包 | 自动清理冗余文件 | 生成干净数据集
    路径：完全复用 110005 工具目录 | 无交互循环 | 流程打印+最终暂停
"""
import os
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 规范：兼容导入目录构建模块
try:
    import 浪兮工具包.辅助工具.目录构建 as lxpb
except:
    import 目录构建 as lxpb

# ===================== 【全局配置】严格遵守项目规范 =====================
TOOL_NAME = "i2Rv数据包整合"
# 核心：复用110005文件夹路径
SHARE_TOOL_ID = "110005"
BASE_RES_DIR = lxpb.R_DIR
SHARE_TOOL_ROOT = os.path.join(BASE_RES_DIR, "图片资源", SHARE_TOOL_ID)

# 源路径（待整合的两个文件夹）
SOURCE_OUTPUT_DIR = os.path.join(SHARE_TOOL_ROOT, "i2Rv输出")  # 含img+原始csv
SOURCE_SPLIT_DIR = os.path.join(SHARE_TOOL_ROOT, "i2Rv切分结果")  # 含分类csv文件夹
# 目标路径（最终整合数据包）
TARGET_MERGE_DIR = os.path.join(SHARE_TOOL_ROOT, "i2Rv整合数据包")


# ===================== 【UI样式】项目统一风格 =====================
class Colors:
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"


ICONS = {
    "success": "✅", "folder": "📂", "file": "📄",
    "start": "🚀", "clear": "🗑️", "done": "🎉", "warn": "⚠️"
}


def print_divider(char="=", length=60):
    print(Colors.BLUE + char * length + Colors.RESET)


def log(msg, color=Colors.RESET):
    print(f"{color}{msg}{Colors.RESET}")


# ===================== 【核心：全自动整合流程】 =====================
def auto_merge_i2rv_packages():
    log(f"{ICONS['start']} 开始执行 i2Rv 数据包全自动整合流程", Colors.BLUE)
    print_divider()

    # ===================== 步骤1：检查源文件夹是否存在 =====================
    log("[1/6] 检查源数据文件夹是否存在...", Colors.BLUE)
    if not os.path.exists(SOURCE_OUTPUT_DIR):
        log(f"{ICONS['error']} 错误：未找到 i2Rv输出 文件夹！", Colors.RED)
        return
    if not os.path.exists(SOURCE_SPLIT_DIR):
        log(f"{ICONS['error']} 错误：未找到 i2Rv切分结果 文件夹！", Colors.RED)
        return
    log(f"{ICONS['success']} 源文件夹校验通过", Colors.GREEN)

    # ===================== 步骤2：创建整合根目录 =====================
    log("[2/6] 创建最终整合数据包目录...", Colors.BLUE)
    os.makedirs(TARGET_MERGE_DIR, exist_ok=True)
    log(f"{ICONS['success']} 整合目录创建完成：{TARGET_MERGE_DIR}", Colors.GREEN)

    # ===================== 步骤3：获取所有同名数据包 =====================
    log("[3/6] 扫描并匹配同名数据包...", Colors.BLUE)
    # 获取i2Rv输出中的所有数据包文件夹
    output_packages = [f for f in os.listdir(SOURCE_OUTPUT_DIR) if os.path.isdir(os.path.join(SOURCE_OUTPUT_DIR, f))]
    # 获取i2Rv切分结果中的数据包文件夹
    split_packages = [f for f in os.listdir(SOURCE_SPLIT_DIR) if os.path.isdir(os.path.join(SOURCE_SPLIT_DIR, f))]
    # 取交集（同名数据包）
    match_packages = [pkg for pkg in output_packages if pkg in split_packages]

    if not match_packages:
        log(f"{ICONS['warn']} 未找到匹配的同名数据包，整合终止！", Colors.YELLOW)
        return
    log(f"{ICONS['success']} 找到 {len(match_packages)} 个匹配数据包：{match_packages}", Colors.GREEN)

    # ===================== 步骤4：逐个合并数据包 =====================
    log("[4/6] 开始合并数据包...", Colors.BLUE)
    for pkg_name in match_packages:
        log(f"\n└─ 处理数据包：{pkg_name}", Colors.CYAN)

        # 定义路径
        src_output_pkg = os.path.join(SOURCE_OUTPUT_DIR, pkg_name)  # 源输出数据包
        src_split_pkg = os.path.join(SOURCE_SPLIT_DIR, pkg_name)  # 源切分数据包
        target_pkg = os.path.join(TARGET_MERGE_DIR, pkg_name)  # 目标整合数据包
        os.makedirs(target_pkg, exist_ok=True)

        # 移动 img 文件夹（从输出数据包 → 整合数据包）
        img_src = os.path.join(src_output_pkg, "img")
        img_dst = os.path.join(target_pkg, "img")
        if os.path.exists(img_src):
            shutil.move(img_src, img_dst)
            log(f"   ✅ 移动 img 文件夹完成", Colors.GREEN)

        # 移动 所有分类CSV文件夹（从切分数据包 → 整合数据包）
        category_folders = [f for f in os.listdir(src_split_pkg) if os.path.isdir(os.path.join(src_split_pkg, f))]
        for cat_folder in category_folders:
            cat_src = os.path.join(src_split_pkg, cat_folder)
            cat_dst = os.path.join(target_pkg, cat_folder)
            shutil.move(cat_src, cat_dst)
            log(f"   ✅ 移动分类文件夹：{cat_folder}", Colors.GREEN)

    # ===================== 步骤5：清理冗余文件/文件夹 =====================
    log("[5/6] 清理原始冗余数据...", Colors.BLUE)
    # 删除i2Rv输出整个文件夹
    if os.path.exists(SOURCE_OUTPUT_DIR):
        shutil.rmtree(SOURCE_OUTPUT_DIR)
        log(f"{ICONS['clear']} 已删除：i2Rv输出", Colors.YELLOW)
    # 删除i2Rv切分结果整个文件夹
    if os.path.exists(SOURCE_SPLIT_DIR):
        shutil.rmtree(SOURCE_SPLIT_DIR)
        log(f"{ICONS['clear']} 已删除：i2Rv切分结果", Colors.YELLOW)

    # ===================== 步骤6：完成 =====================
    print_divider()
    log(f"{ICONS['done']} 【全部整合完成！】", Colors.GREEN)
    log(f"📂 最终干净数据包路径：{TARGET_MERGE_DIR}", Colors.BLUE)
    log(f"📦 共整合 {len(match_packages)} 个数据包", Colors.BLUE)
    log(f"✅ 每个数据包包含：img文件夹 + 所有分类CSV文件夹", Colors.GREEN)
    print_divider()

    # 防止信息被覆盖，暂停等待用户回车
    input("\n按 回车键 继续...")


# ===================== 【项目启动接口】遵守规范 =====================
def run_app(run_app_main, *info):
    # 规范：不打印启动器文字，直接执行全自动流程
    auto_merge_i2rv_packages()


if __name__ == "__main__":
    run_app(None)