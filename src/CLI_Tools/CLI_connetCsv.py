#!/usr/bin/env python3
"""
将输入目录中所有 CSV 文件按 _partN 顺序拼接为一个 CSV 文件，
输出文件名由原文件名去掉 _partN 部分得到。
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path


def parse_part(filename_stem: str):
    """
    从文件名（无扩展名）中提取 _partN 部分，返回 (base_name, part_number)。
    如果不存在 _partN，则返回 (filename_stem, None)。
    """
    match = re.search(r'_(part\d+)$', filename_stem)  # 优先匹配结尾的 _partN
    if not match:
        # 如果不在结尾，尝试匹配任意位置的 _part数字
        match = re.search(r'_(part\d+)_', filename_stem)
    if not match:
        match = re.search(r'_(part\d+)', filename_stem)

    if match:
        part_str = match.group(1)          # e.g. "part1"
        number = int(part_str[4:])         # 提取数字部分
        base_name = filename_stem[:match.start()] + filename_stem[match.end():]
        # 去掉可能产生的多余下划线或连字符，简单处理：将连续下划线替换为单个
        base_name = re.sub(r'_+', '_', base_name).strip('_')
        return base_name, number
    else:
        # 没有 _partN，直接使用原名，排序号设为0（或特殊处理）
        return filename_stem, None


def collect_csv_files(input_dir: Path):
    """收集 input_dir 下所有 .csv 文件并解析名称。"""
    files = []
    for f in input_dir.glob('*.csv'):
        stem = f.stem
        base, num = parse_part(stem)
        files.append((base, num, f))
    return files


def validate_and_sort(files):
    """
    检查所有文件的 base_name 是否一致，并按 part 数字排序。
    返回 (base_name, sorted_files)
    """
    # 筛选出有 part 数字的文件
    numbered = [(b, n, f) for b, n, f in files if n is not None]
    unnumbered = [(b, n, f) for b, n, f in files if n is None]

    if unnumbered:
        print("警告：以下文件不包含 _partN 模式，将被当做整体处理：",
              [f.name for _, _, f in unnumbered], file=sys.stderr)

    if not numbered:
        # 所有文件都无 part，则按文件名排序，合并后的名字取第一个文件的 stem
        sorted_files = sorted(files, key=lambda x: x[2].name)
        return sorted_files[0][0], [f for _, _, f in sorted_files]

    # 检查所有带编号的 base 是否一致
    bases = {b for b, n, f in numbered}
    if len(bases) > 1:
        raise ValueError(f"文件的基础名称不一致: {bases}")
    base_name = bases.pop()

    # 如果有 unnumbered，也检查是否与 base_name 一致
    for b, n, f in unnumbered:
        if b != base_name:
            raise ValueError(f"文件 {f.name} 的基础名称 '{b}' 与主要基础名称 '{base_name}' 不一致")

    # 按数字排序
    sorted_numbered = sorted(numbered, key=lambda x: x[1])
    all_sorted = [f for _, _, f in sorted_numbered] + [f for _, _, f in unnumbered]
    return base_name, all_sorted


def merge_csv(file_list, output_path):
    """合并所有 CSV 文件，只保留第一个文件的表头。"""
    if not file_list:
        print("没有找到 CSV 文件。", file=sys.stderr)
        return

    with open(file_list[0], 'r', encoding='utf-8', newline='') as f_in:
        reader = csv.reader(f_in)
        header = next(reader)

    with open(output_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)

        for i, file_path in enumerate(file_list):
            with open(file_path, 'r', encoding='utf-8', newline='') as f_in:
                reader = csv.reader(f_in)
                # 跳过表头
                next(reader)
                if i == 0:
                    # 第一个文件已经读取过表头，这里不需要额外操作
                    pass
                for row in reader:
                    writer.writerow(row)

    print(f"合并完成，输出文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="将输入目录下所有 CSV 文件按 _partN 顺序合并为一个文件"
    )
    parser.add_argument('input_dir', type=str, help='输入目录路径')
    parser.add_argument('output_dir', type=str, help='输出目录路径')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"错误: 输入目录 '{input_dir}' 不存在。", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_csv_files(input_dir)
    if not files:
        print(f"在 '{input_dir}' 中未找到 CSV 文件。", file=sys.stderr)
        sys.exit(1)

    try:
        base_name, sorted_files = validate_and_sort(files)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    output_file = output_dir / f"{base_name}.csv"
    merge_csv(sorted_files, output_file)


if __name__ == '__main__':
    main()