#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

'''

用途
对第一列作映射变换 emotion变为VA向量

用法
python CLI_emotion2VAVec2.py input_folder output_folder

'''


VEC_MAP = [(-0.6, 0.6), (-0.7, 0.2), (-0.5, 0.8),
           (0.8, 0.2), (-0.8, -0.4), (0.2, 0.8), (0.0, 0.0)]

def getVec2(n):
    return VEC_MAP[n]

def main():
    parser = argparse.ArgumentParser(description='Map emotion (first column, 0-6) to 2D coords.')
    parser.add_argument('input_folder', help='Input folder with CSV files')
    parser.add_argument('output_folder', help='Output folder for transformed CSVs')
    args = parser.parse_args()

    in_dir = Path(args.input_folder)
    out_dir = Path(args.output_folder)

    for csv_path in in_dir.rglob('*.csv'):
        rel_path = csv_path.relative_to(in_dir)
        out_path = out_dir / rel_path.with_name(f'{rel_path.stem}_transformed{rel_path.suffix}')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, newline='') as fin, open(out_path, 'w', newline='') as fout:
            reader = csv.reader(fin)
            writer = csv.writer(fout)
            for i, row in enumerate(reader):
                if i == 0:
                    # 保留列名行
                    writer.writerow(row)
                    continue
                if row:  # 跳过空行
                    try:
                        n = int(row[0])
                        if 0 <= n <= 6:
                            x, y = getVec2(n)
                            row[0] = f"{x} {y}"
                    except (ValueError, IndexError):
                        pass  # 转换失败则保留原值
                writer.writerow(row)

if __name__ == '__main__':
    main()