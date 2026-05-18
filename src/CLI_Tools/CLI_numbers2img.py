import csv
import os
import argparse
import numpy as np
from PIL import Image

'''

用途
把csv中的数组转成图片,命名格式为 {行号(占5位)}_{emotion}{usage}
依赖
pip install pillow numpy
用法
python CLI_numbers2img.py data.csv output_path

'''

def process_csv(input_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # 跳过可能的标题行（如果第一行第一列不是纯数字，则认为它是表头）
        first_row = next(reader, None)
        if first_row and not first_row[0].strip().isdigit():
            pass  # 已跳过
        else:
            # 如果第一行就是数据，回退处理
            rows = [first_row] if first_row else []
            reader = rows + list(reader)

        for idx, row in enumerate(reader, start=1):
            if len(row) < 3:
                continue
            number_str = row[0].strip()
            # 额外保险：如果数字列不合法则跳过
            if not number_str.lstrip('-').isdigit():
                continue

            pixels = list(map(int, row[1].strip().split()))
            string = row[2].strip()
            if len(pixels) != 2304:
                continue

            img = Image.fromarray(np.array(pixels, dtype=np.uint8).reshape(48, 48), 'L')
            filename = f"{idx:05d}_{number_str}{string}.jpg"
            img.save(os.path.join(output_dir, filename), 'JPEG')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从CSV生成48x48灰度JPG图片")
    parser.add_argument("input_csv", help="输入CSV文件路径")
    parser.add_argument("output_dir", help="输出文件夹路径")
    args = parser.parse_args()
    process_csv(args.input_csv, args.output_dir)