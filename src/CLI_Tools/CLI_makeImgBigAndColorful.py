import csv, os, argparse
import numpy as np
from PIL import Image
from torchvision import transforms

'''

用途
对csv中每行的灰度图像矩阵进行处理，得到新的矩阵
依赖
pip install pillow numpy torchvision
用法
python CLI_makeImgBig&Colorful.py data.csv output_path

'''

def main():
    parser = argparse.ArgumentParser(description="将 CSV 中 48x48 灰度图像转为 224x224 RGB 图像并保存")
    parser.add_argument("csv_file", help="输入 CSV 文件路径")
    parser.add_argument("output_dir", help="输出文件夹路径")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, os.path.basename(args.csv_file))

    # 定义变换：缩放到 224x224
    resize = transforms.Resize((224, 224))

    with open(args.csv_file, "r", newline="") as fin, open(out_path, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        # 写入表头（第一行不处理）
        writer.writerow(next(reader))

        for row in reader:
            # 第二列是空格分隔的 2304 个灰度值
            gray_vals = np.array(row[1].strip().split(), dtype=np.uint8).reshape(48, 48)
            # 灰度 -> RGB -> Resize -> 平铺为空格分隔的字符串
            img = Image.fromarray(gray_vals, mode="L").convert("RGB")
            img_resized = resize(img)
            rgb_flat = np.array(img_resized).flatten()
            row[1] = " ".join(map(str, rgb_flat))
            writer.writerow(row)

if __name__ == "__main__":
    main()