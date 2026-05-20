import torch
import os
import argparse
import csv
from pathlib import Path
from PIL import Image
from torchvision import transforms

os.environ['XFORMERS_DISABLED'] = '1'

# ---------- 特征提取模型（只加载一次） ----------
def load_model():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = model.eval()
    return model

def extract_feature(model, image_path, transform):
    """提取单张图片的 768 维 CLS token 特征，返回空格分隔字符串，失败返回空字符串"""
    try:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)          # (1, 3, 224, 224)
        with torch.no_grad():
            out = model.forward_features(tensor)
            cls_token = out['x_norm_clstoken']          # (1, 768)
        # 转为空格分隔的字符串，保留6位小数
        feat_str = " ".join([f"{v:.6f}" for v in cls_token[0].tolist()])
        return feat_str
    except Exception as e:
        print(f"[警告] 处理图片 {image_path} 失败: {e}")
        return ""

def process_csv(input_csv, output_csv, model, transform, input_dir):
    """处理单个 CSV 文件，添加 FeatureA 列"""
    rows = []
    with open(input_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        # 假设已有三列，列名任意；新增第四列列名 FeatureA
        new_header = header + ['FeatureA']
        rows.append(new_header)

        for line_num, row in enumerate(reader, start=2):
            if len(row) < 3:
                print(f"[警告] {input_csv} 第 {line_num} 行列数不足 3，跳过该行")
                row.append("")   # 补足 FeatureA 列为空
                rows.append(row)
                continue

            img_path = os.path.join(input_dir,row[1])   # 第二列为图片路径（索引1）
            feat_str = extract_feature(model, img_path, transform)
            row.append(feat_str)
            rows.append(row)

    # 写入输出文件
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"已生成: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='从 CSV 中的图片路径提取 DINOv2 ViT-B/14 特征 (768维)')
    # 位置参数，不需要 -- 前缀，顺序传参
    parser.add_argument('input', help='输入文件夹，内含若干 CSV 文件')
    parser.add_argument('output', help='输出文件夹，生成处理后的 CSV 文件')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 预处理流程（与原始代码一致）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 加载模型
    print("正在加载 DINOv2 ViT-B/14 模型...")
    model = load_model()
    print("模型加载完成。\n")

    # 收集所有 CSV 文件
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"在 {input_dir} 中未找到任何 CSV 文件。")
        return

    for csv_path in csv_files:
        out_csv = output_dir / csv_path.name
        print(f"处理中: {csv_path}")
        process_csv(str(csv_path), str(out_csv), model, transform, input_dir)

    print(f"\n全部完成！结果保存在 {output_dir}")

if __name__ == "__main__":
    main()