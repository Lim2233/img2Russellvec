import argparse
import pandas as pd
from pathlib import Path

'''

用途
根据usage切分表格,命名格式为 {oldFileName}_{usage}
依赖
pip install argparse pandas pathlib
用法
python CLI_numbers2img.py data_path output_path

'''

def main():
    parser = argparse.ArgumentParser(description="Split CSV files by third column value.")
    parser.add_argument("input_dir", help="Path to input directory containing CSV files.")
    parser.add_argument("output_dir", help="Path to output directory for split CSV files.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(input_dir.glob("*.csv"))
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        col_name = df.columns[2]  # 第三列的列名
        for category, group in df.groupby(col_name):
            safe_cat = str(category).replace("/", "_").replace("\\", "_")  # 可添加更多替换
            out_name = f"{csv_path.stem}_{safe_cat}{csv_path.suffix}"
            group.to_csv(output_dir / out_name, index=False)

if __name__ == "__main__":
    main()