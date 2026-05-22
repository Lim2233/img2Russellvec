# 支流B几何特征提取脚本
# 用法：python src/CLI_Tools/CLI_ExtractFeatureB.py -i slicedCsv -o processedCsv/feature_B
# 依赖：pip install mediapipe opencv-python numpy tqdm
# 注意：运行前确保img文件夹和slicedCsv文件夹在同一目录下
# 输出：processedCsv/feature_B文件夹下的所有CSV，新增feature B列

import argparse
import csv
import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# 引入新版 Tasks API 模块
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LEFT_PUPIL = 468
RIGHT_PUPIL = 473
LEFT_BROW_IN = 55
RIGHT_BROW_IN = 285
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
NOSE_BASE = 2
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
UPPER_LIP = 13
LOWER_LIP = 14

def compute_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def extract_feature_b(image_path, face_mesh):
    default_feature = "[0.0, 0.0, 0.0, 0.0]"
    
    if not os.path.exists(image_path):
        return default_feature
    
    img = cv2.imread(image_path)
    if img is None:
        return default_feature
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    try:
        # 新版要求：输入必须转换为 mp.Image 对象
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        # 新版检测方法由 .process() 改为 .detect()
        results = face_mesh.detect(mp_image)
    except Exception:
        return default_feature
    
    # 新版结果字段由 multi_face_landmarks 改为 face_landmarks
    if not results.face_landmarks:
        return default_feature
    
    # 获取第一个检测到的人脸特征点
    landmarks = results.face_landmarks[0]
    h, w = img.shape[:2]
    
    def get_point(idx):
        return (landmarks[idx].x * w, landmarks[idx].y * h)
    
    left_pupil = get_point(LEFT_PUPIL)
    right_pupil = get_point(RIGHT_PUPIL)
    D_ipd = compute_distance(left_pupil, right_pupil)
    
    if D_ipd < 1e-6:
        D_ipd = 1.0
    
    left_brow_in = get_point(LEFT_BROW_IN)
    right_brow_in = get_point(RIGHT_BROW_IN)
    d_brow = compute_distance(left_brow_in, right_brow_in) / D_ipd
    
    left_mouth = get_point(LEFT_MOUTH)
    right_mouth = get_point(RIGHT_MOUTH)
    mouth_mid = ((left_mouth[0] + right_mouth[0]) / 2, (left_mouth[1] + right_mouth[1]) / 2)
    nose_base = get_point(NOSE_BASE)
    h_mouth = compute_distance(mouth_mid, nose_base) / D_ipd
    
    left_eye_top = get_point(LEFT_EYE_TOP)
    left_eye_bottom = get_point(LEFT_EYE_BOTTOM)
    right_eye_top = get_point(RIGHT_EYE_TOP)
    right_eye_bottom = get_point(RIGHT_EYE_BOTTOM)
    w_eye = (compute_distance(left_eye_top, left_eye_bottom) + compute_distance(right_eye_top, right_eye_bottom)) / 2 / D_ipd
    
    upper_lip = get_point(UPPER_LIP)
    lower_lip = get_point(LOWER_LIP)
    h_jaw = compute_distance(upper_lip, lower_lip) / D_ipd
    
    return f"[{d_brow:.4f}, {h_mouth:.4f}, {w_eye:.4f}, {h_jaw:.4f}]"

def process_csv(input_file, output_file, face_mesh):
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames.copy()
        if 'feature B' not in fieldnames:
            fieldnames.append('feature B')
        
        rows = list(reader)
    
    for row in tqdm(rows, desc=f"Processing {os.path.basename(input_file)}", leave=False):
        image_path = row.get('image_path', '')
        row['feature B'] = extract_feature_b(image_path, face_mesh)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description='Extract Feature B (facial geometric features) from sliced CSV files')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing sliced CSV files')
    parser.add_argument('-o', '--output', required=True, help='Output directory for processed CSV files')
    args = parser.parse_args()
    
    # 新版 Tasks API 初始化配置
    # 注意：请确保运行目录下存在 'face_landmarker.task' 模型文件
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE, # 用于处理单张图片累计的模式
        num_faces=1,                          # 相当于旧版的 max_num_faces=1
        min_face_detection_confidence=0.5     # 相当于旧版的 min_detection_confidence=0.5
    )
    
    # 创建新版 FaceLandmarker 实例
    face_mesh = vision.FaceLandmarker.create_from_options(options)
    
    os.makedirs(args.output, exist_ok=True)
    
    csv_files = sorted([f for f in os.listdir(args.input) if f.endswith('.csv')])
    
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        input_path = os.path.join(args.input, csv_file)
        output_path = os.path.join(args.output, csv_file)
        process_csv(input_path, output_path, face_mesh)
    
    # 新版释放资源的正确方法
    face_mesh.close()
    print(f"Feature B extraction completed. Output saved to {args.output}")

if __name__ == '__main__':
    main()