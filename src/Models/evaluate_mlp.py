import argparse
import os
import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from mlp_model import EmotionMLP

RUSSELL_7_CLASS = {
    0: [-0.6, 0.6],
    1: [-0.7, 0.2],
    2: [-0.5, 0.8],
    3: [0.8, 0.2],
    4: [-0.8, -0.4],
    5: [0.2, 0.8],
    6: [0.0, 0.0]
}


def parse_vector_string(s):
    return [float(x) for x in str(s).strip().split()]


def load_directory(data_dir, feature_mode='ab'):
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    all_features = []
    all_labels = []
    skipped = 0

    for csv_path in tqdm(csv_files, desc=f"Loading {os.path.basename(data_dir)} ({feature_mode})"):
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            fb = parse_vector_string(row['feature B'])
            fa = parse_vector_string(row['feature A'])
            va = parse_vector_string(row['emotion'])

            if feature_mode in ('ab', 'a') and all(abs(x) < 1e-8 for x in fb):
                skipped += 1
                continue

            if feature_mode == 'a':
                features = fa
            elif feature_mode == 'b':
                features = fb
            else:
                features = fb + fa

            all_features.append(features)
            all_labels.append(va)

    if feature_mode in ('ab', 'a'):
        print(f"Skipped {skipped} samples with all-zero feature B")

    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)

    return torch.tensor(features), torch.tensor(labels)


def knn_class_from_va(va_vectors):
    standard_points = torch.tensor(list(RUSSELL_7_CLASS.values()), dtype=torch.float32)
    predicted_labels = []

    for va in va_vectors:
        va_tensor = va.cpu() if va.is_cuda else va
        distances = torch.sqrt(torch.sum(torch.square(standard_points - va_tensor), dim=1))
        nearest_idx = torch.argmin(distances).item()
        predicted_labels.append(nearest_idx)

    return predicted_labels


def ccc_metric(pred, target):
    pred_mean = torch.mean(pred, dim=0)
    target_mean = torch.mean(target, dim=0)

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    covariance = torch.mean(pred_centered * target_centered, dim=0)

    pred_var = torch.var(pred, dim=0, unbiased=False)
    target_var = torch.var(target, dim=0, unbiased=False)

    denominator = pred_var + target_var + torch.square(pred_mean - target_mean)
    ccc = (2 * covariance) / (denominator + 1e-8)

    return torch.mean(ccc).item()


def mse_metric(pred, target):
    return nn.MSELoss()(pred, target).item()


def mae_metric(pred, target):
    mae_v = torch.mean(torch.abs(pred[:, 0] - target[:, 0])).item()
    mae_a = torch.mean(torch.abs(pred[:, 1] - target[:, 1])).item()
    mae_total = torch.mean(torch.abs(pred - target)).item()
    return mae_total, mae_v, mae_a


def accuracy_metric(pred_labels, true_labels):
    correct = sum(p == t for p, t in zip(pred_labels, true_labels))
    return correct / len(true_labels) if len(true_labels) > 0 else 0.0


def confusion_matrix(true_classes, pred_classes, num_classes=7):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(true_classes, pred_classes):
        cm[t, p] += 1
    return cm


def class_metrics(cm):
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    class_counts = cm.sum(axis=1)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (precision[i] + recall[i]) > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1[i] = 0.0

    macro_f1 = np.mean(f1)
    weighted_f1 = np.sum(f1 * class_counts) / class_counts.sum() if class_counts.sum() > 0 else 0.0

    return precision, recall, f1, macro_f1, weighted_f1


def compute_all_metrics(pred_va, true_va):
    pred_classes = knn_class_from_va(pred_va)
    true_classes = knn_class_from_va(true_va)

    mae_total, mae_v, mae_a = mae_metric(pred_va, true_va)

    metrics = {
        'mse': mse_metric(pred_va, true_va),
        'mae': mae_total,
        'mae_valence': mae_v,
        'mae_arousal': mae_a,
        'ccc': ccc_metric(pred_va, true_va),
        'accuracy_7class': accuracy_metric(pred_classes, true_classes),
    }

    cm = confusion_matrix(true_classes, pred_classes)
    precision, recall, f1, macro_f1, weighted_f1 = class_metrics(cm)

    metrics['macro_f1'] = macro_f1
    metrics['weighted_f1'] = weighted_f1

    per_class = {}
    for i in range(7):
        per_class[f'class_{i}_precision'] = precision[i]
        per_class[f'class_{i}_recall'] = recall[i]
        per_class[f'class_{i}_f1'] = f1[i]

    return metrics, cm, per_class


def plot_confusion_matrix(cm, output_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(7))
    ax.set_xticklabels([f'C{i}' for i in range(7)])
    ax.set_yticklabels([f'C{i}' for i in range(7)])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (7-Class)')

    for i in range(7):
        for j in range(7):
            text = ax.text(j, i, cm[i, j],
                           ha="center", va="center", color="black", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print("Confusion matrix plot saved.")


def plot_prf_bar(per_class, output_dir):
    import matplotlib.pyplot as plt

    precision = [per_class[f'class_{i}_precision'] for i in range(7)]
    recall = [per_class[f'class_{i}_recall'] for i in range(7)]
    f1 = [per_class[f'class_{i}_f1'] for i in range(7)]

    x = np.arange(7)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x, recall, width, label='Recall', color='lightgreen', edgecolor='black')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='salmon', edgecolor='black')

    ax.set_ylabel('Score')
    ax.set_title('Per-Class Precision / Recall / F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}\n{RUSSELL_7_CLASS[i]}' for i in range(7)], fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prf_bar.png'), dpi=150)
    plt.close()
    print("PRF bar plot saved.")


def plot_dimension_error(true_va, pred_va, output_dir):
    import matplotlib.pyplot as plt

    true_np = true_va.cpu().numpy()
    pred_np = pred_va.cpu().numpy()
    err_v = pred_np[:, 0] - true_np[:, 0]
    err_a = pred_np[:, 1] - true_np[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(err_v, bins=50, color='blue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(np.mean(err_v), color='green', linestyle='--', linewidth=2, label=f'Mean={np.mean(err_v):.4f}')
    ax.set_xlabel('Valence Error (Pred - True)')
    ax.set_ylabel('Number of samples')
    ax.set_title('Valence Prediction Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(err_a, bins=50, color='orange', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(np.mean(err_a), color='green', linestyle='--', linewidth=2, label=f'Mean={np.mean(err_a):.4f}')
    ax.set_xlabel('Arousal Error (Pred - True)')
    ax.set_ylabel('Number of samples')
    ax.set_title('Arousal Prediction Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimension_error.png'), dpi=150)
    plt.close()
    print("Dimension error plot saved.")


def plot_mae_distribution(true_va, pred_va, output_dir):
    import matplotlib.pyplot as plt

    true_np = true_va.cpu().numpy()
    pred_np = pred_va.cpu().numpy()
    mae_per_sample = np.mean(np.abs(pred_np - true_np), axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mae_per_sample, bins=50, color='teal', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(mae_per_sample), color='red', linestyle='--', linewidth=2,
               label=f'Mean MAE={np.mean(mae_per_sample):.4f}')
    ax.set_xlabel('Per-sample MAE')
    ax.set_ylabel('Number of samples')
    ax.set_title('MAE Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mae_distribution.png'), dpi=150)
    plt.close()
    print("MAE distribution plot saved.")


def plot_russell_distribution(true_va, pred_va, output_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    true_np = true_va.cpu().numpy()
    pred_np = pred_va.cpu().numpy()

    ax = axes[0]
    ax.scatter(true_np[:, 0], true_np[:, 1], alpha=0.3, s=10, c='blue', label='True')
    for cid, (vx, vy) in RUSSELL_7_CLASS.items():
        ax.scatter(vx, vy, s=200, c='red', marker='x', linewidths=3)
        ax.text(vx + 0.05, vy + 0.05, f'C{cid}', color='red', fontsize=10)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_title('True Label Distribution on Russell Ring')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.scatter(pred_np[:, 0], pred_np[:, 1], alpha=0.3, s=10, c='green', label='Predicted')
    for cid, (vx, vy) in RUSSELL_7_CLASS.items():
        ax.scatter(vx, vy, s=200, c='red', marker='x', linewidths=3)
        ax.text(vx + 0.05, vy + 0.05, f'C{cid}', color='red', fontsize=10)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_title('Predicted Distribution on Russell Ring')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'russell_distribution.png'), dpi=150)
    plt.close()
    print("Russell distribution plot saved.")


def plot_intensity_histogram(true_va, output_dir):
    import matplotlib.pyplot as plt

    true_np = true_va.cpu().numpy()
    intensity = np.sqrt(true_np[:, 0] ** 2 + true_np[:, 1] ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(intensity, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', label='Origin')
    ax.set_xlabel('Emotion Intensity (distance from origin)')
    ax.set_ylabel('Number of samples')
    ax.set_title('Intensity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(true_np[:, 0], bins=50, alpha=0.5, label='Valence', color='blue')
    ax.hist(true_np[:, 1], bins=50, alpha=0.5, label='Arousal', color='orange')
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel('Value')
    ax.set_ylabel('Number of samples')
    ax.set_title('Valence & Arousal Marginal Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'intensity_histogram.png'), dpi=150)
    plt.close()
    print("Intensity histogram saved.")


def plot_class_distribution(true_va, output_dir):
    import matplotlib.pyplot as plt

    true_classes = knn_class_from_va(true_va)
    class_counts = np.bincount(true_classes, minlength=7)
    class_names = [f'C{i}\n{RUSSELL_7_CLASS[i]}' for i in range(7)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(7), class_counts, color='steelblue', edgecolor='black')
    ax.set_xticks(range(7))
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_ylabel('Number of samples')
    ax.set_title('Sample Count per Russell Class (on Test Set)')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{count}\n({count / len(true_classes) * 100:.1f}%)',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150)
    plt.close()
    print("Class distribution plot saved.")


def plot_error_analysis(true_va, pred_va, output_dir):
    import matplotlib.pyplot as plt

    true_np = true_va.cpu().numpy()
    pred_np = pred_va.cpu().numpy()
    errors = pred_np - true_np
    mse_per_sample = np.mean(errors ** 2, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    n_sample = min(500, len(true_np))
    idx = np.random.choice(len(true_np), n_sample, replace=False)
    ax.quiver(true_np[idx, 0], true_np[idx, 1], errors[idx, 0], errors[idx, 1],
              angles='xy', scale_units='xy', scale=2, alpha=0.5, color='purple', width=0.003)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_title(f'Prediction Error Vectors (n={n_sample})')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(mse_per_sample, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(mse_per_sample), color='red', linestyle='--', linewidth=2,
               label=f'Mean MSE={np.mean(mse_per_sample):.4f}')
    ax.set_xlabel('Per-sample MSE')
    ax.set_ylabel('Number of samples')
    ax.set_title('MSE Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(true_np[:, 0], pred_np[:, 0], alpha=0.3, s=10)
    ax.plot([-1, 1], [-1, 1], 'r--', label='Perfect prediction')
    ax.set_xlabel('True Valence')
    ax.set_ylabel('Predicted Valence')
    ax.set_title('Valence: True vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(true_np[:, 1], pred_np[:, 1], alpha=0.3, s=10)
    ax.plot([-1, 1], [-1, 1], 'r--', label='Perfect prediction')
    ax.set_xlabel('True Arousal')
    ax.set_ylabel('Predicted Arousal')
    ax.set_title('Arousal: True vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_analysis.png'), dpi=150)
    plt.close()
    print("Error analysis plot saved.")


def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Evaluating"):
            features = features.to(device)

            if torch.isnan(features).any() or torch.isinf(features).any():
                continue

            try:
                outputs = model(features)

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    continue

                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())
            except Exception:
                continue

    if len(all_preds) == 0:
        print("Warning: no valid predictions generated during evaluation.")
        return {}, None, None, None, None

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics, cm, per_class = compute_all_metrics(all_preds, all_labels)

    return metrics, cm, per_class, all_preds, all_labels


def measure_inference_speed(model, test_loader, device, num_warmup=10):
    model.eval()
    all_features = []
    with torch.no_grad():
        for features, _ in test_loader:
            all_features.append(features.to(device))
            if len(all_features) >= num_warmup + 50:
                break

    for _ in range(3):
        for features in all_features[:num_warmup]:
            _ = model(features)

    start_time = time.time()
    total_samples = 0
    with torch.no_grad():
        for features in all_features[num_warmup:]:
            _ = model(features)
            total_samples += features.size(0)
    end_time = time.time()

    elapsed = end_time - start_time
    ms_per_sample = (elapsed / total_samples) * 1000
    return ms_per_sample


def main():
    parser = argparse.ArgumentParser(description='Evaluate EmotionMLP')
    parser.add_argument('--test_dir', default='../../Data/PrivateTest', help='Path to PrivateTest data directory')
    parser.add_argument('--model_path', default='./outputs/emotion_mlp_best.pth', help='Path to trained model weights (.pth)')
    parser.add_argument('-o', '--output_dir', default='./outputs', help='Directory to save evaluation results')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--feature_mode', default='ab', choices=['a', 'b', 'ab'], help='a=featureA only, b=featureB only, ab=fused')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    test_features, test_labels = load_directory(args.test_dir, feature_mode=args.feature_mode)
    print(f"Loaded test data: {test_features.shape[0]} samples, {test_features.shape[1]} features")

    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.feature_mode == 'a':
        input_dim, hidden_dim, num_blocks = 768, 256, 3
    elif args.feature_mode == 'b':
        input_dim, hidden_dim, num_blocks = 4, 128, 2
    else:
        input_dim, hidden_dim, num_blocks = 772, 256, 3

    model = EmotionMLP(input_dim=input_dim, hidden_dim=hidden_dim, num_blocks=num_blocks, output_dim=2, dropout_p=0.2).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from {args.model_path}")

    print("\n=== Evaluation Started ===")
    metrics, cm, per_class, all_preds, all_labels = evaluate(model, test_loader, device)

    if not metrics:
        return

    print("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        print(f"{key.upper()}: {value:.4f}")

    print("\n=== Per-Class Metrics ===")
    for i in range(7):
        print(f"C{i}: Precision={per_class[f'class_{i}_precision']:.4f}, "
              f"Recall={per_class[f'class_{i}_recall']:.4f}, "
              f"F1={per_class[f'class_{i}_f1']:.4f}")

    print("\n=== Inference Speed ===")
    ms_per_sample = measure_inference_speed(model, test_loader, device)
    print(f"Inference speed: {ms_per_sample:.4f} ms/sample")
    print(f"Throughput: {1000 / ms_per_sample:.1f} samples/second")

    results_path = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write("Evaluation Metrics\n")
        f.write("=" * 30 + "\n")
        for key, value in metrics.items():
            f.write(f"{key.upper()}: {value:.4f}\n")
        f.write("\nPer-Class Metrics\n")
        f.write("=" * 30 + "\n")
        for i in range(7):
            f.write(f"C{i}: Precision={per_class[f'class_{i}_precision']:.4f}, "
                    f"Recall={per_class[f'class_{i}_recall']:.4f}, "
                    f"F1={per_class[f'class_{i}_f1']:.4f}\n")
        f.write("\nInference Speed\n")
        f.write("=" * 30 + "\n")
        f.write(f"MS per sample: {ms_per_sample:.4f}\n")
        f.write(f"Throughput: {1000 / ms_per_sample:.1f} samples/second\n")
    print(f"\nResults saved to {results_path}")

    print("\n=== Generating Analysis Plots ===")
    plot_confusion_matrix(cm, args.output_dir)
    plot_prf_bar(per_class, args.output_dir)
    plot_dimension_error(all_labels, all_preds, args.output_dir)
    plot_mae_distribution(all_labels, all_preds, args.output_dir)
    plot_russell_distribution(all_labels, all_preds, args.output_dir)
    plot_intensity_histogram(all_labels, args.output_dir)
    plot_class_distribution(all_labels, args.output_dir)
    plot_error_analysis(all_labels, all_preds, args.output_dir)
    print("\nAll plots saved to", args.output_dir)


if __name__ == '__main__':
    main()
