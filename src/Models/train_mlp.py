import argparse
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from mlp_model import EmotionMLP


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ccc_loss(pred, target):
    pred_mean = torch.mean(pred, dim=0)
    target_mean = torch.mean(target, dim=0)

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    covariance = torch.mean(pred_centered * target_centered, dim=0)

    pred_var = torch.var(pred, dim=0, unbiased=False)
    target_var = torch.var(target, dim=0, unbiased=False)

    denominator = pred_var + target_var + torch.square(pred_mean - target_mean)

    ccc = (2 * covariance) / (denominator + 1e-8)
    ccc_mean = torch.mean(ccc)

    return 1 - ccc_mean


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


def validate_tensor(tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return False
    if torch.abs(tensor).sum() < 1e-8:
        return False
    return True


def val_epoch(model, val_loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            if not validate_tensor(features):
                continue

            try:
                outputs = model(features)
                if not validate_tensor(outputs):
                    continue
                loss = ccc_loss(outputs, labels)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                total_loss += loss.item()
                num_batches += 1
            except Exception:
                continue

    return total_loss / max(num_batches, 1)


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)

        if not validate_tensor(features):
            continue

        try:
            optimizer.zero_grad()
            outputs = model(features)

            if not validate_tensor(outputs):
                continue

            loss = ccc_loss(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        except Exception:
            continue

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description='Train EmotionMLP with CCC Loss')
    parser.add_argument('--train_dir', default='../../Data/Training', help='Path to Training data directory')
    parser.add_argument('--test_dir', default='../../Data/PrivateTest', help='Path to PrivateTest data directory')
    parser.add_argument('-o', '--output_dir', default='./outputs', help='Root directory to save model weights and logs')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--early_stop', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--feature_mode', default='ab', choices=['a', 'b', 'ab'], help='a=featureA only, b=featureB only, ab=fused')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of training data to use for validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, Seed: {args.seed}")

    # Load data
    train_features, train_labels = load_directory(args.train_dir, feature_mode=args.feature_mode)
    print(f"Loaded training data: {train_features.shape[0]} samples, {train_features.shape[1]} features")

    test_features, test_labels = load_directory(args.test_dir, feature_mode=args.feature_mode)
    print(f"Loaded test data: {test_features.shape[0]} samples, {test_features.shape[1]} features")

    # Split train into train / val
    full_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    n_total = len(full_dataset)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train/Val split: {n_train} / {n_val} samples")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model config
    if args.feature_mode == 'a':
        input_dim, hidden_dim, num_blocks = 768, 256, 3
        model_name = 'emotion_mlp_ablation_A'
    elif args.feature_mode == 'b':
        input_dim, hidden_dim, num_blocks = 4, 128, 2
        model_name = 'emotion_mlp_ablation_B'
    else:
        input_dim, hidden_dim, num_blocks = 772, 256, 3
        model_name = 'emotion_mlp_best'

    # Create experiment subdirectory
    exp_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment dir: {exp_dir}")

    model = EmotionMLP(input_dim=input_dim, hidden_dim=hidden_dim, num_blocks=num_blocks, output_dim=2, dropout_p=0.2).to(device)
    print(f"Model [{args.feature_mode}] created with {sum(p.numel() for p in model.parameters())} parameters")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    log_records = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = val_epoch(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        log_records.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})

        # Learning rate scheduling based on VAL loss
        scheduler.step(val_loss)

        # Early stopping based on VAL loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_path = os.path.join(exp_dir, f'{model_name}_best.pth')
            torch.save(model.state_dict(), best_path)
            print(f"Best model saved to {best_path} (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs (best val: {best_val_loss:.4f})")

        if epochs_no_improve >= args.early_stop:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Save final model
    final_path = os.path.join(exp_dir, f'{model_name}_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

    # Save training log
    log_path = os.path.join(exp_dir, f'{model_name}_log.csv')
    pd.DataFrame(log_records).to_csv(log_path, index=False)
    print(f"Training log saved to {log_path}")


if __name__ == '__main__':
    main()
