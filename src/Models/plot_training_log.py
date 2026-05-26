import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_log(log_path, save_path=None):
    df = pd.read_csv(log_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CCC Loss (1 - CCC)')
    ax.set_title('Training Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark best val epoch
    best_idx = df['val_loss'].idxmin()
    best_epoch = df.loc[best_idx, 'epoch']
    best_val = df.loc[best_idx, 'val_loss']
    ax.axvline(best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best Val Epoch={int(best_epoch)}')
    ax.scatter([best_epoch], [best_val], color='red', zorder=5, s=50)

    plt.tight_layout()
    if save_path is None:
        save_path = log_path.replace('.csv', '.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True, help='Path to *_log.csv')
    parser.add_argument('--out', default=None, help='Output image path')
    args = parser.parse_args()
    plot_log(args.log, args.out)


if __name__ == '__main__':
    main()
