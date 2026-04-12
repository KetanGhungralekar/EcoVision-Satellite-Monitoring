import os
import io
import cv2
import json
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Optional torch / TF imports inside so we don't crash if they are missing
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    import tensorflow as tf
except ImportError:
    torch = None
    tf = None

IMG_SHAPE = [64, 64]

_FEATURE_NAMES = [
    'elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph',
    'pr', 'pdsi', 'NDVI', 'erc', 'population',
    'PrevFireMask', 'FireMask',
]
_INPUT_NAMES = _FEATURE_NAMES[:-1]


def _parse_tfrecord(serialised):
    features_spec = {k: tf.io.FixedLenFeature(IMG_SHAPE, tf.float32) for k in _FEATURE_NAMES}
    return tf.io.parse_single_example(serialised, features_spec)


# ---------------------------------------------------------------------------
# Grid topology helper (shared between model definition and predictor)
# ---------------------------------------------------------------------------
_topology_cache = {}

def get_grid_topology(H, W, device):
    key = (H, W, str(device))
    if key in _topology_cache:
        return _topology_cache[key]
    src, dst = [], []
    for r in range(H):
        for c in range(W):
            node = r * W + c
            for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W:
                    src.append(node); dst.append(nr*W+nc)
    t = torch.tensor([src, dst], dtype=torch.long, device=device)
    _topology_cache[key] = t
    return t


# ---------------------------------------------------------------------------
# Model architecture – matches the training / inference notebook exactly
# ---------------------------------------------------------------------------
if torch:
    class SEBlock(nn.Module):
        """Squeeze-Excitation block."""
        def __init__(self, channels, reduction=8):
            super().__init__()
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // reduction, 1),
                nn.ReLU(),
                nn.Conv2d(channels // reduction, channels, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return x * self.fc(x)

    class ResBlock(nn.Module):
        """Residual Conv Block."""
        def __init__(self, in_c, out_c):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c)
            )
            self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.conv(x) + self.skip(x))


    class HybridFireGNN(nn.Module):
        def __init__(self, in_channels=12, hidden_dim=96, dropout=0.35):
            super().__init__()
            self.dropout = dropout

            # --- Encoder (DEEPER + RESIDUAL + SE) ---
            self.enc1 = nn.Sequential(
                ResBlock(in_channels, 48),
                SEBlock(48),
                nn.MaxPool2d(2)
            )
            self.enc2 = nn.Sequential(
                ResBlock(48, 96),
                SEBlock(96),
                nn.MaxPool2d(2)
            )
            self.enc3 = nn.Sequential(
                ResBlock(96, hidden_dim),
                SEBlock(hidden_dim),
                nn.MaxPool2d(2)
            )

            # --- Multi-scale context ---
            self.dilated = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4),
                nn.ReLU()
            )

            # --- DEEP GAT (5 layers + residual + dropout) ---
            self.gats = nn.ModuleList([
                GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
                for _ in range(5)
            ])

            # --- Decoder ---
            self.up1 = nn.ConvTranspose2d(hidden_dim, 96, 2, 2)
            self.conv_up1 = nn.Sequential(
                ResBlock(96 + 96, 96),
                nn.Dropout2d(dropout)
            )

            self.up2 = nn.ConvTranspose2d(96, 48, 2, 2)
            self.conv_up2 = nn.Sequential(
                ResBlock(48 + 48, 48),
                nn.Dropout2d(dropout)
            )

            self.up3 = nn.ConvTranspose2d(48, 32, 2, 2)

            self.final = nn.Sequential(
                nn.Conv2d(32, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, 1)
            )

        def forward(self, x):
            B, C, H, W = x.shape

            # --- Encoder ---
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)

            # --- Multi-scale context ---
            e3 = e3 + self.dilated(e3)

            # --- Grid -> Graph ---
            B, C_lat, H_lat, W_lat = e3.shape
            x_flat = e3.permute(0, 2, 3, 1).reshape(B * H_lat * W_lat, C_lat)

            single_edges = get_grid_topology(H_lat, W_lat, x.device)
            edge_indices = [single_edges + i * (H_lat * W_lat) for i in range(B)]
            batched_edges = torch.cat(edge_indices, dim=1)

            # --- Deep GAT with residual stacking ---
            g = x_flat
            for gat in self.gats:
                g = F.elu(gat(g, batched_edges)) + g

            # --- Graph -> Grid ---
            x_gnn = g.reshape(B, H_lat, W_lat, C_lat).permute(0, 3, 1, 2)

            # --- Decoder ---
            d1 = self.conv_up1(torch.cat([self.up1(x_gnn), e2], dim=1))
            d2 = self.conv_up2(torch.cat([self.up2(d1), e1], dim=1))
            out = self.final(self.up3(d2))

            return out


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------
class WildfireSpreadPredictor:
    # Fixed threshold from the validation set, as used in the inference notebook
    DEFAULT_THRESHOLD = 0.45

    def __init__(self, model_path):
        if not torch or not tf:
            raise ImportError(
                "PyTorch, TensorFlow, and PyTorch Geometric are required "
                "for Wildfire Spread Prediction."
            )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HybridFireGNN(in_channels=12, hidden_dim=96, dropout=0.35).to(self.device)

        # The checkpoint stores just the state_dict (no wrapper dict)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)

        # In case it *is* wrapped (e.g. from a training checkpoint)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        # Remove module. prefix if saved with DataParallel
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace('module.', '') if k.startswith('module.') else k] = v

        self.model.load_state_dict(cleaned)
        self.model.eval()

        self.threshold = self.DEFAULT_THRESHOLD

    # -----------------------------------------------------------------------
    def _normalise(self, x_np, y_np):
        """Per-sample z-score normalisation (matches notebook)."""
        x_tensor = torch.tensor(x_np, dtype=torch.float32)   # (12, 64, 64)
        mean = x_tensor.mean(dim=(1, 2), keepdim=True)
        std  = x_tensor.std(dim=(1, 2), keepdim=True)
        x_tensor = (x_tensor - mean) / (std + 1e-6)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
        return x_tensor.unsqueeze(0), y_tensor.unsqueeze(0)   # add batch dim

    # -----------------------------------------------------------------------
    def process_tfrecord(self, filepath):
        """Parse the first sample from a .tfrecord file."""
        ds = tf.data.TFRecordDataset([filepath]).map(_parse_tfrecord)
        for sample in ds:
            x_np = np.stack(
                [sample[k].numpy() for k in _INPUT_NAMES], axis=0
            ).astype(np.float32)                                # (12, 64, 64)
            y_np = sample['FireMask'].numpy()[None].astype(np.float32)  # (1, 64, 64)
            return self._normalise(x_np, y_np)

    # -----------------------------------------------------------------------
    def process_npy(self, filepath):
        """Load a .npy file (shape 13×64×64: 12 inputs + 1 target)."""
        data = np.load(filepath).astype(np.float32)             # (13, 64, 64)
        x_np = data[:-1, :, :]                                  # (12, 64, 64)
        y_np = data[-1:, :, :]                                  # (1, 64, 64)
        return self._normalise(x_np, y_np)

    # -----------------------------------------------------------------------
    def predict(self, filepath):
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.npy':
            x_b, y_b = self.process_npy(filepath)
        else:
            x_b, y_b = self.process_tfrecord(filepath)
        x_b = x_b.to(self.device)
        y_b = y_b.to(self.device)

        with torch.no_grad():
            logits = self.model(x_b)
            probs_b = torch.sigmoid(logits)

        # --- Binarise target the same way as the notebook: (y > 0) ---
        target = (y_b[0, 0].cpu().numpy() > 0).astype(float)

        # Previous-day fire mask (channel index 11 = PrevFireMask).
        # After z-score normalisation, >0 still picks out fire pixels.
        prev_fire = (x_b[0, 11].cpu().numpy() > 0).astype(float)

        prob_map  = probs_b[0, 0].cpu().numpy()
        pred_bin  = (prob_map > self.threshold).astype(float)

        # ----- Generate images -----
        def fig_to_base64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05, dpi=150)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return img_b64

        # Calculate AUPRC
        from sklearn.metrics import average_precision_score
        target_flat = target.flatten()
        prob_flat = prob_map.flatten()
        if target_flat.sum() == 0:
            auprc = 0.0
        else:
            auprc = average_precision_score(target_flat, prob_flat)

        # 1. Previous Day fire mask
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(prev_fire, cmap='Reds', vmin=0, vmax=1)
        ax.set_title('Input: Previous Day Fire', fontsize=10)
        ax.axis('off')
        prev_day_b64 = fig_to_base64(fig)

        # 2. Predicted Next Day Mask (binary, using threshold)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(pred_bin, cmap='Reds', vmin=0, vmax=1)
        ax.set_title(f'Predicted (thresh={self.threshold:.2f})', fontsize=10)
        ax.axis('off')
        pred_next_day_b64 = fig_to_base64(fig)

        # 3. Ground-truth next-day mask
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(target, cmap='Reds', vmin=0, vmax=1)
        ax.set_title('Ground Truth: Next Day', fontsize=10)
        ax.axis('off')
        ground_truth_b64 = fig_to_base64(fig)

        # 4. Probability heat-map
        fig, ax = plt.subplots(figsize=(4.6, 4))
        img = ax.imshow(prob_map, cmap='jet', vmin=0, vmax=1)
        ax.set_title('GNN Prediction (Prob)', fontsize=10)
        ax.axis('off')
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        prob_map_b64 = fig_to_base64(fig)

        # 5. Error map — continuous |prediction - ground_truth| (matches notebook)
        fig, ax = plt.subplots(figsize=(4.6, 4))
        error_map = np.abs(prob_map - target)
        im_err = ax.imshow(error_map, cmap='jet', vmin=0, vmax=1)
        ax.set_title('Error Map', fontsize=10)
        ax.axis('off')
        plt.colorbar(im_err, ax=ax, fraction=0.046, pad=0.04)
        error_map_b64 = fig_to_base64(fig)

        return {
            "prediction_text": "Wildfire Spread Predicted Successfully",
            "threshold": float(self.threshold),
            "auprc": float(auprc),
            "prev_day_base64": prev_day_b64,
            "pred_next_day_base64": pred_next_day_b64,
            "ground_truth_base64": ground_truth_b64,
            "probability_map_base64": prob_map_b64,
            "error_map_base64": error_map_b64,
        }


if __name__ == "__main__":
    predictor = WildfireSpreadPredictor(
        "../Wildfire Spread Prediction/model/best_model.pth"
    )
