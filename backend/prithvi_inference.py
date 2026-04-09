"""
Burn Scar Inference using IBM NASA Geospatial Prithvi-100M-burn-scar checkpoint.

The checkpoint was saved by mmsegmentation and has keys:
  backbone.*   -> ViT encoder (MaskedAutoencoderViT)
  neck.*       -> ConvTransformerTokensToEmbeddingNeck
  decode_head.* / auxiliary_head.*  -> FCNHead (we only need decode_head)

We reconstruct the model using components from prithvi-pytorch then remap the
state dict so keys match.
"""

import os
import sys
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import rasterio
from typing import Tuple

# ── make prithvi_pytorch importable (no setup.py in the repo) ─────────────────
_PRITHVI_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'prithvi-pytorch')
if _PRITHVI_REPO not in sys.path:
    sys.path.insert(0, os.path.abspath(_PRITHVI_REPO))

from prithvi_pytorch.encoder import MaskedAutoencoderViT
from prithvi_pytorch.decoder import ConvTransformerTokensToEmbeddingNeck
from omegaconf import OmegaConf
from einops import rearrange


# Normalization stats from burn_scars_Prithvi_100M.py (model's training config)
# Values are in [0,1] (images were converted to float32 before these were applied)
MEANS = np.array([0.033349706741586264, 0.05701185520536176,
                  0.05889748132001316,  0.2323245113436119,
                  0.1972854853760658,   0.11944914225186566], dtype=np.float32)
STDS  = np.array([0.02269135568823774,  0.026807560223070237,
                  0.04004109844362779,  0.07791732423672691,
                  0.08708738838140137,  0.07241979477437814], dtype=np.float32)

IMG_SIZE   = 224
NUM_FRAMES = 1
EMBED_DIM  = 768
NUM_CLASSES = 2   # 0=Unburnt, 1=Burn scar


class PrithviBurnScarModel(nn.Module):
    """
    Mirrors the mmseg architecture exactly:
      backbone  : TemporalViTEncoder  (≈ MaskedAutoencoderViT)
      neck      : ConvTransformerTokensToEmbeddingNeck
      decode_head: FCNHead (1 conv + conv_seg)
    """
    def __init__(self, cfg_path: str):
        super().__init__()
        cfg = OmegaConf.load(cfg_path)
        # Force the settings we know from the burn-scar config
        cfg.model_args.num_frames  = NUM_FRAMES
        cfg.model_args.in_chans    = 6
        cfg.model_args.img_size    = IMG_SIZE
        cfg.model_args.embed_dim   = EMBED_DIM
        cfg.model_args.depth       = 12
        cfg.model_args.num_heads   = 12
        cfg.model_args.patch_size  = 16
        cfg.model_args.tubelet_size = 1

        self.backbone = MaskedAutoencoderViT(**cfg.model_args)

        # neck: fpn1 (upsample x4) + fpn2 (upsample x2)  -> output_embed_dim=768
        num_tokens = IMG_SIZE // 16   # 14
        self.neck = ConvTransformerTokensToEmbeddingNeck(
            embed_dim=EMBED_DIM * NUM_FRAMES,
            output_embed_dim=EMBED_DIM,  # matches mmseg config
            Hp=num_tokens,
            Wp=num_tokens,
            drop_cls_token=True,
        )

        # decode_head: FCNHead with 1 conv (768->256) + conv_seg (256->2)
        self.decode_convs = nn.Sequential(
            nn.Conv2d(EMBED_DIM, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.decode_seg = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, H, W)
        # Add temporal dimension required by MAE encoder
        B = x.shape[0]
        x = rearrange(x, 'b c h w -> b c () h w')  # (B,6,1,H,W)

        # --- Manual Backbone Forward (mmsegmentation style) ---
        # 1. Patch Embed
        x = self.backbone.patch_embed(x) # (B, num_patches, D)
        
        # 2. Append CLS token
        cls_token = self.backbone.cls_token
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, 1+num_patches, D)
        
        # 3. Add Position Embedding (Full)
        x = x + self.backbone.pos_embed
        
        # 4. Apply Transformer blocks
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)
        # -----------------------------------------------------

        # neck expects (B, 1+num_patches, D); it handles dropping cls inside
        neck_out = self.neck(x)               # (B, output_embed_dim, H', W')

        out = self.dropout(neck_out)
        out = self.decode_convs(out)
        out = self.decode_seg(out)            # (B, 2, H', W')

        # upsample back to input resolution
        out = nn.functional.interpolate(out, size=(IMG_SIZE, IMG_SIZE),
                                        mode='bilinear', align_corners=False)
        return out                            # (B, 2, 224, 224)


def _remap_state_dict(raw_sd: dict) -> dict:
    """
    Map mmseg checkpoint keys -> our PrithviBurnScarModel keys.

    backbone.*               -> backbone.*
    neck.fpn1/fpn2.*         -> neck.*
    decode_head.convs.0.*    -> decode_convs.0.*
    decode_head.conv_seg.*   -> decode_seg.*
    (auxiliary_head.*        -> dropped)
    """
    new_sd = {}
    for k, v in raw_sd.items():
        if k.startswith('auxiliary_head'):
            continue   # drop

        # backbone: direct mapping
        if k.startswith('backbone.'):
            new_sd[k] = v
            continue

        # neck: fpn1 -> neck.fpn1, fpn2 -> neck.fpn2
        if k.startswith('neck.'):
            new_sd[k] = v
            continue

        # decode_head.convs.0.conv.weight -> decode_convs.0.weight
        if k.startswith('decode_head.convs.'):
            # e.g. decode_head.convs.0.conv.weight  -> decode_convs.0.weight
            #      decode_head.convs.0.bn.weight    -> decode_convs.1.weight
            rest = k[len('decode_head.convs.'):]   # "0.conv.weight" or "0.bn.weight"
            parts = rest.split('.', 2)              # ["0", "conv", "weight"]
            idx = int(parts[0])
            sub = parts[1]  # "conv" or "bn"
            tail = parts[2]
            layer_idx = idx * 3 + (0 if sub == 'conv' else 1)   # conv=0, bn=1
            new_sd[f'decode_convs.{layer_idx}.{tail}'] = v
            continue

        # decode_head.conv_seg.* -> decode_seg.*
        if k.startswith('decode_head.conv_seg.'):
            tail = k[len('decode_head.conv_seg.'):]
            new_sd[f'decode_seg.{tail}'] = v
            continue

    return new_sd


class BurnScarPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Use the base Prithvi config (burn scar config.yaml was empty)
        self.cfg_path = os.path.join(base_dir, '..', 'prithvi-pytorch',
                                     'tests', 'Prithvi_100M_config.yaml')
        self.ckpt_path = os.path.join(base_dir, 'weights',
                                      'burn_scars_Prithvi_100M.pth')
        self.model: PrithviBurnScarModel | None = None

    def load_model(self):
        if self.model is not None:
            return

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.ckpt_path}. "
                "Run download_weights.py first."
            )

        print("Building PrithviBurnScarModel …")
        model = PrithviBurnScarModel(cfg_path=self.cfg_path)

        print("Loading checkpoint …")
        raw = torch.load(self.ckpt_path, map_location='cpu', weights_only=False)
        raw_sd = raw['state_dict'] if 'state_dict' in raw else raw

        remapped = _remap_state_dict(raw_sd)
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        print(f"  Missing keys  : {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")

        model.to(self.device).eval()
        self.model = model
        print("Burn Scar model ready.")

    # ── pre-processing ────────────────────────────────────────────────────────
    def preprocess_tiff(self, file_path: str) -> Tuple[torch.Tensor, str]:
        """
        Read a multi-band GeoTIFF, normalize, return:
          - tensor  : (1, 6, 224, 224) float32
          - rgb_b64 : base64 PNG of the visible-light composite
        """
        with rasterio.open(file_path) as src:
            img_data = src.read().astype(np.float32)   # (bands, H, W)

        # Pad to 6 bands if needed
        if img_data.shape[0] < 6:
            pad = np.zeros((6 - img_data.shape[0], *img_data.shape[1:]),
                           dtype=np.float32)
            img_data = np.concatenate([img_data, pad], axis=0)

        # Clip nodata
        img_data = np.where(img_data == -9999, 0, img_data)

        # Resize each band to 224×224
        resized = np.zeros((6, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        for i in range(6):
            resized[i] = cv2.resize(img_data[i], (IMG_SIZE, IMG_SIZE),
                                    interpolation=cv2.INTER_LINEAR)

        # ── build RGB preview BEFORE normalising ─────────────────────────────
        # HLS/Sentinel band order: B02=Blue, B03=Green, B04=Red (indices 0,1,2)
        rgb = np.stack([resized[2], resized[1], resized[0]], axis=-1)  # R,G,B
        p2, p98 = np.nanpercentile(rgb[rgb > 0], 2), np.nanpercentile(rgb, 98)
        rgb_display = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1) * 255
        rgb_display = rgb_display.astype(np.uint8)
        _, buf = cv2.imencode('.png', cv2.cvtColor(rgb_display, cv2.COLOR_RGB2BGR))
        rgb_b64 = base64.b64encode(buf).decode('utf-8')

        # ── normalise for the model ────────────────────────────────────────────
        # HLS data from IBM dataset is already in reflectance [0, ~0.5] range.
        # Do NOT divide by 10000 — the training means/stds match this scale.
        for i in range(6):
            resized[i] = (resized[i] - MEANS[i]) / (STDS[i] + 1e-8)

        tensor = torch.from_numpy(resized).unsqueeze(0)  # (1,6,224,224)
        return tensor, rgb_b64

    # ── inference ─────────────────────────────────────────────────────────────
    def predict(self, file_path: str) -> Tuple[str, str]:
        self.load_model()
        tensor, rgb_b64 = self.preprocess_tiff(file_path)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)          # (1, 2, 224, 224)

        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (224,224)

        # Orange semi-transparent overlay for burn scars
        mask_rgba = np.zeros((IMG_SIZE, IMG_SIZE, 4), dtype=np.uint8)
        mask_rgba[pred == 1] = [0, 100, 255, 200]   # BGR orange + strong alpha

        _, buf_mask = cv2.imencode('.png', mask_rgba)
        mask_b64 = base64.b64encode(buf_mask).decode('utf-8')

        burned_pct = float((pred == 1).sum()) / pred.size * 100
        print(f"Burn scar coverage: {burned_pct:.1f}%")

        return mask_b64, rgb_b64


# Singleton loaded on first request (lazy) to keep startup fast
burn_scar_predictor = BurnScarPredictor()
