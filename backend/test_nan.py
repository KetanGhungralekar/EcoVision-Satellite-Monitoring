import numpy as np
import tensorflow as tf
from wildfire_spread_inference import _parse_tfrecord, _INPUT_NAMES, _FEATURE_NAMES, IMG_SHAPE, CLIP_MIN, CLIP_MAX, GLOBAL_MEAN, GLOBAL_STD

ds = tf.data.TFRecordDataset(["../Wildfire Spread Prediction/test_dataset/next_day_wildfire_spread_test_00.tfrecord"]).map(_parse_tfrecord)
for sample in ds:
    x_np = np.stack([sample[k].numpy() for k in _INPUT_NAMES], axis=0).astype(np.float32)
    print("Initial x_np has nan:", np.isnan(x_np).any())
    for i, name in enumerate(_INPUT_NAMES):
        if np.isnan(x_np[i]).any(): print(f"NaN in {name}")
    
    x_np = np.clip(x_np, CLIP_MIN[:, None, None], CLIP_MAX[:, None, None])
    print("After clip has nan:", np.isnan(x_np).any())
    
    # Log channels (6: pr, 10: population)
    for c in [6, 10]:
        x_np[c] = np.log1p(np.maximum(x_np[c], 0.0))
        
    print("After log1p has nan:", np.isnan(x_np).any())
    
    # engineer_features equivalent check
    import cv2
    eps = 1e-6
    wind_rad = np.deg2rad(x_np[1])
    sin_w = np.sin(wind_rad).astype(np.float32)
    cos_w = np.cos(wind_rad).astype(np.float32)
    temp_range = (x_np[4] - x_np[3]).astype(np.float32)
    
    prev_bin = (x_np[11] > 0).astype(np.uint8)
    if prev_bin.sum() > 0:
        dist = cv2.distanceTransform(1 - prev_bin, cv2.DIST_L2, 3)
        dist = (dist / (dist.max() + eps)).astype(np.float32)
    else:
        dist = np.ones((64, 64), dtype=np.float32)

    extra = np.stack([sin_w, cos_w, temp_range, dist], axis=0)
    print("Extra has nan:", np.isnan(extra).any())

    x_np_full = np.concatenate([x_np, extra], axis=0)

    mean16 = np.concatenate([GLOBAL_MEAN, np.zeros(4)])
    std16 = np.concatenate([GLOBAL_STD, np.ones(4)])
    print("Mean16 nan:", np.isnan(mean16).any(), "Std16 nan:", np.isnan(std16).any())
    
    x_np_fin = (x_np_full - mean16[:, None, None]) / (std16[:, None, None] + 1e-6)
    print("Fin has nan:", np.isnan(x_np_fin).any())

    if np.isnan(x_np_fin).any():
        for i in range(16):
            if np.isnan(x_np_fin[i]).any():
                print(f"Index {i} has NaN at final!")
    break
