"""
Extract ALL samples from the large test tfrecord files into individual
.npy files (shape 13x64x64: 12 input channels + 1 target channel),
matching the format used by the training/inference notebook.
"""
import os
import numpy as np
import tensorflow as tf

IMG_SHAPE = [64, 64]

features_dict = {
    'elevation':    tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'th':           tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'vs':           tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'tmmn':         tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'tmmx':         tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'sph':          tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'pr':           tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'pdsi':         tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'NDVI':         tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'erc':          tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'population':   tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'PrevFireMask': tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
    'FireMask':     tf.io.FixedLenFeature(IMG_SHAPE, tf.float32),
}


def parse_tfrecord(example_proto):
    parsed = tf.io.parse_single_example(example_proto, features_dict)
    inputs = tf.stack([
        parsed['elevation'], parsed['th'], parsed['vs'],
        parsed['tmmn'], parsed['tmmx'], parsed['sph'],
        parsed['pr'], parsed['pdsi'], parsed['NDVI'],
        parsed['erc'], parsed['population'], parsed['PrevFireMask']
    ], axis=0)                                          # (12, 64, 64)
    target = tf.expand_dims(parsed['FireMask'], axis=0) # (1, 64, 64)
    return tf.concat([inputs, target], axis=0)           # (13, 64, 64)


TEST_DIR   = os.path.join(os.path.dirname(__file__), "..", "Wildfire Spread Prediction", "test_dataset")
OUTPUT_DIR = os.path.join(TEST_DIR, "samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

source_files = [
    os.path.join(TEST_DIR, "next_day_wildfire_spread_test_00.tfrecord"),
    os.path.join(TEST_DIR, "next_day_wildfire_spread_test_01.tfrecord"),
]

count = 0
for src in source_files:
    if not os.path.exists(src):
        print(f"Skipping {src} (not found)")
        continue

    ds = tf.data.TFRecordDataset([src]).map(parse_tfrecord)

    for data in ds:
        out_path = os.path.join(OUTPUT_DIR, f"sample_{count:04d}.npy")
        np.save(out_path, data.numpy())

        if count % 100 == 0:
            size_kb = os.path.getsize(out_path) / 1024
            print(f"  [{count}] {os.path.basename(out_path)}  ({size_kb:.1f} KB)")
        count += 1

print(f"\nDone - wrote {count} .npy sample files to {OUTPUT_DIR}")
