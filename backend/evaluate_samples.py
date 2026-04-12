import os
import glob
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from collections import defaultdict
import warnings

# Suppress warnings from sklearn undefined metrics and PyTorch Geometric imports if any
warnings.filterwarnings('ignore')

from wildfire_spread_inference import WildfireSpreadPredictor

def evaluate_all_samples():
    print("Initializing predictor...")
    model_path = os.path.join(
        os.path.dirname(__file__), "..",
        "Wildfire Spread Prediction", "model", "best_model.pth"
    )
    predictor = WildfireSpreadPredictor(model_path)
    
    samples_dir = os.path.join(
        os.path.dirname(__file__), "..",
        "Wildfire Spread Prediction", "test_dataset", "samples"
    )
    
    npy_files = glob.glob(os.path.join(samples_dir, "*.npy"))
    if not npy_files:
        print(f"No .npy files found in {samples_dir}")
        return

    print(f"Found {len(npy_files)} samples. Evaluating...")
    
    results = []
    
    # Evaluate every sample
    for i, filepath in enumerate(npy_files):
        if i % 100 == 0 and i > 0:
            print(f"  Processed {i}/{len(npy_files)}")
            
        x_b, y_b = predictor.process_npy(filepath)
        x_b = x_b.to(predictor.device)
        y_b = y_b.to(predictor.device)
        
        with torch.no_grad():
            logits = predictor.model(x_b)
            probs_b = torch.sigmoid(logits)
            
        # target in [1, 64, 64] -> binarise
        target = (y_b[0, 0].cpu().numpy() > 0).astype(float)
        prob_map = probs_b[0, 0].cpu().numpy()
        
        target_flat = target.flatten()
        prob_flat = prob_map.flatten()
        
        gt_pixels = int(target_flat.sum())
        
        if gt_pixels == 0:
            auprc = 0.0
        else:
            auprc = average_precision_score(target_flat, prob_flat)
            
        results.append({
            "filename": os.path.basename(filepath),
            "gt_pixels": gt_pixels,
            "auprc": auprc
        })
        
    print(f"\nEvaluation complete for {len(results)} samples.")
    
    # Bin limits: 0-50, 51-100, 101-150, etc.
    # Group into bins
    bins = defaultdict(list)
    
    for r in results:
        p = r["gt_pixels"]
        if p == 0:
            bin_name = "0 (No Fire)"  # Separate bin for exactly 0
            bin_index = -1
        else:
            # 1-50 -> index 0, 51-100 -> index 1, etc.
            # formula: (p-1) // 50
            idx = (p - 1) // 50
            start = idx * 50 + 1
            end = start + 49
            bin_name = f"{start} - {end}"
            bin_index = idx
            
        bins[(bin_index, bin_name)].append(r)
        
    # Sort bins by their natural order
    sorted_bin_keys = sorted(bins.keys(), key=lambda x: x[0])
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS SORTED BY GT FIRE PIXELS AND AUPRC")
    print("="*60)
    
    for _, bin_name in sorted_bin_keys:
        samples_in_bin = bins[(_, bin_name)]
        # Sort by AUPRC descending
        samples_in_bin.sort(key=lambda x: x["auprc"], reverse=True)
        
        print(f"\n--- Bin: {bin_name} fire pixels ({len(samples_in_bin)} samples) ---")
        
        # Display top 10 and bottom 3 to not blow up terminal completely, or we can save to file.
        # Given "for every sample gives auprc... sort...", I will save the full report to a text file
        # and print a summary.
        for r in samples_in_bin[:10]: # showing top 10 in console
            print(f"  {r['filename']}: GT={r['gt_pixels']}, AUPRC={r['auprc']:.4f}")
            
        if len(samples_in_bin) > 10:
            print("  ...")
            
    # Save the full report to a text file for easy reading
    report_path = os.path.join(os.path.dirname(__file__), "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("Full Evaluation Report\n")
        for _, bin_name in sorted_bin_keys:
            samples_in_bin = bins[(_, bin_name)]
            f.write(f"\n--- Bin: {bin_name} fire pixels ({len(samples_in_bin)} samples) ---\n")
            for r in samples_in_bin:
                f.write(f"{r['filename']}: GT={r['gt_pixels']}, AUPRC={r['auprc']:.4f}\n")
                
    print(f"\nFull detailed report saved to {report_path}")

if __name__ == "__main__":
    evaluate_all_samples()
