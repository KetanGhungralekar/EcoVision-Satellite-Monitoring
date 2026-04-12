import numpy as np
import torch
from wildfire_spread_inference import WildfireSpreadPredictor

predictor = WildfireSpreadPredictor("../Wildfire Spread Prediction/model/best_model.pth")
x_b, y_b = predictor.process_tfrecord("../Wildfire Spread Prediction/test_dataset/next_day_wildfire_spread_test_00.tfrecord")

print(f"x_b shape: {x_b.shape}, has nan: {torch.isnan(x_b).any().item()}, max: {x_b.max().item()}, min: {x_b.min().item()}")
print(f"y_b shape: {y_b.shape}, has nan: {torch.isnan(y_b).any().item()}, max: {y_b.max().item()}, min: {y_b.min().item()}")

x_b = x_b.to(predictor.device)
with torch.no_grad():
    out = predictor.model(x_b)
    probs_b = torch.sigmoid(out)

print(f"out has nan: {torch.isnan(out).any().item()}, probs_b mean: {probs_b.mean().item()}, probs max: {probs_b.max().item()}")
