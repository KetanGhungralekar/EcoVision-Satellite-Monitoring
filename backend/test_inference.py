import numpy as np
import torch
from wildfire_spread_inference import WildfireSpreadPredictor

import os
from huggingface_hub import hf_hub_download
from hf_router.core import TokenManager

token_mgr = TokenManager(tokens_file=os.path.join(os.path.dirname(__file__), "tokens.json"))
repo_id = "PrathamChawdhry/EcoVision-Wildfire-Spread"
token = token_mgr.get_token_for_repo(repo_id)

model_path = hf_hub_download(
    repo_id=repo_id,
    filename="best_model.pth",
    token=token
)
predictor = WildfireSpreadPredictor(model_path)
x_b, y_b = predictor.process_tfrecord("../Wildfire Spread Prediction/test_dataset/next_day_wildfire_spread_test_00.tfrecord")

print(f"x_b shape: {x_b.shape}, has nan: {torch.isnan(x_b).any().item()}, max: {x_b.max().item()}, min: {x_b.min().item()}")
print(f"y_b shape: {y_b.shape}, has nan: {torch.isnan(y_b).any().item()}, max: {y_b.max().item()}, min: {y_b.min().item()}")

x_b = x_b.to(predictor.device)
with torch.no_grad():
    out = predictor.model(x_b)
    probs_b = torch.sigmoid(out)

print(f"out has nan: {torch.isnan(out).any().item()}, probs_b mean: {probs_b.mean().item()}, probs max: {probs_b.max().item()}")
