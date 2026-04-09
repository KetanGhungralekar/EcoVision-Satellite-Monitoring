import os
from huggingface_hub import list_repo_files, hf_hub_download

repo_id = "ibm-nasa-geospatial/Prithvi-100M-burn-scar"
files = list_repo_files(repo_id)
print("Files in repo:", files)

# Find config and weights
cfg_file = next((f for f in files if f.endswith('.yaml')), None)
ckpt_file = next((f for f in files if f.endswith('.pt') or f.endswith('.pth') or f.endswith('.bin')), None)

print(f"Config: {cfg_file}, Ckpt: {ckpt_file}")

if cfg_file and ckpt_file:
    print("Downloading...")
    os.makedirs("weights", exist_ok=True)
    hf_hub_download(repo_id=repo_id, filename=cfg_file, local_dir="weights")
    hf_hub_download(repo_id=repo_id, filename=ckpt_file, local_dir="weights")
    print("Downloaded to weights/")
else:
    print("Could not find required files!")
