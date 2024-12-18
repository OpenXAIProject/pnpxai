#%%
from pathlib import Path
from huggingface_hub import snapshot_download
repo_dir = snapshot_download("devilops/blended-diffusion-custom")
repo_path = Path(repo_dir)
# %%
import os
os.listdir(repo_path / 'guided_diffusion' / 'guided_diffusion')
# %%
import sys
sys.path.append(str(repo_path))
from guided_diffusion.guided_diffusion.script_util import create_model_and_diffusion