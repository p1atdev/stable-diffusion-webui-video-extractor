import os
from typing import Literal
from pathlib import Path

from modules import scripts, paths

TaggerProccessType = Literal["faster", "slower"]
TaggerModelType = Literal["wd14-vit-v2", "wd14-swinv2-v2"]

LaionAestheticModelType = Literal["sac+logos+ava1-l14-linearMSE", "ava+logos-l14-linearMSE"]

BlurryTags = ["blurry"]

LAION_AESTHETIC_MODELS_PATH = None

extensions_dir = Path(paths.script_path, "extensions")

if os.path.isdir(f'{extensions_dir}/stable-diffusion-webui-blip2-captioner'):
    LAION_AESTHETIC_MODELS_PATH = extensions_dir / "stable-diffusion-webui-blip2-captioner" / "models"
else:
    LAION_AESTHETIC_MODELS_PATH = Path(scripts.basedir(), "models")

