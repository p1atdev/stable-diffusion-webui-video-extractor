from pathlib import Path
import torch
import torch.nn as nn
import clip
import os
from PIL import Image

from common import LaionAestheticModelType

class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

class LaionAesthetic():
    def __init__(self, model_name: LaionAestheticModelType = "sac+logos+ava1-l14-linearMSE.pth", model_path = "./models") -> None:
        if not Path(model_path).exists():
            os.makedirs(model_path)
        state_name = Path(model_path) / model_name
        if not Path(state_name).exists():
            url = f"https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/{model_name}?raw=true"
            import requests
            r = requests.get(url)
            with open(state_name, "wb") as f:
                f.write(r.content)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load the model you trained previously or the model available in this repo
        pt_state = torch.load(state_name)

        # CLIP embedding dim is 768 for CLIP ViT L 14
        self.predictor = AestheticPredictor(768)
        self.predictor.load_state_dict(pt_state)
        self.predictor.to(self.device)
        self.predictor.eval()

        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)


    def get_image_features(self, image: Image):
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            # l2 normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().detach().numpy()
        return image_features

    def get_score(self, image: Image):
        image_features = self.get_image_features(image)
        score = self.predictor(torch.from_numpy(image_features).to(self.device).float())
        return score.item()
    
    def unload(self):
        del self.predictor
        del self.clip_model
        del self.clip_preprocess
        torch.cuda.empty_cache()