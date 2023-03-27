from typing import List, Dict
from PIL import Image

from aesthetic.laion import LaionAesthetic

from common import LAION_AESTHETIC_MODELS_PATH, LaionAestheticModelType

predictors: Dict[str, LaionAesthetic] = {}

class Predictor():
    def __init__(self, model_name: LaionAestheticModelType):
        self.predictor = LaionAesthetic(model_name, model_path=LAION_AESTHETIC_MODELS_PATH)

    def predict(self, image_path: str) -> List[float]:
        return self.predictor.get_score(image_path)
    
    def unload(self):
        self.predictor.unload()

class LaionAestheticPredictor():
    global predictors
    predictors = {
        "sac+logos+ava1-l14-linearMSE": Predictor(
            "sac+logos+ava1-l14-linearMSE.pth"
        ),
        "ava+logos-l14-linearMSE": Predictor(
            "ava+logos-l14-linearMSE.pth"
        ),
    }

    def predict(self, model_name: LaionAestheticModelType, image: Image) -> float:
        return predictors[model_name].predict(image)

def unload_laion_aesthetic_predictor():
    global predictors
    for predictor in predictors.values():
        predictor.unload()
    