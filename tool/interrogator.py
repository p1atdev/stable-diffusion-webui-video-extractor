from typing import Dict
import numpy as np
from PIL import Image

# from tagger.tagger.interrogator import WaifuDiffusionInterrogator
from tagger.tagger import utils 

utils.refresh_interrogators()

class WD14Tagger():
    interrogator_names = ["wd14-vit-v2", "wd14-swinv2-v2"]

    def __init__(self, name: str) -> None:
        self.name = name

    def unload(self) -> bool:
        return utils.interrogators[self.name].unload()
    
    def predict(self, image: Image) -> bool:
        _, tags = utils.interrogators[self.name].interrogate(image)

        return tags

    def any_match(self, image: Image, checklist: Dict[str, float]) -> bool:
        _, tags = utils.interrogators[self.name].interrogate(image)

        print("Hit tags: ", [t for t in tags.keys() if tags[t] >= 0.35])

        for tag in checklist.keys():
            if tag not in tags.keys():
                continue
            if tags[tag] >= checklist[tag]:
                print("Matched: ", tag, ": ", "{:.2f}%".format(tags[tag] * 100))
                return True
            else:
                print("Not Matched: ", tag, ": ", "{:.2f}%".format(tags[tag] * 100))

        return False

def unload_wd14tagger():
    for interrogator in utils.interrogators:
        utils.interrogators[interrogator].unload()