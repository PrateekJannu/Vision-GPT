import torch.nn as nn
from Transformers.SiglipVisionTransformer import SiglipVisionTransformer

class SiglipVisionModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self,pixel_values):
        vision_model= self.vision_model(pixel_values=pixel_values)
        print("SiglipVisionModel: ",vision_model.shape)
        return vision_model