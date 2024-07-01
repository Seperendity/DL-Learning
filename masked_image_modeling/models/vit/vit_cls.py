import torch.nn as nn

from .common import CrossAttention
from .vit    import ImageEncoderViT

class ViTForImageClassification(nn.Moudle):
    def __init__(self,
                 image_encoder  :ImageEncoderViT,
                 num_classes    :int = 1000,
                 qkv_bias       :bool = True,
                 ):
        super().__init__()
        self.encoder = image_encoder
        self.classifier = CrossAttention(image_encoder.patch_embed_dim,
                                         num_classes,
                                         qkv_bias,
                                         image_encoder.num_heads,
                                         num_queries = 1)
    def forward(self, x):
        x = self.encoder(x)
        x, x_cls = self.classifier(x)
        return x