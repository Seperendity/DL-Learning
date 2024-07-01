import torch.nn as nn
from .vit import ImageEncoderViT, ViTForImageClassification

if __name__=="__main__":
    import time

    img_encoder = ImageEncoderViT(
        img_size = 224,
        patch_size = 16,
        in_chans = 3,
        patch_embed_dim = 192,
        depth = 12,
        num_heads = 3,
        mlp_ratio = 4,
        act_layer = nn.GELU,
        dropout = 0.1)
    model = ViTForImageClassification()