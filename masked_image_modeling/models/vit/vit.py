import torch
import torch.nn as nn

try:
    from .common import PatchEmbed, ViTBlock
except:
    from  common import PatchEmbed, ViTBlock

class ImageEncoderViT(nn.Moudle):
    def __init__(self,
                 img_size: int,
                 patch_size:int,
                 in_chans: int,
                 patch_embed_dim: int,
                 depth: int,
                 num_heads: int,
                 mlp_ratio: int,
                 act_layer: nn.GELU,
                 dropout: float = 0.,
                 ) -> None:
        super().__init__()
        #-------------------Basic parameters---------------
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_embed_dim = patch_embed_dim
        self.num_heads = num_heads
        self.num_patches = (img_size // patch_size) ** 2
        #----------------------Model -----------------------
        self.patch_embed = PatchEmbed(in_chans, patch_embed_dim, patch_size, stride=patch_size)
        self.pos_embed = nn.Paramter(torch.zeros(1, self.num_patches, patch_embed_dim))
        self.norm = nn.LayerNorm(patch_embed_dim)
        self.blocks = nn.MouduleList([
            ViTBlock(patch_embed_dim, num_heads, mlp_ratio, True, act_layer, dropout)
            for _ in range(depth)])
        self._init_weights()
    
    def _init_weights(self):
        pos_embed = self.get_posembed(self.pos_embed.shape[-1], int(self.num_patches**.5))
        self.pos_embed.data.copy_(pos_embed)

        # initialize nn.Linear and nn.LayerNorm
        for m in self.modules():           
            if isinstance(m, nn.Linear):
                # we use xavier_uniform following official JAX ViT:
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def get_posembed(self, embed_dim, grid_size, temperature=10000):
            scale = 2 * torch.pi
            grid_h, grid_w = grid_size, grid_size
            num_pos_feats = embed_dim // 2
            # get grid
            y_embed, x_embed = torch.meshgrid([torch.arange(grid_h, dtype=torch.float32),
                                            torch.arange(grid_w, dtype=torch.float32)])
            # normalize grid coords
            y_embed = y_embed / (grid_h + 1e-6) * scale
            x_embed = x_embed / (grid_w + 1e-6) * scale
        
            dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
            dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
            dim_t = temperature ** (2 * dim_t_)

            pos_x = torch.div(x_embed[..., None], dim_t)
            pos_y = torch.div(y_embed[..., None], dim_t)
            pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
            pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

            # [H, W, C] -> [N, C]
            pos_embed = torch.cat((pos_y, pos_x), dim=-1).view(-1, embed_dim)

            return pos_embed.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).permute(0, 2, 1, 3).contiguous()

        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)
        x = self.norm_layer(x)
        return x          

# ------------------------ Model Functions ------------------------
def build_vit(model_name="vit_t", img_size=224, patch_size=16, img_dim=3):
    if model_name == "vit_t":
        return ImageEncoderViT(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=img_dim,
                               patch_embed_dim=192,
                               depth=12,
                               num_heads=3,
                               mlp_ratio=4.0,
                               act_layer=nn.GELU,
                               dropout = 0.1)
    if model_name == "vit_s":
        return ImageEncoderViT(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=img_dim,
                               patch_embed_dim=384,
                               depth=12,
                               num_heads=6,
                               mlp_ratio=4.0,
                               act_layer=nn.GELU,
                               dropout = 0.1)
    
if __name__ == '__main__':
     import torch
     from thop import profile

     bs, c, h, w = 2, 3, 224, 224
     x = torch.randn(bs, c, h, w)
     patch_size = 16

     model = build_vit(patch_size=patch_size)

     outputs = model(x)

     # Compute FLOPs & Params
     print('==============================')
     model.eval()
     flops, params = profile(model, inputs=(x,), verbose= False)
     
     