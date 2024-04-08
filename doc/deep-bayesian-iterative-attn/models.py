# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from einops import repeat, rearrange

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
# from block import Cross_Block, Cross_Attention


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2

class AttnDistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        
        scale = self.embed_dim ** -0.5 / 16
        print("scale=", scale)
        self.digup = Attention(
            query_dim=self.embed_dim,
            context_dim=self.embed_dim,
            heads=1,
            dim_head=self.embed_dim,
            scale=scale,
        )
        self.latent = nn.Parameter(torch.zeros(2, self.embed_dim))
        self.latent_layer_norm = nn.LayerNorm(self.embed_dim)
        del self.norm

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        latent = repeat(self.latent, 'n d -> b n d', b = B)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        latent = latent + self.digup(latent, x)
        latent = self.latent_layer_norm(latent)

        for blk in self.blocks:
            x = blk(x)
            
            latent = latent + self.digup(latent, x)
            latent = self.latent_layer_norm(latent)

        # x = self.norm(x)
        return latent[:, 0], latent[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2
        
        
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, scale=None, dropout=0.):

        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (inner_dim == query_dim and heads == 1)

        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5 if scale is None else scale # / 10.
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim) if project_out else nn.Identity()

    def forward(self, x, context = None):
        h = self.heads

        q = self.to_q(x)
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q,k,v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)
        return out

    
class AttnVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        scale = self.embed_dim ** -0.5 / 36
        print("scale=", scale)
        self.digup = Attention(
            query_dim=self.embed_dim,
            context_dim=self.embed_dim,
            heads=1,
            dim_head=self.embed_dim,
            scale=scale,
        )
        
        # self.latent = nn.Parameter(torch.zeros(1, self.embed_dim))
        latent = torch.zeros(1, self.embed_dim)
        self.register_buffer('latent', latent)
        self.latent_layer_norm = nn.LayerNorm(self.embed_dim)
        del self.norm
        
    def forward_features(self, x):
        B = x.shape[0]
        latent = repeat(self.latent, 'n d -> b n d', b = B)
        
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        latent = latent + self.digup(latent, x)
        latent = self.latent_layer_norm(latent)
        
        for blk in self.blocks:
            x = blk(x)
            
            latent = latent + self.digup(latent, x)
            latent = self.latent_layer_norm(latent)

        return latent
    
    def forward_head(self, latent):
        # latent = reduce(latent, 'b n d -> b d', 'mean')
        logits = nn.Flatten()(latent)
        logits = self.fc_norm(logits)
        logits = self.head_drop(logits)
        logits = self.head(logits)
        return logits
        
    def forward(self, x):
        latent = self.forward_features(x)
        logits = self.forward_head(latent)
        return logits
    
    
    

# @register_model
# def deit_tiny_patch16_224(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
#             map_location="cpu", check_hash=True
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model

@register_model
def attn_deit_tiny_patch16_224(pretrained_cfg=None, pretrained_cfg_overlay=None, pretrained=False, **kwargs):
    model = AttnVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    model.default_cfg = _cfg()
    
    return model


# @register_model
# def deit_small_patch16_224(pretrained_cfg=None, pretrained_cfg_overlay=None, pretrained=False, **kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
#             map_location="cpu", check_hash=True
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model


@register_model
def attn_deit_small_patch16_224(pretrained_cfg=None, pretrained_cfg_overlay=None, pretrained=False, **kwargs):
    model = AttnVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    
    return model


# @register_model
# def attn_deit_small_noshare_patch16_224(pretrained_cfg=None, pretrained_cfg_overlay=None, pretrained=False, **kwargs):
#     model = AttnNoShareVisionTransformer(
#         patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs, block_fn=Cross_Block)
#     model.default_cfg = _cfg()
    
#     return model


@register_model
def attn_deit_base_patch16_224(pretrained_cfg=None, pretrained_cfg_overlay=None, pretrained=False, **kwargs):
    model = AttnVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    
    return model


# @register_model
# def attn_deit_base_noshare_patch16_224(pretrained_cfg=None, pretrained_cfg_overlay=None, pretrained=False, **kwargs):
#     model = AttnNoShareVisionTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs, block_fn=Cross_Block)
#     model.default_cfg = _cfg()
    
#     return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def attn_deit_tiny_distilled_patch16_224(pretrained_cfg=None, pretrained_cfg_overlay=None, pretrained=False, **kwargs):
    model = AttnDistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
   
    return model

@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
