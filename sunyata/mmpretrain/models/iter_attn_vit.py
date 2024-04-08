# %%
from mmpretrain.models.backbones.deit3 import DeiT3, DeiT3TransformerEncoderLayer
from mmpretrain.models.backbones.vision_transformer import VisionTransformer
# %%
backbone=dict(
    arch='s',
    img_size=224,
    patch_size=16,
    drop_path_rate=0.05,
    with_cls_token=False,
    out_type='avg_featmap',
    )
# %%
deit3 = DeiT3(**backbone)
# %%
vit = VisionTransformer(
    out_type='avg_featmap',
    with_cls_token=False,
    )
# %%
