# %%
import torch
import torch.nn as nn
from mmpretrain.models import ResNet, ConvNeXt
from mmpretrain.registry import MODELS

from sunyata.pytorch.layer.attention import Attention
from einops import rearrange, repeat

# %%
@MODELS.register_module()
class IterResNet(ResNet):
    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ],
                 drop_path_rate=0.0):
        super(IterResNet, self).__init__(
                 depth,
                 in_channels,
                 stem_channels,
                 base_channels,
                 expansion,
                 num_stages,
                 strides,
                 dilations,
                 out_indices,
                 style,
                 deep_stem,
                 avg_down,
                 frozen_stages,
                 conv_cfg,
                 norm_cfg,
                 norm_eval,
                 with_cp,
                 zero_init_residual,
                 init_cfg,
                 drop_path_rate,            
        )

        self.digups = nn.ModuleList([
            *[nn.Sequential(
                nn.Conv2d(64 * i * self.expansion, 2048, kernel_size=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                ) for i in (1, 2, 4) 
                ],
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                # self.fc,
            )
        ])

        log_prior = torch.zeros(1, 2048)
        self.register_buffer('log_prior', log_prior)
        self.logits_layer_norm = nn.LayerNorm(2048)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        log_prior = self.log_prior.repeat(batch_size, 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer in enumerate([
            self.layer1, self.layer2,
            self.layer3, self.layer4
        ]):
            for block in layer:
                x = block(x)
                logits = self.digups[i](x)
                log_prior = log_prior + logits
                log_prior = self.logits_layer_norm(log_prior)
        return (log_prior,)


# %%
@MODELS.register_module()
class IterConvNeXt(ConvNeXt):
    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        assert out_indices == -1 and gap_before_final_norm == True
        super().__init__(
                 arch,
                 in_channels,
                 stem_patch_size,
                 norm_cfg,
                 act_cfg,
                 linear_pw_conv,
                 use_grn,
                 drop_path_rate,
                 layer_scale_init_value,
                 out_indices,
                 frozen_stages,
                 gap_before_final_norm,
                 with_cp,
                 init_cfg,
        )

        self.digups = nn.ModuleList()
        for channel in self.channels:
            digup = Attention(
                query_dim=self.channels[-1],
                context_dim=channel,
                heads=1,
                dim_head=self.channels[-1],
            )
            self.digups.append(digup)

        self.features = nn.Parameter(torch.zeros(1, self.channels[-1]))
        self.iter_layer_norm = nn.LayerNorm(self.channels[-1])
        self.norm3 = None


    def forward(self, x):
        batch_size, _, _, _ = x.shape
        features = repeat(self.features, 'n d -> b n d', b = batch_size)

        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            input = x.permute(0, 2, 3, 1)
            input = rearrange(input, 'b ... d -> b (...) d')
            features = features + self.digups[i](features, input)
            features = self.iter_layer_norm(features)

            for layer in stage:
                x = layer(x)
                input = x.permute(0, 2, 3, 1)
                input = rearrange(input, 'b ... d -> b (...) d')
                features = features + self.digups[i](features, input)
                features = self.iter_layer_norm(features)

        return (features.flatten(1),)

# %%
model = IterResNet(depth=50)
# %%
model = ResNet(depth=50)

# model = IterConvNeXt()
# %%
# model = ConvNeXt(
#     arch='atto',
#     gap_before_final_norm=True,
#     out_indices=-1,
#     )
# %%
input = torch.randn(2, 3, 224, 224)
# %%
outputs = model(input)
# %%
for item in outputs:
    print(item.shape)
# %%
output = model(input)
# %%
