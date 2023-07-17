# %%
from typing import Any, Callable, List, Optional, Tuple, Type, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from sunyata.pytorch.layer.attention import Attention
from einops import rearrange, repeat

# %%
class BayesResNet(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )

        avgpool1 = nn.AdaptiveAvgPool2d((2, 4))
        avgpool2 = nn.AdaptiveAvgPool2d((2, 2))
        avgpool3 = nn.AdaptiveAvgPool2d((2, 1))
        self.avgpools = nn.ModuleList([
            avgpool1,
            avgpool2, 
            avgpool3,
            self.avgpool,
        ])
        log_prior = torch.zeros(1, num_classes)
        self.register_buffer('log_prior', log_prior)
        self.logits_bias = Parameter(torch.zeros(1, num_classes))

    def _forward_impl(self, x: Tensor) -> Tensor:
        batch_size, _, _, _ = x.shape
        log_prior = self.log_prior.repeat(batch_size, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer in enumerate([
            self.layer1, self.layer2,
            self.layer3, self.layer4
        ]):
            for block in layer:
                x = block(x)
                logits = self.avgpools[i](x)
                logits = torch.flatten(logits, start_dim=1)
                logits = self.fc(logits)
                log_prior = log_prior + logits
                log_prior = log_prior - torch.mean(log_prior, dim=-1, keepdim=True) + self.logits_bias
                log_prior = F.log_softmax(log_prior, dim=-1)
        return log_prior

# %%
class BayesResNet2(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )

        expansion = block.expansion
        self.digups = nn.ModuleList([
            *[nn.Sequential(
                nn.Conv2d(64 * i * expansion, 2048, kernel_size=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                ) for i in (1, 2, 4) 
                ],
            nn.Sequential(
                self.avgpool,
                nn.Flatten(),
                # self.fc,
            )
        ])

        log_prior = torch.zeros(1, 2048)
        self.register_buffer('log_prior', log_prior)
        self.logits_layer_norm = nn.LayerNorm(2048)
        # self.logits_bias = Parameter(torch.zeros(1, num_classes), requires_grad=True)

    def _forward_impl(self, x: Tensor) -> Tensor:
        batch_size, _, _, _ = x.shape
        log_prior = self.log_prior.repeat(batch_size, 1)
        # log_priors = torch.empty(0)

        x = self.conv1(x)
        x = self.bn1(x)
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
                # log_priors = torch.cat([log_priors, log_prior])
                # log_prior = log_prior - torch.mean(log_prior, dim=-1, keepdim=True) + self.logits_bias
                # log_prior = F.log_softmax(log_prior, dim=-1)
        return self.fc(log_prior)
# %%
class BayesResNet3(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )

        expansion = block.expansion
        self.digups = nn.ModuleList([
            *[Attention(
                query_dim=2048,
                context_dim=64 * i * expansion,
                heads=1,
                dim_head=2048,
            ) for i in (1, 2, 4) 
                ],
            nn.Sequential(
                self.avgpool,
                nn.Flatten(),
                # self.fc,
            )
        ])

        log_prior = torch.zeros(1, 2048)
        self.register_buffer('log_prior', log_prior)
        self.logits_layer_norm = nn.LayerNorm(2048)
        # self.logits_bias = Parameter(torch.zeros(1, num_classes), requires_grad=True)

    def _forward_impl(self, x: Tensor) -> Tensor:
        batch_size, _, _, _ = x.shape
        log_prior = repeat(self.log_prior, 'n d -> b n d', b = batch_size)
        # log_priors = torch.empty(0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer in enumerate([
            self.layer1, self.layer2,
            self.layer3
        ]):
            for block in layer:
                x = block(x)
                input = x.permute(0, 2, 3, 1)
                input = rearrange(input, 'b ... d -> b (...) d')
                log_prior = log_prior + self.digups[i](log_prior, input)
                log_prior = self.logits_layer_norm(log_prior)
                # log_priors = torch.cat([log_priors, log_prior])
                # log_prior = log_prior - torch.mean(log_prior, dim=-1, keepdim=True) + self.logits_bias
                # log_prior = F.log_softmax(log_prior, dim=-1)
        log_prior = log_prior.flatten(1)
        for block in self.layer4:
            x = block(x)
            logits = self.digups[-1](x)
            log_prior = log_prior + logits
            log_prior = self.logits_layer_norm(log_prior)
        
        return self.fc(log_prior)

# %%
class ResNet2(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )


    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        multi_logits = torch.empty(0)
        for block in self.layer4:
            x = block(x)
            logits = self.avgpool(x)
            logits = torch.flatten(logits, 1)
            logits = self.fc(logits)
            multi_logits = torch.cat([multi_logits, logits])
        return multi_logits

# %%
model = BayesResNet3(Bottleneck, [3, 4, 6, 3])
input = torch.randn(2, 3, 224, 224)
output = model(input)
# %%
