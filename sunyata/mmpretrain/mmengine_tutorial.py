# %%
from mmpretrain.registry import MODELS
# %%
from mmengine.config import Config
# %%
Config(dict(
    _base_ = 'mmpretrain::resnet/resnet50_8xb32_in1k.py'
))
# %%
cfg = Config.fromfile('sunyata/mmpretrain/configs/resnet50.py')
# %%
cfg['model']
# %%
model = MODELS.build(cfg['model'])
# %%
model
# %%
from mmengine.analysis import get_model_complexity_info
# %%
analysis_results = get_model_complexity_info(model, input_shape=(3,224,224))
# %%
analysis_results.keys()
# %%
print(analysis_results['out_table'])
# %%
