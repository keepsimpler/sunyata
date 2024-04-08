# %%
import json

filename = 'notebooks/attn-log.txt'
# filename = 'notebooks/convnext-log.txt'

fp = open(filename)    

log = []


for line in fp:
    log.append(json.loads(line))

fp.close()
# %%
import pandas as pd
# %%
pd_attn_log = pd.DataFrame(log)
# %%
pd_attn_log['test_acc1'].plot()
# %%
pd_attn_log['test_acc1']
# %%
