# %%  grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# %%
import os, math, random
import json
import pandas as pd
from PIL import Image
import torch
import torchvision
from sunyata.pytorch.data.img_folder import find_classes, samples_by_cls, random_split, VisionDatasetWrap
from sunyata.pytorch.data.results import get_one_run_dataframe, list_hparams_yaml_filenames
from sunyata.pytorch.data.results import get_one_run_dataframe, list_hparams_yaml_filenames
from sunyata.pytorch.arch.foldnet import FoldNetCfg, FoldNetRepeat

import matplotlib.pyplot as plt

pd.set_option("display.max.columns", None)
# pd.set_option("display.max.rows", None)
pd.set_option("display.precision", 4)

#############Results################
# %% PlantVillage
root_dir = '.data/results/plant/'
df_version_392 = get_one_run_dataframe(root_dir + 'version_392.foldnet.yaml')
df_version_388 = get_one_run_dataframe(root_dir + 'version_388.foldnet.yaml')
df_version_387 = get_one_run_dataframe(root_dir + 'version_387.foldnet.yaml')
# %%
df_version_388['val_accuracy'].max(), df_version_392['val_accuracy'].max(), df_version_387['val_accuracy'].max()
# %% FGVC8
df_version_398 = get_one_run_dataframe(root_dir + 'version_398.foldnet.yaml')
df_version_406 = get_one_run_dataframe(root_dir + 'version_406.foldnet.yaml')
df_version_410 = get_one_run_dataframe(root_dir + 'version_410.foldnet.yaml')
df_version_411 = get_one_run_dataframe(root_dir + 'version_411.foldnet.yaml')
# %%
df_version_398['val_accuracy'].max(), df_version_406['val_accuracy'].max(), df_version_410['val_accuracy'].max()
# %% PDR2018
df_version_417 = get_one_run_dataframe(root_dir + 'version_417.foldnet.yaml')
df_version_418 = get_one_run_dataframe(root_dir + 'version_418.foldnet.yaml')
df_version_419 = get_one_run_dataframe(root_dir + 'version_419.foldnet.yaml')
df_version_421 = get_one_run_dataframe(root_dir + 'version_421.foldnet.yaml')
# %%
print(df_version_417['val_accuracy'].max(), df_version_418['val_accuracy'].max(), df_version_419['val_accuracy'].max(), df_version_421['val_accuracy'].max())
# %%
fig, ax = plt.subplots(figsize=(6,4), dpi= 200)

ax.plot('epoch', 'val_accuracy', data=df_version_421, linewidth=1.5)
ax.plot('epoch', 'train_accuracy', data=df_version_421, linewidth=1.5)
ax.set_xlim([0,100])
ax.set_xticks([0,20,40,60,80,100])
ax.set_ylim([0., 1.0])
ax.set_yticks([0,0.5,0.8,0.9,1.])
ax.set_yticklabels([0,50,80,90,100])
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")

linestyles = {'loosely dashed': (0, (5, 10))}  # matplotlib linestyle dict
ax.axhline(y=0.8, color='k', linestyle=linestyles['loosely dashed'], linewidth=0.5)
ax.axhline(y=0.9, color='k', linestyle=linestyles['loosely dashed'], linewidth=0.5)

leg = ax.legend(['Validation','Train'], loc=(0.7,0.02), frameon=False)  #" lower left"
for line in leg.get_lines():
    line.set_linewidth(3)

ax.set_xlabel("Epoch Num.")
ax.set_ylabel("Accuracy (%)")

# fig.show()

# %%
list_hparams_yaml_filenames(root_dir, 'foldnet')

############PDR2018#################
valid_root_dir = '.data/plant/PDR2018/valid/'
# %%
valid_idx_to_img_file = 'AgriculturalDisease_validation_annotations.json'
# %%
valid_idx_to_img = pd.read_json(valid_root_dir + valid_idx_to_img_file)
valid_samples = valid_idx_to_img[['image_id', 'disease_class']].values.tolist()
valid_samples = [ (valid_root_dir + "resized_256/" + img_fname, idx) for img_fname, idx in valid_samples]
valid_idx_to_img.groupby(['disease_class']).count()
# %%
for idx, row in valid_idx_to_img.iterrows():
    img_path = valid_root_dir + "images/" + row['image_id']
    img = Image.open(img_path)
    img_resized = torchvision.transforms.Resize((256,256))(img)
    img_resized_path = valid_root_dir + "resized_256/" + row['image_id']
    img_resized.save(img_resized_path)
    print(img_resized.size)
# %%
train_idx_to_img_file = "F:\\BaiduNetdiskDownload\\ai_challenger_pdr2018_trainingset_20181023\\AgriculturalDisease_trainingset\\AgriculturalDisease_train_annotations.json"
train_idx_to_img = pd.read_json(train_idx_to_img_file)
train_idx_to_img.groupby(['disease_class']).count()
# %%
for idx, row in train_idx_to_img.iterrows():
    img_path = "F:\\BaiduNetdiskDownload\\ai_challenger_pdr2018_trainingset_20181023\\AgriculturalDisease_trainingset\\images\\" + row['image_id']
    img = Image.open(img_path)
    img_resized = torchvision.transforms.Resize((256,256))(img)
    img_resized_path = "F:\\BaiduNetdiskDownload\\ai_challenger_pdr2018_trainingset_20181023\\AgriculturalDisease_trainingset\\resized_256\\" + row['image_id']
    img_resized.save(img_resized_path)
    print(img_resized.size)
############FGVC8#################
# %%
root_dir = ".data/plant/plant-pathology-2021-fgvc8/"
cls_to_img_file = 'train.csv'
# %%
cls_to_img = pd.read_csv(root_dir + cls_to_img_file)
cls_to_img['idx'] = pd.factorize(cls_to_img['labels'])[0]
# %%
ratio = 0.8
total_length = len(cls_to_img)
train_length = math.floor(ratio * total_length)
train_or_valid = [0] * train_length + [1] * (total_length - train_length)
random.shuffle(train_or_valid)
cls_to_img['train_or_valid'] = pd.Series(train_or_valid)
# %%
cls_to_img[cls_to_img['train_or_valid'] == 0][['image', 'idx']].values.tolist()
# %%
classes = cls_to_img['labels'].value_counts()
# %% classes chosen: 
classes_chosen = ["healthy", "scab", "frog_eye_leaf_spot", "rust"]
classes_chosen = [
    "complex", 'powdery_mildew',
    'scab frog_eye_leaf_spot',
    'scab frog_eye_leaf_spot complex',
    'frog_eye_leaf_spot complex',
    'rust frog_eye_leaf_spot',
    'rust complex',
    'powdery_mildew complex'
]
classes_chosen = pd.DataFrame(classes_chosen, columns=['labels']).reset_index()
# %%
cls_to_img_chosen = cls_to_img.merge(classes_chosen, on=['labels'])
# %%
samples = []
for idx, row in cls_to_img_chosen.iterrows():
    img_fname = row["image"]
    img_path = root_dir + "train_images/" + img_fname
    img = Image.open(img_path)
    img_resized = torchvision.transforms.Resize((256, 256))(img)
    img_resized_path = root_dir + 'resized_256/' + img_fname
    img_resized.save(img_resized_path)
    idx = row['index']
    print(idx)
    samples.append((img_fname, idx))
# %%
dataset = VisionDatasetWrap(root_dir, samples)
# %%
random_split()
# %%


############PlantVillage#################
# %% 
root_dir = '.data/plant/Plant_leave_diseases_dataset_without_augmentation/'

classes_file = "classes.txt"
classes_file_path = os.path.join(root_dir, classes_file)
cls_to_idx = find_classes(classes_file_path)

# samples_by_cls(root_dir, cls_to_idx)
# ratio = 0.8
# train_samples, val_samples = random_split(root_dir, cls_to_idx, ratio)
# train_samples[0]
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128,shuffle=False, num_workers=4)
train_samples_file = "train_samples.json"
val_samples_file = "val_samples.json"

with open(root_dir + train_samples_file) as fp:
    train_samples = json.load(fp)
train_samples = list(train_samples.items())

with open(root_dir + val_samples_file) as fp:
    val_samples = list(json.load(fp).items())

val_samples_new = []
for img, idx in val_samples:
    img_new = '/'.join(root_dir.split('/') + img.split('/')[-2:])
    val_samples_new.append((img_new, idx))

# %%
samples = list(train_samples) + list(val_samples)
df_samples = pd.DataFrame(samples, columns=['img', 'idx'])
df_cls_to_idx = pd.DataFrame(cls_to_idx.items(), columns=['cls', 'idx'])
df_cls_groupby = df_samples.merge(df_cls_to_idx,on=['idx']).groupby(['cls'])[['img']].count().reset_index()
df_cls_groupby[['crop', 'disease']] = pd.DataFrame(df_cls_groupby['cls'].str.split('___').tolist(), columns=['crop', 'disease'])
df_cls_groupby = df_cls_groupby.rename(columns={'img': 'count'})
df_cls_groupby.to_csv('cls_count.csv', index=False)
# %%
val_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

# %%
val_dataset = VisionDatasetWrap(root_dir, transform=val_transform, samples=val_samples_new)

# %%
ckpt_path = ".data/results/plant/version_387.epoch=98-step=33660.ckpt"
# %%
model = FoldNetRepeat.load_from_checkpoint(ckpt_path)
# %%
model.eval()
# %%
model.training
# %% grad-cam
img = Image.open(val_samples_new[8610][0])
input_tensor = val_transform(img).unsqueeze(0)
target_layers = [model.layers[-1].units[0]]
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
grayscale_cam = cam(input_tensor=input_tensor)
visualization = show_cam_on_image(np.array(img, dtype=np.float32)/255., grayscale_cam[0,:])
Image.fromarray(visualization)
# Image.fromarray(np.uint8(cm.gist_earth(grayscale_cam[0]) * 255))

# %%
actual = torch.empty(0)
predicted = torch.empty(0)
# %%
for i, (input, target) in enumerate(val_dataset):
    with torch.no_grad():
        input = input.unsqueeze(0)
        logits = model(input)
        one_predicted = logits.argmax(dim=1)
        predicted = torch.cat([predicted, one_predicted])
        target = torch.tensor([target])
        actual = torch.cat([actual, target])
        
        print(i, predicted.shape, actual.shape)

# %%
(predicted == actual).float().mean()

# %%
df = pd.DataFrame(predicted, columns=['predicted'])
df['actual'] = pd.Series(actual)
df['count'] = 1
df_groupby = df.groupby(['actual', 'predicted'])['count'].count().reset_index()
# %%  number of samples wrongly predicted
df_wrongly_predicted = df_groupby[df_groupby['actual'] != df_groupby['predicted']]
df_wrongly_predicted['count'].sum()
# %% the indexes of samples that are wrongly predicted
df_sample_indexes_wrong = df.reset_index().merge(df_wrongly_predicted, on=['actual', 'predicted'])
# %% the image files of wrongly predicted samples
images_wrongly_predicted = [val_samples_new[i] for i in list(df_sample_indexes_wrong["index"])]
# %%
pillow_images = [Image.open(img) for img, actual_label in images_wrongly_predicted]

# %%
display(*pillow_images)
# %%
for cls, idx in cls_to_idx.items():
    print(cls.split('___'))
# %%
idx_to_cls = {idx: cls for cls, idx in cls_to_idx.items()}

# %%
confusions = []
for idx, row in df_groupby.iterrows():
    actual = int(row["actual"])
    predicted = int(row["predicted"])
    count = int(row["count"])
    actual_cls = idx_to_cls[actual]
    predicted_cls = idx_to_cls[predicted]
    one_pair = {
        "actual": ["plant:"+actual_cls.split('___')[0]+":"+actual_cls.split('___')[1]],
        "observed": ["plant:"+predicted_cls.split('___')[0]+":"+predicted_cls.split('___')[1]],
        "count": count,
    }
    confusions.append(one_pair)
    print(row["actual"], row["predicted"], row["count"])
# %%
confusions
