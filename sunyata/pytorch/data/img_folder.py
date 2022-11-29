import os, math, random

from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader


def find_classes(classes_file_path):
    with open(classes_file_path) as f:
        classes = list(map(lambda s: s.strip(), f.readlines()))
    
    classes = list(filter(lambda s: not s.startswith('#'), classes))
    classes.sort()
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    return cls_to_idx

def samples_by_cls(root_dir, cls_to_idx):
    samples_by_cls = []
    for cls, idx in cls_to_idx.items():
        cls_path = os.path.join(root_dir, cls)
        if os.path.isdir(cls_path):
            samples_by_cls.append(len(list(os.listdir(cls_path))))
    return samples_by_cls


def random_split(root_dir, cls_to_idx, ratio):
    samples = []

    for cls, idx in cls_to_idx.items():
        cls_path = os.path.join(root_dir, cls)
        if os.path.isdir(cls_path):
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                sample = (img_path, idx)
                samples.append(sample)
    random.seed(0)
    random.shuffle(samples)
    train_max_idx = math.floor(ratio * len(samples))
    train_samples = samples[:train_max_idx]
    val_samples = samples[train_max_idx:]
    return train_samples, val_samples


class VisionDatasetWrap(VisionDataset):
    def __init__(self, root_dir, samples, transform=None):
        super().__init__(root_dir, transform=transform)
        self.samples =samples
        self.loader = default_loader
        
    def __getitem__(self, idx: int):
        img_path, target = self.samples[idx]
        img = self.loader(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
    
    def __len__(self):
        return len(self.samples)