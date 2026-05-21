import os
from datasets import load_dataset

dataset = load_dataset("tanganke/stanford_cars")

save_root = "stanford_cars_local"
os.makedirs(save_root, exist_ok=True)

for split in ["train", "test"]:
    split_dir = os.path.join(save_root, split)
    os.makedirs(split_dir, exist_ok=True)

    for i, sample in enumerate(dataset[split]):
        img = sample["image"]
        label = sample["label"]

        class_dir = os.path.join(split_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)

        img.save(os.path.join(class_dir, f"{i}.png"))