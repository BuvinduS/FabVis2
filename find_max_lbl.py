from dataset import FabVisDataset
import os

IMG_PATH = "FabVisDataset/test/images"
LBL_PATH = "FabVisDataset/test/labels"

label_set = set()
train_dataset = FabVisDataset(IMG_PATH, LBL_PATH)

for _, t in train_dataset:
    label_set.update(t["labels"].tolist())

print("Unique labels:", sorted(label_set))
print("Max label:", max(label_set))


files_with_10 = []

for fname in os.listdir(LBL_PATH):
    with open(os.path.join(LBL_PATH, fname)) as f:
        for line in f:
            cls = int(line.split()[0])
            if cls == 10:
                files_with_10.append(fname)
                break

print("Files containing class 10:")
for f in files_with_10:
    print(f)