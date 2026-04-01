from dataset import FabVisDataset
from collections import Counter

test_dataset = FabVisDataset("FabVisDataset/test/images", "FabVisDataset/test/labels")
all_labels = []
for _, t in test_dataset:
    all_labels.extend(t["labels"].tolist())

print(Counter(all_labels))