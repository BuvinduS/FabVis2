import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import FabVisDataset
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.optim as optim

TRAIN_IMG_PATH = "FabVisDataset/train/images"
TRAIN_LBL_PATH = "FabVisDataset/train/labels"

VALIDATION_IMG_PATH = "FabVisDataset/valid/images"
VALIDATION_LBL_PATH = "FabVisDataset/valid/labels"

def collate_fn(batch):
    return tuple(zip(*batch))

def validate (model, dataloader, device):
    model.train()
    val_loss = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            val_loss += losses.item()
            # val_loss  = val_loss / len(dataloader)

    print(f"Validation Loss: {val_loss:.4f}")


def train_model(model, dataloader, val_dataloader, num_epochs, device, optimizer):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            # epoch_loss = epoch_loss / len(dataloader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

        # validate training (basically training without learning (training without updates))
        validate(model, val_dataloader, device)

train_dataset = FabVisDataset(TRAIN_IMG_PATH, TRAIN_LBL_PATH)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True ,collate_fn=collate_fn)

validation_dataset = FabVisDataset(VALIDATION_IMG_PATH, VALIDATION_LBL_PATH)
val_Loader = DataLoader(validation_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Load the model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Replace the classifier head (VERY important):
num_classes = 12
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

images, targets = next(iter(train_loader))
print(images[0].shape)
print(targets[0])

train_model(model, train_loader, val_Loader, num_epochs=10, device=device, optimizer=optimizer)

torch.save(model.state_dict(), "trained_models/FASTRCNN_model_10_epochs.pth")