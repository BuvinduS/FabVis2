import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from dataset import FabVisDataset

TEST_IMG_PATH = "FabVisDataset/test/images"
TEST_LBL_PATH = "FabVisDataset/test/labels"
MODEL_PATH = "trained_models/FASTRCNN_model_10_epochs.pth"

def collate_fn(batch):
    return tuple(zip(*batch))

def compute_iou(box1, box2):
    # box = [x1, y1, x2, y2]
    # Where (x1, y1) is the top left corner and (x2, y2) is the bottom right corner

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # intersection area
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    # areas of boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # union
    union = area1 + area2 - inter_area

    if union == 0:
        return 0

    return inter_area / union

def evaluate(model, dataloader, device, iou_threshold=0.5):
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images) # Get predictions

            confidence_threshold = 0.5

            for i in range(len(outputs)):
                predictions = outputs[i]
                ground_truth = targets[i]

                prediction_boxes = predictions['boxes'].cpu()
                prediction_labels = predictions['labels'].cpu()
                prediction_scores = predictions['scores'].cpu()

                ground_truth_boxes = ground_truth['boxes']
                ground_truth_labels = ground_truth['labels']

                matched_ground_truth_boxes = set()

                # Sort predictions by confidence level so that predictions with higher confidence gets priority
                sorted_indices = torch.argsort(prediction_scores, descending=True)

                for idx in sorted_indices:
                    if prediction_scores[idx] < confidence_threshold:
                        continue

                    pred_box = prediction_boxes[idx]
                    pred_label = prediction_labels[idx]

                    best_iou = 0
                    best_ground_truth_idx = -1

                    for j in range(len(ground_truth_boxes)):
                        if j in matched_ground_truth_boxes:
                            continue

                        iou = compute_iou(pred_box, ground_truth_boxes[j])
                        if iou > best_iou:
                            best_iou = iou
                            best_ground_truth_idx = j

                    # Check whether it meets the threshold and whether the label match
                    if best_ground_truth_idx != -1 and best_iou >= iou_threshold and pred_label == ground_truth_labels[best_ground_truth_idx]:
                        matched_ground_truth_boxes.add(best_ground_truth_idx)
                        total_tp += 1
                    else:
                        total_fp += 1

                # If any ground truth boxes are left without matching after this then they result in false negatives
                total_fn += len(ground_truth_boxes) - len(matched_ground_truth_boxes)


    return total_tp, total_fp, total_fn

def compute_precision_and_recall(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall


test_dataset = FabVisDataset(TEST_IMG_PATH, TEST_LBL_PATH)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Load model architecture
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
num_classes = 12
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load trained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

tp, fp, fn = evaluate(model, test_loader, device)

precision, recall = compute_precision_and_recall(tp, fp, fn)

print(f"TP: {tp}, FP: {fp}, FN: {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")








