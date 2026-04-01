import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from dataset import FabVisDataset
import numpy as np

TEST_IMG_PATH = "FabVisDataset/test/images"
TEST_LBL_PATH = "FabVisDataset/test/labels"
MODEL_PATH = "trained_models/FASTRCNN_model_20_epochs.pth"

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

    # NEW: for mAP — store per-class detections and GT counts
    class_detections = {}  # class_id -> list of (score, is_tp)
    class_gt_counts = {}  # class_id -> total number of GT boxes

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

                # NEW: count GT boxes per class
                for lbl in ground_truth_labels.tolist():
                    class_gt_counts[lbl] = class_gt_counts.get(lbl, 0) + 1

                matched_ground_truth_boxes = set()

                # Sort predictions by confidence level so that predictions with higher confidence gets priority
                sorted_indices = torch.argsort(prediction_scores, descending=True)

                for idx in sorted_indices:
                    score = prediction_scores[idx].item()
                    pred_box = prediction_boxes[idx]
                    pred_label = prediction_labels[idx].item()

                    # NEW: initialise list for this class if first time seen
                    if pred_label not in class_detections:
                        class_detections[pred_label] = []

                    # Find best matching GT box (no score filter here)
                    best_iou = 0
                    best_ground_truth_idx = -1

                    for j in range(len(ground_truth_boxes)):
                        if j in matched_ground_truth_boxes:
                            continue

                        iou = compute_iou(pred_box, ground_truth_boxes[j])
                        if iou > best_iou:
                            best_iou = iou
                            best_ground_truth_idx = j

                    is_tp = (best_ground_truth_idx != -1 and best_iou >= iou_threshold and pred_label == ground_truth_labels[best_ground_truth_idx].item())

                    # Always record for mAP (all scores)
                    class_detections[pred_label].append((score, is_tp))
                    if is_tp:
                        matched_ground_truth_boxes.add(best_ground_truth_idx)

                    # Only count TP/FP for precision/recall at confidence threshold
                    if score >= confidence_threshold:
                        if is_tp:
                            total_tp += 1
                        else:
                            total_fp += 1

                # If any ground truth boxes are left without matching after this then they result in false negatives
                total_fn += len(ground_truth_boxes) - len(matched_ground_truth_boxes)


    return total_tp, total_fp, total_fn, class_detections, class_gt_counts

def compute_precision_and_recall(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall

def compute_ap(detections, n_gt):
    """
        Compute Average Precision for a single class using 11-point interpolation.
        detections: list of (score, is_tp) for this class
        n_gt:       total number of GT boxes for this class
    """
    if n_gt == 0:
        return 0.0

        # Sort by descending score
    detections = sorted(detections, key=lambda x: -x[0])

    tp_cumsum = np.cumsum([1 if d[1] else 0 for d in detections])
    fp_cumsum = np.cumsum([0 if d[1] else 1 for d in detections])

    recalls = tp_cumsum / n_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # 11-point interpolation (standard PASCAL VOC method)
    ap = 0.0
    for threshold in np.linspace(0, 1, 11):
        precisions_at_recall = precisions[recalls >= threshold]
        ap += (precisions_at_recall.max() if len(precisions_at_recall) > 0 else 0.0)
    ap /= 11.0

    return ap

def compute_map(class_detections, class_gt_counts):
    """Average the per-class APs."""
    aps = []
    for cls_id, n_gt in class_gt_counts.items():
        detections = class_detections.get(cls_id, [])
        ap = compute_ap(detections, n_gt)
        print(f"  Class {cls_id:2d}: AP = {ap:.4f}  (GT boxes: {n_gt})")
        aps.append(ap)
    return sum(aps) / len(aps) if aps else 0.0

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

tp, fp, fn, class_detections, class_gt_counts = evaluate(model, test_loader, device)

precision, recall = compute_precision_and_recall(tp, fp, fn)
mAP = compute_map(class_detections, class_gt_counts)

print(f"\nTP: {tp}, FP: {fp}, FN: {fn}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"mAP@0.5   : {mAP:.4f}")








