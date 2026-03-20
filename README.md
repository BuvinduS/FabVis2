# Evaluation metrics used

---

## 1) IoU

- Measures how well a predicted box matches a ground truth box, i.e. actual box indicating the defect
- Computed by $IoU = Area\ of\ Overlap\ / Area\ of\ Union$
- If $IoU\ = 1 \Rightarrow\ Perfect\ match$
- If $IoU\ \geq 0.5 \Rightarrow\ We\ consider\ as\ correct\ detection$
- If $IoU\ = 0 \Rightarrow\ No\ overlap$

## 2) True Positive (TP)

- A prediction is a **True Positive** if
  - Correct class
  - $IoU\ \geq threshold$ (we use 0.5)
  - Matched to a ground truth box

> Correctly detected

## 3) False Positive (FP)

- A prediction is a **False Positive** if
  - No matching ground truth ($IoU$ too low)
  - Wrong class
  - Duplicate detection of the same

> Predicted something that should not be counted

## 4) False Negative (FN)

- A False Negative happens when:
  - A ground truth object exists
  - But model failed to detect it

> Model missed an object/detection

## 5) Precision

$Precision = TP\ /(TP\ + FP)$

Out of all the detections, how many were correct?

> $High\ precision \rightarrow$ few false alarms <br>
> $Low\ precision \rightarrow$ Many incorrect detections

## 6) Recall

$Recall = TP\ /(TP\ + FN)$

Out of all real objects, how many did we detect

> $High\ recall \rightarrow$ detects most objects <br>
> $Low\ recall \rightarrow$ misses many objects

## 7) mAP (Mean Average Precision)

**Main evaluation metric in object detection**

$mAP\ = Mean\ of\ Average\ Precision\ (AP)\ across\ all\ classes$

$mAP\ = (AP_1 + AP_2 + \ ...\ + AP_n)\ /\ n\ $

Where n = number of classes and <br>
$AP = Area\ under\ the\ Precision-Recall\ curve\ $