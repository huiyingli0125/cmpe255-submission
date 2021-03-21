| Experiement | Accuracy | Confusion Matrix | Comment |
|-------------|----------|------------------|---------|
| Baseline    | 0.6770833333333334 | [[114  16] [ 46  16]] |  |
| Solution 1   | 0.796875  | [[118  12] [ 27  35]] |  Add few more features, such as glucose and bp. They can increase the accuracy. At the same the time, TP and FN increase while FP and FN decrease. In other words, these two features affects the result positively. |
| Solution 2   | 0.6822916666666666  | [[115  15] [ 46  16]] |  Tuning parameter C, it will achieve best score when C=0.01. However, the accuracy only increase a little bit. Confusion matrix does not change a lot. |
| Solution 3   | 0.8020833333333334  | [[118  12] [ 26  36]] |  Solution3 is the combination of solution1 and solution2, changing features set as well as tuning  parameter. It achieves the best accuracy which is around 0.802. It also has a better confusion matrix, TP and FN increase while FP and FN decrease.|
