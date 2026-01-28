# cal f1 from tp, fp, fn

tp = 6
fp = 87
fn = 1544

f1 = 2 * tp / (2 * tp + fp + fn)
print(f1)