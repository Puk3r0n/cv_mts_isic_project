import numpy as np


def binary_counts(scores, positive, tau):
    pred = scores >= tau
    pos = positive.astype(bool)
    return {
        "tp": int(np.sum(pred & pos)),
        "fp": int(np.sum(pred & ~pos)),
        "fn": int(np.sum(~pred & pos)),
        "tn": int(np.sum(~pred & ~pos)),
    }


def rates(counts):
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    specificity = tn / max(1, tn + fp)
    fpr = fp / max(1, fp + tn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"recall": recall, "precision": precision, "specificity": specificity, "fpr": fpr, "f1": f1}


def uniform_risk(counts, cost_fp, cost_fn):
    return cost_fp * counts["fp"] + cost_fn * counts["fn"]


def threshold_grid(step=0.05):
    n = int(round(1.0 / step)) + 1
    return [round(i * step, 4) for i in range(n)]


def sweep(scores, positive, taus, cost_fp=1, cost_fn=10):
    rows = []
    for tau in taus:
        counts = binary_counts(scores, positive, tau)
        rows.append({"threshold": tau, **counts, **rates(counts), "risk": uniform_risk(counts, cost_fp, cost_fn)})
    return rows


def reliability_bins(confidences, correct, n_bins=10):
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = max(1, len(confidences))
    ece = 0.0
    bins = []
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        mask = (confidences >= low) & (confidences < high if i < n_bins - 1 else confidences <= high)
        count = int(np.sum(mask))
        if count == 0:
            bins.append({"low": float(low), "high": float(high), "count": 0, "acc": None, "conf": None})
            continue
        acc = float(np.mean(correct[mask]))
        conf = float(np.mean(confidences[mask]))
        ece += (count / total) * abs(acc - conf)
        bins.append({"low": float(low), "high": float(high), "count": count, "acc": acc, "conf": conf})
    return {"n_bins": n_bins, "ece": float(ece), "bins": bins}


def macro_f1_from_confusion(confusion):
    confusion = np.asarray(confusion, dtype=float)
    f1s = []
    for c in range(confusion.shape[0]):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        precision = tp / max(1e-9, tp + fp)
        recall = tp / max(1e-9, tp + fn)
        f1s.append(2 * precision * recall / max(1e-9, precision + recall))
    return float(np.mean(f1s))
