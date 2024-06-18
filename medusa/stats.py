import numpy as np

def get_confusion_matrix_stats(tp, tn, fp, fn):
    """ This function returns a collection of statistics given a confusion
    matrix. For more information about these statistics, refer to
    https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values.

    Parameters
    -------------
    tp : int
        Number of true positives
    tn : int
        Number of true negatives
    fp : int
        Number of false positives
    fn : int
        Number of false negatives

    Returns
    --------------
    dict()
        Dictionary that contains the following statistics:
            - "prevalence"
            - "accuracy"
            - "ba", i.e. balanced accuracy
            - "ppv", i.e. positive predictive value
            - "precision", same as PPV
            - "fdr", i.e. false discovery rate
            - "f1", i.e. F1 score
            - "for", i.e. false omission rate
            - "npv", i.e. negative predictive value
            - "fm", i.e. Fowlkes-Mallows index
            - "informedness"
            - "bm", i.e. bookmaker informedness (informedness)
            - "tpr", i.e. true positive rate
            - "sensitivity", i.e. same as TPR
            - "recall", i.e. same as TPR
            - "fpr", i.e. false positive rate (fall-out or type I error)
            - "lr+", positive likelihood ratio
            - "mk", i.e. markedness (deltaP)
            - "mcc", i.e. Matthews correlation coefficient
            - "pt", i.e. prevalence threshold
            - "fnr", i.e. false negative rate (miss rate or type II error)
            - "tnr", i.e. true negative rate (selectivity)
            - "specificity", i.e. same as TNR
            - "lr-", i.e. negative likelihood ratio
            - "dor", i.e. diagnostic odds ratio
            - "ts", i.e. threat score (Jaccard index)
            - "csi", i.e. critical success index (same as TS)
    """
    stats = dict()
    pp = tp + fp
    pn = fn + tn
    p = tp + fn
    n = fp + tn
    total = tp + tn + fp + fn

    stats["prevalence"] = p / (p + n)
    stats["accuracy"] = (tp + tn) / total
    stats["ppv"] = tp / pp
    stats["precision"] = stats["ppv"]
    stats["fdr"] = fp / pp
    stats["f1"] = 2 * tp / (2 * tp + fp + fn)
    stats["for"] = fn / pn
    stats["npv"] = tn / pn
    stats["tpr"] = tp / (tp + fn)
    stats["sensitivity"] = stats["tpr"]
    stats["recall"] = stats["tpr"]
    stats["fnr"] = fn / (tp + fn)
    stats["fpr"] = fp / (fp + tn)
    stats["tnr"] = tn / (fp + tn)
    stats["specificity"] = tn / (fp + tn)
    stats["lr+"] = stats["tpr"] / stats["fpr"]
    stats["lr-"] = stats["fnr"] / stats["tnr"]
    stats["dor"] = stats["lr+"] / stats["lr-"]
    stats["mk"] = stats["ppv"] + stats["npv"] - 1
    stats["informedness"] = stats["tpr"] + stats["tnr"] - 1
    stats["bm"] = stats["informedness"]
    stats["pt"] = (np.sqrt(stats["tpr"] * stats["fpr"]) - stats["fpr"]) / (
        stats["tpr"] - stats["fpr"])
    stats["ba"] = (stats["tpr"] + stats["tnr"]) / 2
    stats["fm"] = np.sqrt(stats["ppv"] * stats["tpr"])
    stats["mcc"] = np.sqrt(stats["tpr"] + stats["tnr"] + stats["ppv"] +
                           stats["npv"])
    stats["csi"] = tp / (tp + fn + fp)
    stats["ts"] = stats["csi"]
    return stats