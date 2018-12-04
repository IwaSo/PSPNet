# coding=utf-8
import tools as tl
from tools import Patch_DataLoader
import os
import numpy as np
from PIL import Image
import sys
from sklearn.metrics import confusion_matrix


def evaluate_model(model, w_path="/home/sora/project/ips/PSPnet2/", mode="test"):
    """
    evaluate model with segmentation metrics.
    model: str, model name like 'regression/Adam/vgg_p4_size150'.
    data: str, ips or melanoma
    """
    # 評価指標のリストを用意して格納、表示
    jaccard = []
    dice = []
    tpr = []
    tnr = []
    accuracy = []
    class_j = []
    data = model.split("/")[0]
    for i in [1,2,3,4,5]:
        dataset = str(i)
        # one fold 評価
        j, d, tp, tn, acc, c_j = evaluate_one_fold(
            model, dataset, w_path, mode)
        jaccard.append(j)
        dice.append(d)
        tpr.append(tp)
        tnr.append(tn)
        accuracy.append(acc)
        class_j += list(c_j)
        print(dataset + " is done.")

    jaccard = np.asarray(jaccard)
    dice = np.asarray(dice)
    tpr = np.asarray(tpr)
    tnr = np.asarray(tnr)
    accuracy = np.asarray(accuracy)
    class_j = np.asarray(class_j)
    j_mean = np.mean(jaccard)
    j_std = np.std(jaccard)
    d_mean = np.mean(dice)
    d_std = np.std(dice)
    tp_mean = np.mean(tpr)
    tp_std = np.std(tpr)
    tn_mean = np.mean(tnr)
    tn_std = np.std(tnr)
    acc_mean = np.mean(accuracy)
    acc_std = np.std(accuracy)

    cj_nonzero = np.zeros((3,))
    for i in range(3):
        cj_nonzero[i] = np.count_nonzero(np.ceil(class_j[:, i]))

    cj_mean = np.sum(class_j, axis=0) / cj_nonzero
    cj_std = np.zeros((3,))
    for i in range(3):
        cj_tmp = class_j[np.nonzero(class_j[:, i]), i].reshape(-1)
        cj_var = np.sum(np.square(cj_tmp - cj_mean[i])) / cj_nonzero[i]
        cj_std[i] = np.sqrt(cj_var)
    print("model : ", model)
    print("jaccard index : ", j_mean, "+- ", j_std)
    print("dice : ", d_mean, "+- ", d_std)
    print("tpr : ", tp_mean, "+-", tp_std)
    print("tnr : ", tn_mean, "+-", tn_std)
    print("accuracy : ", acc_mean, "+-", acc_std)
    path = os.path.join(w_path, model)
    result = np.array([[j_mean, j_std],
                       [d_mean, d_std],
                       [tp_mean, tp_std],
                       [tn_mean, tn_std],
                       [acc_mean, acc_std],
                       ])
    cj_result = np.vstack((cj_mean, cj_std))

    np.savetxt(os.path.join(path, "seg_result.txt"), result)
    np.savetxt(os.path.join(path, "seg_result_class.txt"), cj_result)


def evaluate_one_fold(directory, dataset, w_path, mode):
    """
    evaluate the result of one fold, with segmentation metrics.
    directory: str, model directory path like 'regression/Adam/vgg_p4_size150'
    dataset: str, ips_ or melanoma_  1 to 5
    w_path: str, path to dataset directory, like 'weights/ips'

    output: tuple of float, segmentation scores
            (jaccard, dice, tpr, tnr, acc)
    """
    dname = dataset[-1]
    path = os.path.join(w_path, directory, dname)
    ld = os.listdir(path)
    if "label" in ld:
        # 単一resolutionの場合はそのままlabelディレクトリ読み込み
        pred_path = tl.getFilelist(
            os.path.join(path, "label"), ".png"
        )
    else:
        # multi resolutionの場合は、最も解像度が高いディレクトリを読み込み
        tmp = 0
        for d in ld:
            if "label" in d and int(d[-1]) > tmp:
                tmp = int(d[-1])
        di = "label" + str(tmp)
        pred_path = tl.getFilelist(
            os.path.join(path, di), ".png"
        )
    pred_path.sort()

    jaccard = []
    dice = []
    tpr = []
    tnr = []
    acc = []
    class_j = []

    # インスタンス化するために適当なパスを読み込み
    if "ips" in path:
        img_list, true_path = tl.load_datapath(dataset, mode=mode)
        labels = [1, 2, 3]
    else:
        img_list, true_path = tl.load_datapath(dataset, mode=mode)
        labels = [1, 2]
    DL = Patch_DataLoader(img_list, true_path)
    for pred, true in zip(pred_path, true_path):
        pred_name, _ = os.path.splitext(pred.split("/")[-1])
        true_name, _ = os.path.splitext(true.split("/")[-1])
        assert pred_name == true_name

        y_pred = np.array(Image.open(pred), int)
        y_true = np.array(Image.open(true), int)
        y_true = DL.image2label(y_true, evaluate=True)

        # out of region of evaluation
        oor = ~(y_true == 0) * 1
        y_pred = y_pred * oor
        j, d, tp, tn, a, c_j = evaluate_one_image(y_true, y_pred, labels)
        class_j.append(c_j)
        jaccard.append(j)
        dice.append(d)
        tpr.append(tp)
        tnr.append(tn)
        acc.append(a)
    jaccard = sum(jaccard) / len(jaccard)
    dice = sum(dice) / len(dice)
    tpr = sum(tpr) / len(tpr)
    tnr = sum(tnr) / len(tnr)
    acc = sum(acc) / len(acc)
    class_j = np.asarray(class_j)
    return jaccard, dice, tpr, tnr, acc, class_j


def evaluate_one_image(y_true, y_pred, labels):
    """
    evaluate segmentation result, using confusion_matrix
    y_true: 2d array,
    y_pred: 2d array,
    labels: the labels dataset contains,
            ips -> [1, 2, 3], melanoma -> [1, 2]
    oor is ignored.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mat = confusion_matrix(y_true, y_pred, labels=labels)
    jaccard = []
    dice = []
    tpr = []
    tnr = []
    acc = []
    class_j = np.zeros((3,))
    for i in range(len(labels)):
        if mat[i, :].sum() == 0:
            continue
        elif len(labels) == 2 and i == 0:
            continue
        tp = mat[i, i]
        tn = mat.sum() - (mat[i, :].sum() + mat[:, i].sum() - mat[i, i])
        fp = mat[:, i].sum() - mat[i, i]
        fn = mat[i, :].sum() - mat[i, i]
        jaccard.append(tp / float(tp + fp + fn))
        class_j[i] = (tp / float(tp + fp + fn))
        dice.append(2 * tp / float(2 * tp + fp + fn))
        tpr.append(tp / float(tp + fn))
        tnr.append(tn / float(fp + tn))
        acc.append((tp + tn) / float(tp + tn + fp + fn))

    jaccard = sum(jaccard) / len(jaccard)
    dice = sum(dice) / len(dice)
    tpr = sum(tpr) / len(tpr)
    tnr = sum(tnr) / len(tnr)
    acc = sum(acc) / len(acc)
    return jaccard, dice, tpr, tnr, acc, class_j


if __name__ == '__main__':
    model = sys.argv[1]
    evaluate_model(model, mode="test")
