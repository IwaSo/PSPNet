# coding=utf-8
import os
import numpy as np
from PIL import Image
import colorsys
from scipy.misc import imresize


def load_datapath(dataset, mode):
    """
    load data path from .txt file
    dataset: str, 'ips' or 'melanoma' + '_1' to '_5'
    output: (list, list), path list of train and test data
    mode: str, 'train' or 'test', decide to read which dataset
    """
    # dataset のパス指定して.txtからファイルパス読み込み
    if mode == "train":
        img_txt = "train_data" + dataset[-1] + ".txt"
        mask_txt = "train_mask" + dataset[-1] + ".txt"
    elif mode == "test":
        img_txt = "test_data" + dataset[-1] + ".txt"
        mask_txt = "test_mask" + dataset[-1] + ".txt"
    else:
        raise ValueError("mode must be 'train' or 'test'")
    img_txt = os.path.join("data", "ips", "dataset", img_txt)
    mask_txt = os.path.join("data", "ips", "dataset", mask_txt)
    img_list = []
    mask_list = []
    for line in open(img_txt, "r"):
        img_list.append(line.strip())
    for line in open(mask_txt, "r"):
        mask_list.append(line.strip())
    img_list.sort()
    mask_list.sort()
    return img_list, mask_list


def getFilelist(path, ext):
    """
    get files path which have specified extension as a list recursiveliy.
    path: str, directory path you want to search
    ext: str, extension

    output: list, the components are file path
    """
    t = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                t.append(os.path.join(root, file))
    return t


def make_fcn_input(in_w, in_h, num_classes, dataset, resize_input, mode):
    """
    make input ant label data for fcn model,
    in_w: int, input width of model
    in_h: int, input height of model
    num_classes: int,
    dataset: str, "ips_1" etc
    """
    img_list, mask_list = load_datapath(dataset, mode=mode)
    DL = Patch_DataLoader(img_list, mask_list)

    X = np.zeros((len(img_list), in_h, in_w, 3)).astype(np.float32)
    y = np.zeros((len(img_list), in_h, in_w)) + num_classes
    y = y.astype(np.int32)
    n = 0
    for im, ma in zip(img_list, mask_list):
        im = Image.open(im)
        ma = Image.open(ma)
        if resize_input:
            im = im.resize((in_w, in_h))
            ma = ma.resize((in_w, in_h))
        img = np.array(im, dtype=np.float32) / 255.
        mask = np.array(ma, dtype=np.int32)
        mask = DL.image2label(mask)
        if in_h > img.shape[0]:
            offset = (in_h - img.shape[0]) // 2
            X[n, offset:offset + img.shape[0], :, :] = img[...]
            y[n, offset:offset + mask.shape[0], :] = mask[...]
        elif in_w > img.shape[1]:
            offset = (in_w - img.shape[1]) // 2
            X[n, :, offset:offset + img.shape[1], :] = img[...]
            y[n, :, offset:offset + mask.shape[1]] = mask[...]
        elif in_h == img.shape[0] and in_w == img.shape[1]:
            X[n, ...] = img[...]
            y[n, ...] = mask[...]
        n += 1
    y = y.reshape(
        y.shape[0], y.shape[1], y.shape[2], 1
    )
    return X, y


class Patch_DataLoader(object):
    label_d = {'ips': (0, 1, 2, 3), 'melanoma': (0, 1)}
    dist_method = ["regression", "ce_dist", "hamming", "sigmoid"]
    # in train phase
    # ips: good -> 0, bad -> 1, bgd -> 2, others -> 3
    # melanoma: background -> 0, tumor -> 1
    # in evaluate
    # ips: good -> 1, bad -> 2, bgd -> 3, others(oor) -> 0
    # melanoma: background -> 1, tumor -> 2
    # oor is out of region of evaluation

    def __init__(self,
                 img_list,
                 mask_list,
                 in_size=0,
                 size=0,
                 step=0,
                 method=None,
                 resolution=None,
                 mode="train",
                 threshold=0.2,
                 border_weight=None):
        """
        img_list: list, image path list, path is str
        mask_list: list,
        in_size: int,
        size: int,
        step: int,
        method: str,
        resolution: list of int,
        """
        # img と mask を一致させるためにファイル名でソート
        img_list.sort()
        mask_list.sort()
        self.img_list = img_list
        self.mask_list = mask_list
        self.in_size = in_size
        self.size = size
        self.step = step
        self.res = resolution
        self.method = method
        self.mode = mode
        self.threshold = threshold
        self.border_weight = border_weight
        if self.method == "hamming":
            self.binary = True
        else:
            self.binary = False
        # datasetを特定
        if 'ips' in img_list[0]:
            self.datatype = 'ips'
            self.num_classes = 3
        else:
            self.datatype = 'melanoma'
            self.num_classes = 2
        if self.in_size != 0:
            self.num_samples = self.count_samples()

    def count_samples(self):
        """
        count sample patches.(for fcn training)
        output: int, number of samples
        """
        num_samples = 0
        size = self.size
        step = self.step
        for mask_path in self.mask_list:
            mask = np.array(Image.open(mask_path))
            mask = self.image2label(mask)
            h, w = mask.shape
            if "ips" in self.datatype:
                for i in range((h - size) // step + 1):
                    for j in range((w - size) // step + 1):
                        m_patch = mask[i * step:(i * step) + size,
                                       j * step:(j * step) + size]
                        t, _ = self.calcTarget(m_patch)
                        if not isinstance(t, bool):
                            num_samples += 1
            else:
                row = (h - size) // step + 1
                column = (w - size) // step + 1
                num_samples += row * column
        return num_samples

    def load_data_two_target(self):
        """
        load dataset from image patch and mask path list.
        get two target, fcn and distribution(regression target)

        output: (array, array, array), X_train, fcn_target, dist_target
        """
        X = []
        fcn_y = []
        dist_y = []
        weights = []
        for img_path, mask_path in zip(self.img_list, self.mask_list):
            img_vecs, targets, s_weight = self.crop_img(img_path, mask_path)
            targets = list(zip(*targets))
            X += img_vecs
            fcn_y += list(targets[0])
            dist_y += list(targets[1])
            weights += s_weight
        X = np.asarray(X)
        fcn_y = np.asarray(fcn_y)
        dist_y = np.asarray(dist_y)
        weights = np.asarray(weights)
        if self.border_weight is not None:
            return X, fcn_y, dist_y, weights
        else:
            return X, fcn_y, dist_y

    def load_data(self):
        """
        load dataset from image path and mask path list.

        output: (array, array), X_train, y_train
                !!! channels last !!!
                the shape is (num_samples, img_dim), (num_samples, target_dim)
        """
        X = []
        y = []
        weights = []
        for img_path, mask_path in zip(self.img_list, self.mask_list):
            # img, mask を一つずつ読み込んでcrop
            img_vecs, targets, s_weight = self.crop_img(img_path, mask_path)
            X += img_vecs
            y += targets
            weights += s_weight
        X = np.asarray(X)
        y = np.asarray(y)
        weights = np.asarray(weights)
        if self.border_weight is not None:
            return X, y, weights
        else:
            return X, y

    def crop_img(self, img_path, mask_path, to_array=False):
        """
        crop patches from original images and masks
        img_path: str, path to image
        mask_path: str or None, path to mask

        output: (list, list), first one is list of patch vectors
                              second is target vectors
                if mask_path is None, output is (list, None)
        In case of to_array = True,
        output: (array, array), np.asarray method is used.
        """
        # 可読性のためsize, stepはローカル変数にしておく
        size = self.size
        step = self.step
        # imgとmaskが一致しているか確認
        img = img_path.split("/")[-1]
        img, _ = os.path.splitext(img)
        mask = mask_path.split("/")[-1]
        mask, _ = os.path.splitext(mask)
        assert img == mask, "file names are different"

        # img, mask読み込み
        # (row, column, channels)
        img = np.array(Image.open(img_path), dtype=np.float32)
        mask = np.array(Image.open(mask_path), dtype=np.int32)
        mask = self.image2label(mask)
        h, w, c = img.shape
        # test用に元画像のサイズを取得しておく
        self.height = h
        self.width = w

        img_vecs = []
        target_list = []
        s_weight = []
        for i in range((h - size) // step + 1):
            for j in range((w - size) // step + 1):
                patch = img[i * step:(i * step) + size,
                            j * step:(j * step) + size, :]
                m_patch = mask[i * step:(i * step) + size,
                               j * step:(j * step) + size]
                # deciding target from tmp
                target = self.calcTarget(m_patch)
                target, border = target
                if isinstance(target, bool):
                    continue
                elif isinstance(target, list):
                    if isinstance(target[0], bool):
                        continue
                else:
                    # targetに値が返っていれば、出力リストに加える
                    if self.in_size != size:
                        # リサイズ
                        patch = self.patch_resize(patch)
                    img_vecs.append(patch.flatten())
                    target_list.append(target)
                    if border:
                        s_weight.append(self.border_weight)
                    else:
                        s_weight.append(1.0)
        if to_array:
            return np.asarray(img_vecs), np.asarray(target_list)
        else:
            return img_vecs, target_list, s_weight

    def image2label(self, mask, evaluate=False):
        """
        convert mask image(.png) to label
        mask: array, mask image
             in ips data, mask shape is (row, column, channels)
             in melanoma, (row, column)

        in default, ips dataset,
        good -> 0, bad -> 1, bgd -> 3, others -> 4
        in melanoma dataset,
        background -> 0, tumor -> 1

        if evaluate = True, in ips, 'others' label is set to 0,
        and screening less than 100 pixel to 0
        good -> 1, bad -> 2, bgd -> 3, others -> 0
        in melanoma, background -> 1, tumor -> 2

        output: matrix array, shape is (row, column), 2 dims
        """
        # 各チャネルを0 or 1 のバイナリにする
        mask_bin = mask // 255
        if self.datatype == 'ips':
            # ips dataset
            # good -> 0, bad -> 1, bgd -> 2, others -> 3　とする
            img_label = mask_bin[:, :, 0] * 1 + \
                mask_bin[:, :, 1] * 2 + mask_bin[:, :, 2] * 3
            # よくわからないラベルを全てothers へ screening
            img_label[img_label == 0] = 4
            img_label[img_label > 3] = 4
            if evaluate:
                img_label[img_label == 4] = 0
                good = (img_label == 1).sum()
                bad = (img_label == 2).sum()
                bgd = (img_label == 3).sum()
                hist = np.array([good, bad, bgd])
                if np.min(hist) < 100:
                    label = np.argmin(hist) + 1
                    img_label[img_label == label] = 0
            else:
                img_label = img_label - 1
        else:
            # melanoma dataset
            if evaluate:
                img_label = mask_bin + 1
            else:
                img_label = mask_bin
        return img_label

    def calcTarget(self, m_patch):
        """
        return the target vector, for classification, regression or fcn.
        m_patch: matrix array, patch of mask converted to label,
                 the shape is (size, size)
        output: vector array or np.int64 or False(bool),
                target vector or one class label (in classification)
                return False if patch is filled by 'others' label
        """
        if self.method in self.dist_method:
            target = self.calcRegTarget(m_patch)
            if not isinstance(target, bool):
                if self.binary:
                    target = self.to_binary(target)
        elif self.method == "classification":
            h, w = m_patch.shape
            label = m_patch[h // 2, w // 2]
            if label == 3 and self.mode == "train":
                target = False
            else:
                target = np.int64(label)
        else:
            if self.datatype == 'ips':
                # ips dataset
                # others が パッチの大部分を占めていた場合、そのパッチはTraining には使わない
                hist = self.class_label_hist(m_patch)
                n = int(m_patch.size * self.threshold)
                if hist[-1] > n and self.mode == "train":
                    return False, False
            m_patch = self.patch_resize(m_patch)
            target = m_patch.flatten()

        # others を除いたラベルでborder判定
        hist = self.class_label_hist(m_patch)
        m = int(hist[:-1].sum() * (1 - self.threshold))
        if hist[:-1].max() < m:
            border = True
        else:
            border = False
        return target, border

    def to_binary(self, target):
        """
        transform target vector to binary
        target: the output of calcRegTarget
        """
        # reshapeしてargmax調べる
        target = target.reshape(-1, self.num_classes)
        max_idx = np.argmax(target, axis=1)
        h, w = target.shape

        # others ラベルを調べる
        s = np.sum(target, axis=1)
        zero_idx = (s < 0.1)

        # argmaxを1に，それ以外を0に
        # othersラベルがあれば0, 0, 0に
        result = np.zeros((h, w))
        result[np.arange(h), max_idx] = 1.
        result[zero_idx, :] = 0.

        return result.flatten()

    def patch_resize(self, im):
        """
        resize patch image to in_size x in_size.
        im: rgb image array, (row, column, channels)

        output: rgb image array, (s, s, channels)
        """
        im = Image.fromarray(np.uint8(im))
        im = im.resize((self.in_size, self.in_size), Image.NEAREST)
        im = np.array(im, dtype=np.float32)
        return im

    def class_label_hist(self, m_patch):
        """
        count the number of class labels in patch, and return histograms

        m_patch: matrix array,
        data: str, now supported 'ips' and 'melanoma'

        output: vector array, the histogram of class labels,
                              !! dtype = float32 !!
                the last elements of hist is ignore label.
        """
        # label 辞書読み込み
        labels = self.label_d[self.datatype]
        hist = []
        for c in labels:
            hist.append((m_patch == c).sum())
        hist = np.asarray(hist)
        return hist.astype(np.float32)

    def calcRegTarget(self, m_patch):
        """
        calcurate target histograms, multi resolution

        m_patch: matrix array,
        output: vector array or None(ips), target histograms
                the length is \sum (res_int**2 * num_classes)
                in ips dataset, discard patches which is almost filled by
                'others'
        """
        result = []
        hist = self.class_label_hist(m_patch)
        if self.datatype == 'ips':
            # ips dataset
            # others が パッチの大部分を占めていた場合、そのパッチはTraining には使わない
            n = int(m_patch.size * self.threshold)
            if hist[-1] > n and self.mode == "train":
                return False
            # 中心ピクセルがothersだった場合、Trainingには使わない
            # h, w = m_patch.shape
            # if m_patch[h // 2, w // 2] == 3 and self.mode == "train":
            #     return False

        for res_int in self.res:
            # resolution を一つずつみてtarget histogramをつくる
            target = self.calcRegTarget_oneRes(m_patch, res_int)
            result += target
        result = np.asarray(result)
        return result

    def calcRegTarget_oneRes(self, m_patch, res_int):
        """
        calcurate one target histograms in one regression.

        m_patch: matrix array,
        res_int: int,

        output: !! list !!, target histograms
                length is (resolution**2 * num_classes)
        """
        result = []
        # patchのサイズからresolutionで割る範囲を決定する
        h, w = m_patch.shape
        sub_h, rest_h = divmod(h, res_int)
        sub_w, rest_w = divmod(w, res_int)

        for h_num in range(res_int):
            for w_num in range(res_int):
                # patchをresolutionによってさらに小さいパッチに分ける
                patch = m_patch[sub_h * h_num:sub_h * (h_num + 1),
                                sub_w * w_num:sub_w * (w_num + 1)]
                hist = self.class_label_hist(patch)
                if self.datatype == "ips":
                    # ips dataset
                    # method によって場合分け
                    # others のラベルを省いてヒストグラムを作る
                    n = int(patch.size * self.threshold)
                    if hist[-1] > n:
                        # othersが多ければ0とする
                        result.append([0., 0., 0.])
                        continue
                    if self.method in self.dist_method:
                        # histogram 正規化
                        hist = hist[:-1] / np.sum(hist[:-1])
                        result.append(hist)
                else:
                    # melanoma dataset
                    # histogram 正規化
                    hist = hist / np.sum(hist)
                    result.append(hist)

        result = np.asarray(result)
        result = result.flatten()
        assert result.size == res_int**2 * self.num_classes
        return list(result)


class ProbMapConstructer(object):
    """
    Inference Map(InfMap) is probabirity map of a whole image.
    Patch Map is inference Map of a local patch.
    Inference Image(InfImg) is visualized image of InfMap.
    """

    def __init__(self,
                 model_out,
                 size=0,
                 step=0,
                 origin_h=0,
                 origin_w=0,
                 label_map=False,
                 data=None,
                 resolution=None):
        """
        model_out: array, output of model
                   the shape is (num_samples, num_outdim)
                   num_outdim equal to...
                   classification -> num_classes
                   regression -> \sum res_int**2 * num_classes
                   if fcn, (nub_samples, in_size, in_size, num_classes)
        size: int, cropped size
        step: int,
        origin_h: int, height of original image
        origin_w: int, width of original image
        data: str, 'ips' or 'melanoma'
        """
        # self.prob = model_out
        self.size = size
        self.step = step
        self.h = origin_h
        self.w = origin_w
        self.res = resolution
        self.label_map = label_map
        self.datatype = data
        if data == 'ips':
            self.num_classes = 3
        elif data == "melanoma":
            self.num_classes = 2
        else:
            raise ValueError("data must be 'ips' or 'melanoma'")
        if self.size != 0:
            self.InfMap = self.construct_InferenceMap(model_out)
        else:
            self.InfMap = model_out

    def save_InfMap(self, model_path, img_name):
        """
        save inference map as image.
        model_path: str, path to model directory
        img_name: str, the image file name !! including extension !!
                       like 'E3230652.png' or 'ISIC_0000000.png', etc.
        """
        if isinstance(self.InfMap, list):
            for infmap, res_int in zip(self.InfMap, self.res):
                vis_img, label_img = self.get_InfImg(infmap)
                vis_img = Image.fromarray(np.uint8(vis_img))
                label_img = Image.fromarray(np.uint8(label_img))
                vis_img.save(
                    os.path.join(model_path, "vis" + str(res_int), img_name)
                )
                label_img.save(
                    os.path.join(model_path, "label" + str(res_int), img_name)
                )
        else:
            vis_img, label_img = self.get_InfImg(self.InfMap)
            vis_img = Image.fromarray(np.uint8(vis_img))
            label_img = Image.fromarray(np.uint8(label_img))
            vis_img.save(os.path.join(model_path, "vis", img_name))
            label_img.save(os.path.join(model_path, "label", img_name))

    def get_InfImg(self, infmap):
        """
        output: (array, array), visualized map and label image.
        """
        if self.num_classes == 3:
            # 未推定部分は0(黒)で表示するためのmaskを作成
            summation = np.sum(infmap, axis=2)
            tmp = (summation > 0.2) * 1
            mask = np.zeros(infmap.shape)
            for i in range(infmap.shape[-1]):
                mask[:, :, i] = tmp[:, :]
            # ips dataset
            # hsv値を確率から求める
            # vは1.で固定
            # 確率の最大値はそのままsaturationにする
            # argmaxによってhをr,g,bに割り当てる
            # 0(good)は赤、1(bad)は緑、2(bgd)は青で表示
            h = np.argmax(infmap, axis=2).astype(float)
            h[h == 0] = 0.
            h[h == 1] = 1. / 3.
            h[h == 2] = 2. / 3.
            if self.label_map:
                s = 1.0
            else:
                s = np.max(infmap, axis=2).astype(float)
            v = np.ones((infmap.shape[0], infmap.shape[1]))
            # colosysの関数を、ベクトル演算できるように再定義
            hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)
            vis_img = hsv_to_rgb(h, s, v)
            vis_img = np.asarray(vis_img).transpose(1, 2, 0) * 255.0
            vis_img = vis_img * mask

            # label_img 作成
            mask = (summation > 0.2) * 1
            # label_imgは
            # good -> 1, bad -> 2, bgd -> 3, oor -> 0となる
            label_img = np.argmax(infmap, axis=2) + 1
            label_img = label_img * mask

        else:
            # melanoma dataset
            # vis_imgとlabel_imgまとめて作成
            # label_img は
            # background -> 1, tumor -> 2　それ以外 -> 0となる
            summation = np.sum(infmap, axis=2)
            mask = (summation > 0.2) * 1
            vis_img = np.argmax(infmap, axis=2)
            label_img = vis_img + 1
            vis_img[vis_img == 1] = 255
            vis_img = vis_img * mask
            label_img = label_img * mask
        return vis_img, label_img

    def construct_InferenceMap(self, model_out):
        """
        construct inference probabirity maps, for test images.
        model_out: array, the output of model

        output: an array or list of arrays,
                the components are inference maps of original images
        """
        maps = self.construct_patchMaps(model_out)
        if isinstance(maps, list):
            result = []
            for m in maps:
                result.append(self.construct_oneInferenceMap(m))
        else:
            result = self.construct_oneInferenceMap(maps)
        return result

    def construct_oneInferenceMap(self, map_array):
        """
        construct inference probabirity map
        this method is part of 'construct_InferenceMap'
        map_array: array, array of patch map
                    the shape is (num_samples, size, size, num_classes)
        output: array, inference map, (origin_h, origin_w, num_classes)
        """
        # 可読性のためローカル変数に代入
        size = self.size
        step = self.step
        output = np.zeros((self.h, self.w, map_array.shape[-1]))
        counter = np.zeros((self.h, self.w))
        count_filter = np.ones((size, size))
        # フィルタを動かす範囲の計算
        # wがx軸(横方向), hがy軸(縦方向)
        w_axis = (self.w - size) // step + 1
        h_axis = (self.h - size) // step + 1
        p_num = 0
        for i in range(h_axis):
            for j in range(w_axis):
                output[i * step:i * step + size, j * step:j *
                       step + size, :] += map_array[p_num, :, :, :]
                counter[i * step:i * step + size, j *
                        step:j * step + size] += count_filter[:, :]
                p_num += 1
        # counterで平均を取る
        # 0除算対策のため0の場所に1を代入する
        counter[counter == 0] = 1
        for n in range(map_array.shape[-1]):
            output[:, :, n] /= counter[:, :]
        return output

    def construct_patchMaps(self, prob):
        """
        normalize probabirity, and make patch probabirity map.
        prob: array, the output  of model
        output: array, patch prob map, (num_samples, )
        """
        # (?, num_classes)に変換
        if len(prob.shape) == 4:
            flag = "fcn"
            self.res = [1]
            num_samples = prob.shape[0]
            out_size = prob.shape[1]
            prob = prob.reshape(-1, self.num_classes)
        elif prob.shape[1] > self.num_classes:
            flag = "regression"
            num_samples = prob.shape[0]
            num_local = prob.shape[1] // self.num_classes
            prob = prob.reshape(num_samples * num_local, self.num_classes)
        else:
            self.res = [1]
            flag = "other"

        prob = self.normalize_prob(prob)

        if flag == "fcn":
            prob = prob.reshape(
                num_samples, out_size, out_size, self.num_classes
            )
            prob_map = self.resampling_map(prob)
            return prob_map
        elif flag == "regression":
            prob = prob.reshape(num_samples, num_local * self.num_classes)
            prob_map = self.restore_Map_allRes(prob)
            return prob_map
        else:
            prob_map = self.restore_patchMap(prob)
            return prob_map

    def normalize_prob(self, reshaped_prob):
        """
        normalize prob,
        reshaped_prob: matrix array, the shape is (num_hist, num_classes)

        output: matrix array, normalized probs
        """
        # クラス推定値の値域を[0, 1)にするため，最小値が負のヒストグラムのみ最小値を引く
        min_axis = np.min(reshaped_prob, axis=1)
        min_bool = (min_axis < 0) * 1
        min_axis3 = np.zeros(reshaped_prob.shape)
        nb_classes = reshaped_prob.shape[1]
        for i in range(reshaped_prob.shape[1]):
            min_axis3[:, i] = min_axis * min_bool
        reshaped_prob = reshaped_prob - min_axis3

        # 各軸の和で正規化
        tmp = np.sum(reshaped_prob, axis=1)
        sum_axis = np.zeros(reshaped_prob.shape)
        for i in range(nb_classes):
            sum_axis[:, i] = tmp
        norm_prob = reshaped_prob / sum_axis
        return norm_prob

    def resampling_map(self, prob_map):
        """
        resize the output of cnn, to crop size
        prob_map: array, the shape is
                         (nb_samples, output_size, output_size, num_classes)
                    !! prob maps must be normalized !!

        output: array, the shape is (nb_sample, self.size, self.size, 3)
        """
        result = []
        prob_map *= 255.
        for n in range(prob_map.shape[0]):
            # サンプルごとにリサイズ
            # channelごとにリサイズして繋げる
            one_prob = np.zeros((self.size, self.size, self.num_classes))
            for c in range(self.num_classes):
                im = Image.fromarray(np.uint8(prob_map[n, :, :, c]))
                im = im.resize((self.size, self.size))
                im = np.array(im, dtype=np.float32)
                one_prob[:, :, c] = im[:, :] / 255.
            result.append(one_prob)
        result = np.stack(result, axis=0)
        return result

    def restore_Map_allRes(self, prob):
        """
        calcurate Map for each Res.
        prob: matrix array, (num_samples, num_dims)
        output: list or array,
                list of patchMaps, each comp is (samples, size, size, 3)
                In single resolution case, return one array.
        """
        output = []
        # skip でtmp_probを得るためのインデックスを調整
        skip = 0
        for count, res_int in enumerate(self.res):
            if count != 0:
                tmp_res = self.res[count - 1]
                skip += tmp_res**2 * self.num_classes
            # resolution一つずつtmp_prob に分割してpatchMapを得る
            tmp_prob = prob[:, skip:(res_int**2 * self.num_classes) + skip]
            output.append(self.restore_patchMap(tmp_prob))
        if len(output) == 1:
            output = output[0]
        return output

    def restore_patchMap(self, prob):
        """
        restore patch map from probabirity.
        this method is part of 'restore_Map_allRes'
        this method is also used for 'classification' prob map.
        prob: matrix array, (num_samples, num_dims)

        output: array, (num_samples, size, size, num_classes)
        """
        result = np.zeros(
            (prob.shape[0], self.size, self.size, self.num_classes)
        )
        tmp_map = np.zeros((self.size, self.size, self.num_classes))
        num_local = prob.shape[1] // self.num_classes
        res = int(np.sqrt(num_local))
        for index in range(prob.shape[0]):  # sample loop
            for y in range(res):            # local resolution loop axis_y
                for x in range(res):        # local resolution loop axis_x
                    tmp_map = self.make_tmpMap(prob, x, y, res, index, tmp_map)
            result[index, :, :, :] = tmp_map[:, :, :]
        return result

    def make_tmpMap(self, prob, x, y, res, index, tmp_map):
        """
        restore local area of map from probabirity.
        this method is part of 'restore_patchMap'
        prob: matrix array, (num_samples, num_dims)
        x: int, index of local area (horizontal axis)
        y: int, index of local area (vertical axis)
        res: int, resolution of patch map
        index: int, the sample index
        tmp_map: array,    !! pass by reference !!

        output: tmp_map
        """
        # sizeがresolutionで割り切れない場合，
        # patch端を一番近い確率値で埋める
        l_size, rem = divmod(self.size, res)
        t = res * y + x
        if rem > 0:
            if (x + 1) == res:
                tmp_map[y * l_size:(y + 1) * l_size, x * l_size:, :] \
                    = prob[index,
                           self.num_classes * t:self.num_classes * (t + 1)]
            elif (y + 1) == res:
                tmp_map[y * l_size:, x * l_size:(x + 1) * l_size, :] \
                    = prob[index,
                           self.num_classes * t:self.num_classes * (t + 1)]
            elif (x + 1) == res and (y + 1) == res:
                tmp_map[y * l_size:, x * l_size:, :] \
                    = prob[index,
                           self.num_classes * t:self.num_classes * (t + 1)]
            else:
                tmp_map[y * l_size:(y + 1) * l_size,
                        x * l_size:(x + 1) * l_size, :] \
                    = prob[index,
                           self.num_classes * t:self.num_classes * (t + 1)]
        else:
            tmp_map[y * l_size:(y + 1) * l_size,
                    x * l_size:(x + 1) * l_size, :] \
                = prob[index,
                       self.num_classes * t:self.num_classes * (t + 1)]

        return tmp_map
