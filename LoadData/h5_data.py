"""H5 数据读取、数据集划分与 tf.data 构建。"""

import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def scan_h5_metadata(h5_paths):
    """扫描所有 H5 文件中的 file_name 和 label。"""
    all_file_names = []
    all_labels = []

    for fp in h5_paths:
        with h5py.File(fp, "r") as hf:
            fns = hf["file_name"][:].astype(str)
            ys = hf["label"][:].astype(np.int64)
            all_file_names.extend(fns.tolist())
            all_labels.extend(ys.tolist())

    file_names = np.asarray(all_file_names, dtype=str)
    labels = np.asarray(all_labels, dtype=np.int64)
    return file_names, labels


def stratified_split_by_file_name(
    file_names,
    labels,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    """按照标签分层，并以 file_name 为样本标识划分 train/val/test。"""
    fn_trainval, fn_test, y_trainval, y_test = train_test_split(
        file_names,
        labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )

    val_size_rel = val_ratio / (1.0 - test_ratio)
    fn_train, fn_val, y_train, y_val = train_test_split(
        fn_trainval,
        y_trainval,
        test_size=val_size_rel,
        random_state=seed,
        stratify=y_trainval,
    )

    return set(fn_train), set(fn_val), set(fn_test), y_train, y_val, y_test


def build_index_map_by_set(h5_paths, allowed_set):
    """为给定 file_name 集合构建每个 H5 文件对应的样本索引。"""
    index_map = []
    n_samples = 0
    allowed_list = list(allowed_set)

    for fp in h5_paths:
        with h5py.File(fp, "r") as hf:
            fns = hf["file_name"][:].astype(str)
            idx = np.where(np.isin(fns, allowed_list))[0]
            if idx.size > 0:
                index_map.append((fp, idx.astype(np.int64)))
                n_samples += int(idx.size)

    return index_map, n_samples


def get_file_names_from_index_map(index_map):
    """按 index_map 的实际遍历顺序返回每个样本的 file_name。
    """
    ordered_file_names = []

    for fp, idx_arr in index_map:
        with h5py.File(fp, "r") as hf:
            fns = hf["file_name"][:].astype(str)
            for i in idx_arr:
                ordered_file_names.append(str(fns[int(i)]))

    return np.asarray(ordered_file_names, dtype=str)


def make_dataset_from_index_map(
    index_map,
    batch_size=16,
    shuffle=True,
    shuffle_buffer=4096,
    prefetch=True,
    time_steps=4000,
    freq_bins=16,
    freq_channels=3,
):
    """根据索引映射构建双输入 tf.data.Dataset。"""

    def generator():
        for fp, idx_arr in index_map:
            with h5py.File(fp, "r") as hf:
                st_img = hf["st_image"]
                st_comp = hf["st_component"]
                labels = hf["label"]

                for i in idx_arr:
                    x_time = np.asarray(st_comp[i], dtype=np.float32)
                    if x_time.ndim == 1:
                        x_time = x_time[..., None]

                    x_freq = np.asarray(st_img[i], dtype=np.float32)
                    y = int(labels[i])
                    yield (x_time, x_freq), np.int32(y)

    output_signature = (
        (
            tf.TensorSpec(shape=(time_steps, 1), dtype=tf.float32),
            tf.TensorSpec(
                shape=(freq_bins, time_steps, freq_channels),
                dtype=tf.float32,
            ),
        ),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature,
    )

    if shuffle:
        ds = ds.shuffle(
            buffer_size=shuffle_buffer,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(batch_size, drop_remainder=False)

    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def cast_batch(inputs, y):
    """统一数据 dtype，避免 float64/int64 混入训练流程。"""
    x_time, x_freq = inputs
    x_time = tf.cast(x_time, tf.float32)
    x_freq = tf.cast(x_freq, tf.float32)
    y = tf.cast(y, tf.int32)
    return (x_time, x_freq), y


def assert_finite_batch(inputs, y):
    """调试用：检查输入中是否存在 NaN/Inf。"""
    x_time, x_freq = inputs
    tf.debugging.assert_all_finite(x_time, "x_time has NaN/Inf")
    tf.debugging.assert_all_finite(x_freq, "x_freq has NaN/Inf")
    return (x_time, x_freq), y
