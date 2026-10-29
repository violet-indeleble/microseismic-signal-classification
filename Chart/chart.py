import os
import csv
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

# ============================================================
# 基础工具
# ============================================================

def _write_csv(rows, file_path):
    """
    将 list[dict] 保存为 CSV。
    每次覆盖写入，适合每个 epoch 更新一次，
    即使训练中途终止，也可以保留此前数据。
    """
    if not rows:
        return

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _get_log_value(logs, key):
    value = logs.get(key, np.nan)

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _get_current_learning_rate(model):
    """
    TF 2.17 兼容：
    1. 普通 Optimizer
    2. LearningRateSchedule
    3. LossScaleOptimizer
    """

    optimizer = model.optimizer

    # mixed precision 情况下尝试获取内部 optimizer
    if hasattr(optimizer, "inner_optimizer"):
        optimizer = optimizer.inner_optimizer

    elif hasattr(optimizer, "_optimizer"):
        optimizer = optimizer._optimizer

    # Keras 3 / TF 2.17 中真正保存 schedule 的对象
    lr_obj = getattr(
        optimizer,
        "_learning_rate",
        optimizer.learning_rate
    )

    if isinstance(
        lr_obj,
        tf.keras.optimizers.schedules.LearningRateSchedule
    ):
        lr = lr_obj(optimizer.iterations)

    elif callable(lr_obj):
        lr = lr_obj(optimizer.iterations)

    else:
        lr = optimizer.learning_rate

    return float(tf.keras.backend.get_value(lr))


def _collect_predictions(
    model,
    dataset,
    steps=None,
    return_probabilities=False,
):
    """
    对 Dataset 做预测。

    注意：
    如果 dataset 使用了 repeat()，
    必须传入 steps，否则会无限循环。

    当 return_probabilities=True 时，额外返回每个样本的
    三分类预测概率，便于记录误分类样本的置信度。
    """

    y_true = []
    y_pred = []
    all_probs = []

    if steps is not None:
        dataset = dataset.take(steps)

    for (x_time, x_freq), y in dataset:

        probs = model(
            [x_time, x_freq],
            training=False
        )

        pred = tf.argmax(
            probs,
            axis=1,
            output_type=tf.int32
        )

        y_true.append(
            y.numpy().astype(np.int32)
        )

        y_pred.append(
            pred.numpy().astype(np.int32)
        )

        if return_probabilities:
            all_probs.append(
                probs.numpy().astype(np.float32)
            )

    if len(y_true) == 0:
        raise RuntimeError(
            "No validation/test samples were collected."
        )

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    if return_probabilities:
        all_probs = np.concatenate(all_probs, axis=0)
        return y_true, y_pred, all_probs

    return y_true, y_pred


# ============================================================
# Epoch 级训练监控 Callback
# ============================================================

class TrainingMetricsMonitor(tf.keras.callbacks.Callback):
    """
    每个 Epoch 结束后：

    1. 从 logs 获取：
        train_loss
        val_loss
        train_accuracy
        val_accuracy

    2. 重新在完整验证集上推理：
        val_macro_precision
        val_macro_recall
        val_macro_f1

        rock_precision
        rock_recall
        rock_f1

        blast_precision
        blast_recall
        blast_f1

        noise_precision
        noise_recall
        noise_f1

    3. 保存：
        training_history.csv
        per_class_history.csv
        Generalization gap.csv

    4. 记录：
        learning_rate
        epoch_time
    """

    def __init__(
        self,
        val_dataset,
        validation_steps,
        save_dir="training_results",
        class_names=None,
        class_keys=None,
    ):
        super().__init__()

        self.val_dataset = val_dataset
        self.validation_steps = validation_steps
        self.save_dir = save_dir

        if class_names is None:
            class_names = [
                "岩石破裂",
                "爆破",
                "噪声",
            ]

        if class_keys is None:
            class_keys = [
                "rock",
                "blast",
                "noise",
            ]

        if len(class_names) != 3:
            raise ValueError(
                "当前代码按三分类设计，class_names 必须包含 3 个类别。"
            )

        if len(class_keys) != 3:
            raise ValueError(
                "class_keys 必须包含 rock / blast / noise 三个类别键。"
            )

        self.class_names = class_names
        self.class_keys = class_keys

        self.training_history = []
        self.per_class_history = []
        self.generalization_gap_history = []

        self._epoch_start_time = None

        os.makedirs(
            self.save_dir,
            exist_ok=True
        )

    def on_epoch_begin(self, epoch, logs=None):

        self._epoch_start_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        # ----------------------------------------------------
        # 1. Epoch 时间
        # ----------------------------------------------------
        # 这里在额外验证推理前记录。
        # 因此 epoch_time 表示：
        # model.fit 自身的训练 + Keras validation 时间。
        epoch_time = (
            time.perf_counter()
            - self._epoch_start_time
        )

        # ----------------------------------------------------
        # 2. Keras 自带指标
        # ----------------------------------------------------
        train_loss = _get_log_value(
            logs,
            "loss"
        )

        val_loss = _get_log_value(
            logs,
            "val_loss"
        )

        train_accuracy = _get_log_value(
            logs,
            "accuracy"
        )

        val_accuracy = _get_log_value(
            logs,
            "val_accuracy"
        )

        # ----------------------------------------------------
        # 3. 当前学习率
        # ----------------------------------------------------
        learning_rate = _get_current_learning_rate(
            self.model
        )

        # ----------------------------------------------------
        # 4. 在完整 Validation Set 上重新预测一次
        # ----------------------------------------------------
        y_true, y_pred = _collect_predictions(
            model=self.model,
            dataset=self.val_dataset,
            steps=self.validation_steps,
        )

        labels = np.arange(
            len(self.class_names)
        )

        # ----------------------------------------------------
        # 5. Macro 指标
        # ----------------------------------------------------
        (
            macro_precision,
            macro_recall,
            macro_f1,
            _
        ) = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        # ----------------------------------------------------
        # 6. 每类别指标
        # ----------------------------------------------------
        (
            precision_each,
            recall_each,
            f1_each,
            support_each
        ) = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )

        # ----------------------------------------------------
        # 7. training_history.csv
        # ----------------------------------------------------
        training_row = {
            "epoch": epoch + 1,

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "train_accuracy":
                train_accuracy,

            "val_accuracy":
                val_accuracy,

            "val_macro_precision":
                float(macro_precision),

            "val_macro_recall":
                float(macro_recall),

            "val_macro_f1":
                float(macro_f1),

            "learning_rate":
                learning_rate,

            "epoch_time":
                epoch_time,
        }

        self.training_history.append(
            training_row
        )

        # ----------------------------------------------------
        # 8. per_class_history.csv
        # ----------------------------------------------------
        per_class_row = {
            "epoch": epoch + 1,

            "rock_precision":
                float(precision_each[0]),

            "rock_recall":
                float(recall_each[0]),

            "rock_f1":
                float(f1_each[0]),

            "blast_precision":
                float(precision_each[1]),

            "blast_recall":
                float(recall_each[1]),

            "blast_f1":
                float(f1_each[1]),

            "noise_precision":
                float(precision_each[2]),

            "noise_recall":
                float(recall_each[2]),

            "noise_f1":
                float(f1_each[2]),
        }

        self.per_class_history.append(
            per_class_row
        )

        # ----------------------------------------------------
        # 9. 泛化差距
        # ----------------------------------------------------
        #
        # Accuracy gap > 0:
        #     Train accuracy 高于 Validation
        #
        # Loss gap > 0:
        #     Validation loss 高于 Train
        #
        accuracy_gap = (
            train_accuracy
            - val_accuracy
        )

        loss_gap = (
            val_loss
            - train_loss
        )

        gap_row = {
            "epoch": epoch + 1,

            "accuracy_gap":
                accuracy_gap,

            "loss_gap":
                loss_gap,
        }

        self.generalization_gap_history.append(
            gap_row
        )

        # ----------------------------------------------------
        # 10. 把自定义指标放回 logs
        # ----------------------------------------------------
        # 虽然这一版不让 Checkpoint 监控这些指标，
        # 但写进 logs 后可以方便后续扩展。
        logs["val_macro_precision"] = float(
            macro_precision
        )

        logs["val_macro_recall"] = float(
            macro_recall
        )

        logs["val_macro_f1"] = float(
            macro_f1
        )

        logs["rock_precision"] = float(
            precision_each[0]
        )

        logs["rock_recall"] = float(
            recall_each[0]
        )

        logs["rock_f1"] = float(
            f1_each[0]
        )

        # ----------------------------------------------------
        # 11. 每一个 Epoch 都立即保存
        # ----------------------------------------------------
        self.save_csv_files()

        # ----------------------------------------------------
        # 12. 输出关键指标
        # ----------------------------------------------------
        print(
            "\n"
            f"[Epoch {epoch + 1}] "
            f"lr={learning_rate:.6e} | "
            f"val_macro_f1={macro_f1:.4f} | "
            f"rock_P={precision_each[0]:.4f} | "
            f"rock_R={recall_each[0]:.4f} | "
            f"rock_F1={f1_each[0]:.4f}"
        )

    def save_csv_files(self):

        _write_csv(
            self.training_history,
            os.path.join(
                self.save_dir,
                "training_history.csv"
            )
        )

        _write_csv(
            self.per_class_history,
            os.path.join(
                self.save_dir,
                "per_class_history.csv"
            )
        )

        _write_csv(
            self.generalization_gap_history,
            os.path.join(
                self.save_dir,
                "Generalization gap.csv"
            )
        )


# ============================================================
# 四张训练曲线
# ============================================================

def plot_training_results(
    monitor,
    save_dir="training_results"
):
    """
    只绘制四张核心图：

    Figure a:
        Train Loss / Validation Loss

    Figure b:
        Train Accuracy / Validation Accuracy

    Figure c:
        Rock Precision / Recall / F1

    Figure d:
        Validation F1 of three classes
    """

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    if len(monitor.training_history) == 0:
        raise ValueError(
            "Training history is empty."
        )

    training_history = monitor.training_history
    per_class_history = monitor.per_class_history

    epoch = np.asarray(
        [
            row["epoch"]
            for row in training_history
        ]
    )

    # ========================================================
    # Figure a
    # Train Loss / Validation Loss
    # ========================================================

    train_loss = np.asarray(
        [
            row["train_loss"]
            for row in training_history
        ]
    )

    val_loss = np.asarray(
        [
            row["val_loss"]
            for row in training_history
        ]
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epoch,
        train_loss,
        linewidth=2,
        label="Train Loss"
    )

    plt.plot(
        epoch,
        val_loss,
        linewidth=2,
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "(a) Training and Validation Loss"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "figure_a_loss.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Figure b
    # Train Accuracy / Validation Accuracy
    # ========================================================

    train_accuracy = np.asarray(
        [
            row["train_accuracy"]
            for row in training_history
        ]
    )

    val_accuracy = np.asarray(
        [
            row["val_accuracy"]
            for row in training_history
        ]
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epoch,
        train_accuracy,
        linewidth=2,
        label="Train Accuracy"
    )

    plt.plot(
        epoch,
        val_accuracy,
        linewidth=2,
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "(b) Training and Validation Accuracy"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "figure_b_accuracy.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Figure c
    # Rock Precision / Recall / F1
    # ========================================================

    rock_precision = np.asarray(
        [
            row["rock_precision"]
            for row in per_class_history
        ]
    )

    rock_recall = np.asarray(
        [
            row["rock_recall"]
            for row in per_class_history
        ]
    )

    rock_f1 = np.asarray(
        [
            row["rock_f1"]
            for row in per_class_history
        ]
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epoch,
        rock_precision,
        linewidth=2,
        label="Rock Precision"
    )

    plt.plot(
        epoch,
        rock_recall,
        linewidth=2,
        label="Rock Recall"
    )

    plt.plot(
        epoch,
        rock_f1,
        linewidth=2,
        label="Rock F1"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Metric"
    )

    plt.ylim(
        0.0,
        1.05
    )

    plt.title(
        "(c) Rock-fracture Validation Metrics"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "figure_c_rock_metrics.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Figure d
    # Three-class Validation F1
    # ========================================================

    blast_f1 = np.asarray(
        [
            row["blast_f1"]
            for row in per_class_history
        ]
    )

    noise_f1 = np.asarray(
        [
            row["noise_f1"]
            for row in per_class_history
        ]
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epoch,
        rock_f1,
        linewidth=2,
        label="Rock"
    )

    plt.plot(
        epoch,
        blast_f1,
        linewidth=2,
        label="Blast"
    )

    plt.plot(
        epoch,
        noise_f1,
        linewidth=2,
        label="Noise"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Validation F1-score"
    )

    plt.ylim(
        0.0,
        1.05
    )

    plt.title(
        "(d) Validation F1-score by Class"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "figure_d_validation_f1.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Test Set 指标
# ============================================================

def evaluate_and_save_test_metrics(
    model,
    test_dataset,
    save_dir="training_results",
    class_names=None,
    class_keys=None,
    sample_names=None,
):
    """
    保存 test_metrics.csv。

    同时记录所有分类错误的测试样本：
    1. misclassified_samples.txt
       仅保存原始样本名，例如 018622.txt
    2. misclassified_samples.csv
       保存样本名、真实类别、预测类别、预测置信度和各类概率。

    CSV 中同时保存完整 confusion matrix 的 9 个元素。

    因此以后不用重新预测模型，
    直接读取 test_metrics.csv
    就可以重建三分类混淆矩阵。
    """

    if class_names is None:
        class_names = [
            "岩石破裂",
            "爆破",
            "噪声",
        ]

    if class_keys is None:
        class_keys = [
            "rock",
            "blast",
            "noise",
        ]

    y_true, y_pred, y_prob = _collect_predictions(
        model=model,
        dataset=test_dataset,
        steps=None,
        return_probabilities=True,
    )

    if sample_names is None:
        sample_names = [
            f"sample_{i:06d}"
            for i in range(len(y_true))
        ]
    else:
        sample_names = [str(name) for name in sample_names]

    if len(sample_names) != len(y_true):
        raise ValueError(
            "sample_names 与 test_dataset 样本数不一致："
            f"len(sample_names)={len(sample_names)}, "
            f"len(y_true)={len(y_true)}。"
            "测试集必须保持 shuffle=False，且 sample_names 顺序必须与 "
            "test_dataset 完全一致。"
        )

    labels = np.arange(
        len(class_names)
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Macro
    # --------------------------------------------------------

    (
        precision_macro,
        recall_macro,
        f1_macro,
        _
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Weighted
    # --------------------------------------------------------

    (
        precision_weighted,
        recall_weighted,
        f1_weighted,
        _
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Per class
    # --------------------------------------------------------

    (
        precision_each,
        recall_each,
        f1_each,
        support_each
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    # --------------------------------------------------------
    # 记录全部误分类样本
    # --------------------------------------------------------

    wrong_indices = np.flatnonzero(
        y_true != y_pred
    )

    misclassified_rows = []

    for sample_index in wrong_indices:

        true_id = int(y_true[sample_index])
        pred_id = int(y_pred[sample_index])
        probs = y_prob[sample_index]

        original_name = sample_names[sample_index]
        sample_basename = os.path.basename(original_name)

        row = {
            "sample_name": original_name,
            "sample_basename": sample_basename,
            "sample_index": int(sample_index),
            "true_class_id": true_id,
            "true_class": class_names[true_id],
            "pred_class_id": pred_id,
            "pred_class": class_names[pred_id],
            "pred_confidence": float(probs[pred_id]),
        }

        for class_i, class_key in enumerate(class_keys):
            row[f"prob_{class_key}"] = float(probs[class_i])

        misclassified_rows.append(row)

    misclassified_csv_path = os.path.join(
        save_dir,
        "misclassified_samples.csv"
    )

    if misclassified_rows:
        _write_csv(
            misclassified_rows,
            misclassified_csv_path
        )
    else:
        # 没有误分类时也生成一个只有表头的 CSV，便于批处理。
        os.makedirs(save_dir, exist_ok=True)
        fieldnames = [
            "sample_name",
            "sample_basename",
            "sample_index",
            "true_class_id",
            "true_class",
            "pred_class_id",
            "pred_class",
            "pred_confidence",
        ] + [
            f"prob_{key}"
            for key in class_keys
        ]

        with open(
            misclassified_csv_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )
            writer.writeheader()

    misclassified_txt_path = os.path.join(
        save_dir,
        "misclassified_samples.txt"
    )

    with open(
        misclassified_txt_path,
        "w",
        encoding="utf-8"
    ) as f:
        for row in misclassified_rows:
            f.write(
                row["sample_basename"] + "\n"
            )

    # --------------------------------------------------------
    # 单行宽格式 CSV
    # --------------------------------------------------------

    result = {
        "num_samples":
            int(len(y_true)),

        "accuracy":
            float(accuracy),

        "precision_macro":
            float(precision_macro),

        "recall_macro":
            float(recall_macro),

        "f1_macro":
            float(f1_macro),

        "precision_weighted":
            float(precision_weighted),

        "recall_weighted":
            float(recall_weighted),

        "f1_weighted":
            float(f1_weighted),

        "misclassified_count":
            int(len(wrong_indices)),

        "misclassified_rate":
            float(len(wrong_indices) / len(y_true)),
    }

    # 每类别指标
    for i, key in enumerate(class_keys):

        result[
            f"{key}_precision"
        ] = float(
            precision_each[i]
        )

        result[
            f"{key}_recall"
        ] = float(
            recall_each[i]
        )

        result[
            f"{key}_f1"
        ] = float(
            f1_each[i]
        )

        result[
            f"{key}_support"
        ] = int(
            support_each[i]
        )

    # 混淆矩阵 3×3 全部写入
    for true_i, true_key in enumerate(class_keys):

        for pred_i, pred_key in enumerate(class_keys):

            result[
                f"cm_true_{true_key}_pred_{pred_key}"
            ] = int(
                cm[true_i, pred_i]
            )

    _write_csv(
        [result],
        os.path.join(
            save_dir,
            "test_metrics.csv"
        )
    )

    # --------------------------------------------------------
    # 控制台输出
    # --------------------------------------------------------

    print(
        "\n========== Test Metrics =========="
    )

    print(
        f"Accuracy            : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision (macro)   : "
        f"{precision_macro:.4f}"
    )

    print(
        f"Recall (macro)      : "
        f"{recall_macro:.4f}"
    )

    print(
        f"F1-score (macro)    : "
        f"{f1_macro:.4f}"
    )

    print(
        "\n========== Per-class Metrics =========="
    )

    for i, class_name in enumerate(class_names):

        print(
            f"{class_name}: "
            f"precision={precision_each[i]:.4f}, "
            f"recall={recall_each[i]:.4f}, "
            f"f1={f1_each[i]:.4f}, "
            f"support={support_each[i]}"
        )

    print(
        "\nConfusion Matrix:"
    )

    print(
        cm
    )

    print(
        f"\nMisclassified samples : {len(wrong_indices)} / {len(y_true)}"
    )
    print(
        f"Saved misclassified list to: {misclassified_txt_path}"
    )
    print(
        f"Saved misclassified details to: {misclassified_csv_path}"
    )

    return result


# ============================================================
# training_summary.txt
# ============================================================

def save_training_summary(
    monitor,
    test_metrics,
    save_dir="training_results",
    config=None,
):
    """
    生成训练摘要。

    当前最佳模型标准：
        val_accuracy 最大

    与当前 ModelCheckpoint / EarlyStopping 保持一致。
    """

    if len(monitor.training_history) == 0:
        raise ValueError(
            "Training history is empty."
        )

    training_history = monitor.training_history
    per_class_history = monitor.per_class_history
    gap_history = monitor.generalization_gap_history

    # --------------------------------------------------------
    # 找 val_accuracy 最大的 epoch
    # --------------------------------------------------------

    val_accuracies = np.asarray(
        [
            row["val_accuracy"]
            for row in training_history
        ],
        dtype=float
    )

    best_index = int(
        np.nanargmax(
            val_accuracies
        )
    )

    best_training = training_history[
        best_index
    ]

    best_class = per_class_history[
        best_index
    ]

    best_gap = gap_history[
        best_index
    ]

    final_training = training_history[-1]
    final_class = per_class_history[-1]

    summary_path = os.path.join(
        save_dir,
        "training_summary.txt"
    )

    lines = []

    lines.append(
        "========================================"
    )

    lines.append(
        "Training Summary"
    )

    lines.append(
        "========================================"
    )

    lines.append(
        ""
    )

    lines.append(
        f"Completed epochs      : "
        f"{len(training_history)}"
    )

    lines.append(
        "Best model monitor    : val_accuracy"
    )

    lines.append(
        f"Best epoch            : "
        f"{best_training['epoch']}"
    )

    lines.append(
        ""
    )

    # --------------------------------------------------------
    # 配置
    # --------------------------------------------------------

    if config is not None:

        lines.append(
            "---------- Training Configuration ----------"
        )

        for key, value in config.items():

            lines.append(
                f"{key:<24}: {value}"
            )

        lines.append(
            ""
        )

    # --------------------------------------------------------
    # Best Epoch
    # --------------------------------------------------------

    lines.append(
        "---------- Best Epoch Metrics ----------"
    )

    lines.append(
        f"Train loss            : "
        f"{best_training['train_loss']:.6f}"
    )

    lines.append(
        f"Validation loss       : "
        f"{best_training['val_loss']:.6f}"
    )

    lines.append(
        f"Train accuracy        : "
        f"{best_training['train_accuracy']:.6f}"
    )

    lines.append(
        f"Validation accuracy   : "
        f"{best_training['val_accuracy']:.6f}"
    )

    lines.append(
        f"Val macro precision   : "
        f"{best_training['val_macro_precision']:.6f}"
    )

    lines.append(
        f"Val macro recall      : "
        f"{best_training['val_macro_recall']:.6f}"
    )

    lines.append(
        f"Val macro F1          : "
        f"{best_training['val_macro_f1']:.6f}"
    )

    lines.append(
        ""
    )

    # --------------------------------------------------------
    # Rock 指标
    # --------------------------------------------------------

    lines.append(
        "---------- Rock-fracture Metrics at Best Epoch ----------"
    )

    lines.append(
        f"Rock precision        : "
        f"{best_class['rock_precision']:.6f}"
    )

    lines.append(
        f"Rock recall           : "
        f"{best_class['rock_recall']:.6f}"
    )

    lines.append(
        f"Rock F1               : "
        f"{best_class['rock_f1']:.6f}"
    )

    lines.append(
        ""
    )

    # --------------------------------------------------------
    # 泛化差距
    # --------------------------------------------------------

    lines.append(
        "---------- Generalization Gap ----------"
    )

    lines.append(
        f"Accuracy gap          : "
        f"{best_gap['accuracy_gap']:.6f}"
    )

    lines.append(
        f"Loss gap              : "
        f"{best_gap['loss_gap']:.6f}"
    )

    lines.append(
        ""
    )

    # --------------------------------------------------------
    # Final epoch
    # --------------------------------------------------------

    lines.append(
        "---------- Final Training Epoch ----------"
    )

    lines.append(
        f"Final epoch           : "
        f"{final_training['epoch']}"
    )

    lines.append(
        f"Validation accuracy   : "
        f"{final_training['val_accuracy']:.6f}"
    )

    lines.append(
        f"Validation macro F1   : "
        f"{final_training['val_macro_f1']:.6f}"
    )

    lines.append(
        f"Rock precision        : "
        f"{final_class['rock_precision']:.6f}"
    )

    lines.append(
        f"Rock recall           : "
        f"{final_class['rock_recall']:.6f}"
    )

    lines.append(
        f"Rock F1               : "
        f"{final_class['rock_f1']:.6f}"
    )

    lines.append(
        ""
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    lines.append(
        "---------- Test Set ----------"
    )

    test_keys = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "rock_precision",
        "rock_recall",
        "rock_f1",
        "blast_precision",
        "blast_recall",
        "blast_f1",
        "noise_precision",
        "noise_recall",
        "noise_f1",
    ]

    for key in test_keys:

        if key in test_metrics:

            lines.append(
                f"{key:<24}: "
                f"{test_metrics[key]:.6f}"
            )

    lines.append(
        ""
    )

    lines.append(
        "========================================"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(lines)
        )

    print(
        f"Saved training summary to: "
        f"{summary_path}"
    )