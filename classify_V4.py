import math
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, mixed_precision

from Chart import chart
from LoadData import h5_data
from NetWork.microseismic_model import build_microseismic_model

# 0) 全局 / 数据参数
SEED = 42    
BATCH_SIZE = 16
EPOCHS = 50
VAL_RATIO = 0.1
TEST_RATIO = 0.1

TIME_STEPS = 4000
FREQ_BINS = 16
FREQ_CHANNELS = 3
NUM_CLASSES = 3

ROOT_DIR = os.environ.get("CLASSIFY_DATA_DIR", "data/processed")
SAVE_DIR = os.environ.get("CLASSIFY_OUTPUT_DIR", ".")
RESULT_DIR = os.path.join(SAVE_DIR, "training_results")

CLASS_NAMES = [
    "岩石破裂",
    "爆破",
    "噪声",
]
CLASS_KEYS = [
    "rock",
    "blast",
    "noise",
]
USE_CLASS_WEIGHT = True
DEBUG_NUMERICS = False
USE_XLA = False

INITIAL_LR = 1e-4
FINAL_LR = 1e-6
ADAM_EPSILON = 1e-4
CLIPNORM = 1.0
EARLY_STOPPING_PATIENCE = 8

# 如需混合精度可开启：
# policy = mixed_precision.Policy("mixed_float16")
# mixed_precision.set_global_policy(policy)

if __name__ == "__main__":

    tf.keras.utils.set_random_seed(SEED)
    # ==========================================================
    # 2) 查找 H5 文件
    # ==========================================================
    h5_paths = sorted(
        str(p)
        for p in Path(ROOT_DIR).rglob("*.h5")
    )

    print("Found H5 files:", len(h5_paths))
    if not h5_paths:
        raise FileNotFoundError(
            f"No .h5 found under: {ROOT_DIR}"
        )

    # ==========================================================
    # 3) H5 元数据扫描与数据集划分
    # ==========================================================
    file_names, labels = h5_data.scan_h5_metadata(h5_paths)

    (
        train_set,
        val_set,
        test_set,
        y_train,
        y_val,
        y_test,
    ) = h5_data.stratified_split_by_file_name(
        file_names=file_names,
        labels=labels,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    train_index_map, n_train = h5_data.build_index_map_by_set(
        h5_paths,
        train_set,
    )
    val_index_map, n_val = h5_data.build_index_map_by_set(
        h5_paths,
        val_set,
    )
    test_index_map, n_test = h5_data.build_index_map_by_set(
        h5_paths,
        test_set,
    )

    # 测试集原始样本名：严格按照 test_index_map 的遍历顺序读取。
    # 该顺序与 make_dataset_from_index_map(test_index_map, shuffle=False)
    # 内部 generator 的输出顺序完全一致，因此可与 y_true / y_pred 一一对应。
    test_sample_names = h5_data.get_file_names_from_index_map(
        test_index_map
    ).tolist()

    if len(test_sample_names) != n_test:
        raise RuntimeError(
            "测试样本名数量与 test_index_map 不一致："
            f"names={len(test_sample_names)}, n_test={n_test}。"
        )

    print(
        f"Samples: train={n_train}, "
        f"val={n_val}, test={n_test}"
    )

    # ==========================================================
    # 4) 构建 tf.data.Dataset
    # ==========================================================
    shuffle_buffer = min(
        4096,
        max(n_train, 1024),
    )

    dataset_kwargs = dict(
        batch_size=BATCH_SIZE,
        prefetch=True,
        time_steps=TIME_STEPS,
        freq_bins=FREQ_BINS,
        freq_channels=FREQ_CHANNELS,
    )

    train_dataset = h5_data.make_dataset_from_index_map(
        train_index_map,
        shuffle=True,
        shuffle_buffer=shuffle_buffer,
        **dataset_kwargs,
    )
    val_dataset = h5_data.make_dataset_from_index_map(
        val_index_map,
        shuffle=False,
        **dataset_kwargs,
    )
    test_dataset = h5_data.make_dataset_from_index_map(
        test_index_map,
        shuffle=False,
        **dataset_kwargs,
    )

    # fit() 中 train / val 使用固定 steps，因此 repeat() 保持原逻辑。
    train_dataset = train_dataset.repeat()
    val_dataset = val_dataset.repeat()

    train_dataset = train_dataset.map(
        h5_data.cast_batch,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_dataset = val_dataset.map(
        h5_data.cast_batch,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    test_dataset = test_dataset.map(
        h5_data.cast_batch,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if DEBUG_NUMERICS:
        train_dataset = train_dataset.map(
            h5_data.assert_finite_batch,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # ==========================================================
    # 5) 类别权重
    # ==========================================================
    class_weight_dict = None

    if USE_CLASS_WEIGHT:
        classes = np.arange(NUM_CLASSES, dtype=np.int64)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        class_weight_dict = {
            int(c): float(w)
            for c, w in zip(classes, class_weights)
        }
        print("Class weights:", class_weight_dict)
    else:
        print(
            "Class weights: disabled "
            "(optimize overall accuracy)"
        )

    # ==========================================================
    # 6) 训练步数与学习率
    # ==========================================================
    steps_per_epoch = int(
        math.ceil(n_train / BATCH_SIZE)
    )
    validation_steps = int(
        math.ceil(n_val / BATCH_SIZE)
    )
    total_steps = EPOCHS * steps_per_epoch

    os.makedirs(RESULT_DIR, exist_ok=True)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=INITIAL_LR,
        decay_steps=total_steps,
        alpha=FINAL_LR / INITIAL_LR,
    )

    base_optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        epsilon=ADAM_EPSILON,
        clipnorm=CLIPNORM,
    )

    if (
        mixed_precision.global_policy().name == "mixed_float16"
        and not isinstance(
            base_optimizer,
            mixed_precision.LossScaleOptimizer,
        )
    ):
        optimizer = mixed_precision.LossScaleOptimizer(
            base_optimizer
        )
    else:
        optimizer = base_optimizer

    # ==========================================================
    # 7) 构建与编译网络
    # ==========================================================
    print("Building model...")
    model = build_microseismic_model(
        time_steps=TIME_STEPS,
        freq_bins=FREQ_BINS,
        freq_channels=FREQ_CHANNELS,
        num_classes=NUM_CLASSES,
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        ),
        metrics=["accuracy"],
        jit_compile=USE_XLA,
    )
    model.summary()

    # ==========================================================
    # 8) 指标监控与 callbacks
    # ==========================================================
    training_monitor = chart.TrainingMetricsMonitor(
        val_dataset=val_dataset,
        validation_steps=validation_steps,
        save_dir=RESULT_DIR,
        class_names=CLASS_NAMES,
        class_keys=CLASS_KEYS,
    )

    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(
                RESULT_DIR,
                "best_model.keras",
            ),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(
                RESULT_DIR,
                "tensorboard_logs",
            ),
            histogram_freq=0,
            write_images=False,
        ),
        training_monitor,
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    # ==========================================================
    # 9) Train
    # ==========================================================
    print("Training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        class_weight=class_weight_dict,
        callbacks=callbacks_list,
        verbose=1,
    )

    # ==========================================================
    # 10) Evaluate
    # ==========================================================
    print("\nEvaluating best weights in memory...")
    test_results = model.evaluate(
        test_dataset,
        verbose=1,
    )
    print("Raw evaluate output:", test_results)

    # ==========================================================
    # 11) 绘图
    # ==========================================================
    chart.plot_training_results(
        monitor=training_monitor,
        save_dir=RESULT_DIR,
    )
    print(
        f"Training plots saved to: {RESULT_DIR}"
    )

    # ==========================================================
    # 12) Test metrics
    # ==========================================================
    test_metrics = chart.evaluate_and_save_test_metrics(
        model=model,
        test_dataset=test_dataset,
        save_dir=RESULT_DIR,
        class_names=CLASS_NAMES,
        class_keys=CLASS_KEYS,
        sample_names=test_sample_names,
    )

    # ==========================================================
    # 13) Training summary
    # ==========================================================
    training_config = {
        "TensorFlow version": tf.__version__,
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps,
        "initial_learning_rate": INITIAL_LR,
        "final_learning_rate": FINAL_LR,
        "lr_schedule": "CosineDecay",
        "optimizer": "Adam",
        "class_weight": class_weight_dict,
        "model_monitor": "val_accuracy",
    }

    chart.save_training_summary(
        monitor=training_monitor,
        test_metrics=test_metrics,
        save_dir=RESULT_DIR,
        config=training_config,
    )

    print("\n=================================")
    print("Training finished.")
    print(f"All results saved in:\n{RESULT_DIR}")
    print("\n=================================")
