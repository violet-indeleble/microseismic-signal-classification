import tensorflow as tf
from tensorflow.keras import layers, models

from Class import CLASS


def residual_block_x(
    x,
    filters,
    kernel_size=7,
    stride=1,
    use_dilations=True,
    dilations=(1, 2, 4),
    use_se=True,
    r=16,
    use_eca=True,
    gamma=2,
):
    """时域 1D 残差块。"""
    shortcut = x

    x = layers.Conv1D(
        filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        activation=None,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    if stride > 1:
        shortcut = layers.Conv1D(
            filters,
            1,
            strides=stride,
            padding="same",
            activation=None,
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = CLASS.MultiScaleDilatedConv_1D(
        filters,
        kernel_size=kernel_size,
        use_dilations=use_dilations,
        dilations=dilations,
        use_se=use_se,
        r=r,
        use_eca=use_eca,
        gamma=gamma,
    )(x)

    x = layers.Activation("relu")(layers.Add()([x, shortcut]))
    return x


def residual_block_y(
    y,
    filters,
    kernel_size=(3, 7),
    stride=(1, 1),
    use_dilations=True,
    dilations=(1, 2, 4),
    use_se=True,
    r=16,
    use_eca=True,
    gamma=2,
    use_coord=True,
    reduction=32,
):
    """频域 2D 残差块。"""
    shortcut = y

    y = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        activation=None,
    )(y)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    if stride > (1, 1):
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            padding="same",
            activation=None,
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = CLASS.MultiScaleDilatedConv_2D(
        filters,
        kernel_size=kernel_size,
        use_dilations=use_dilations,
        dilations=dilations,
        use_se=use_se,
        r=r,
        use_eca=use_eca,
        gamma=gamma,
        use_coord=use_coord,
        reduction=reduction,
    )(y)

    y = layers.Activation("relu")(layers.Add()([y, shortcut]))
    return y


def build_microseismic_model(
    time_steps=4000,
    freq_bins=16,
    freq_channels=3,
    num_classes=3,
):
    """构建时域 + 频域双分支微震事件分类网络。"""
    # ==================== 时域分支 ====================
    input_time = layers.Input(
        shape=(time_steps, 1),
        name="time_input",
    )

    x = residual_block_x(
        input_time,
        filters=16,
        kernel_size=7,
        stride=2,
        dilations=(1, 2, 4, 8, 16, 32),
        use_dilations=True,
        use_se=True,
        use_eca=True,
    )

    x = residual_block_x(
        x,
        filters=32,
        kernel_size=7,
        stride=2,
        dilations=(1, 2, 4, 8, 16),
        use_dilations=True,
        use_se=True,
        use_eca=True,
    )
    x = layers.Dropout(0.2)(x)

    x = residual_block_x(
        x,
        filters=64,
        kernel_size=5,
        stride=2,
        dilations=(1, 2, 4, 8, 16),
        use_dilations=True,
        use_se=True,
        use_eca=True,
    )

    x = residual_block_x(
        x,
        filters=128,
        kernel_size=5,
        stride=2,
        dilations=(1, 2, 4, 8),
        use_dilations=True,
        use_se=True,
        use_eca=True,
    )
    x = layers.Dropout(0.2)(x)

    x = residual_block_x(
        x,
        filters=256,
        kernel_size=3,
        stride=2,
        dilations=(1, 2, 4, 8),
        use_dilations=True,
        use_se=True,
        use_eca=True,
    )

    x = residual_block_x(
        x,
        filters=512,
        kernel_size=3,
        stride=2,
        dilations=(1, 2, 4),
        use_dilations=True,
        use_se=True,
        use_eca=True,
    )
    x = layers.Dropout(0.2)(x)

    x = CLASS.AttentionPooling_1D_CNN()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # ==================== 频域分支 ====================
    input_freq = layers.Input(
        shape=(freq_bins, time_steps, freq_channels),
        name="freq_domain_input",
    )

    y = layers.BatchNormalization()(input_freq)
    y = layers.ReLU()(y)
    y = layers.Conv2D(
        16,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
    )(y)

    y = residual_block_y(
        y,
        32,
        kernel_size=(3, 7),
        stride=(2, 2),
        dilations=(1, 2, 4, 8, 16, 32),
        use_dilations=True,
        use_se=True,
        use_eca=True,
        use_coord=True,
    )

    y = residual_block_y(
        y,
        64,
        kernel_size=(3, 7),
        stride=(1, 2),
        dilations=(1, 2, 4, 8, 16),
        use_dilations=True,
        use_se=True,
        use_eca=True,
        use_coord=True,
    )

    y = residual_block_y(
        y,
        128,
        kernel_size=(3, 5),
        stride=(1, 2),
        dilations=(1, 2, 4, 8, 16),
        use_dilations=True,
        use_se=True,
        use_eca=True,
        use_coord=True,
    )

    y = residual_block_y(
        y,
        256,
        kernel_size=(3, 5),
        stride=(1, 2),
        dilations=(1, 2, 4, 8),
        use_dilations=True,
        use_se=True,
        use_eca=True,
        use_coord=True,
    )

    y = residual_block_y(
        y,
        512,
        kernel_size=(3, 3),
        stride=(1, 2),
        dilations=(1, 2, 4, 8),
        use_dilations=True,
        use_se=True,
        use_eca=True,
        use_coord=True,
    )

    y = residual_block_y(
        y,
        512,
        kernel_size=(3, 3),
        stride=(1, 2),
        dilations=(1, 2, 4),
        use_dilations=True,
        use_se=True,
        use_eca=True,
        use_coord=True,
    )

    y = CLASS.AttentionPooling_2D_CNN()(y)
    y = layers.Dropout(0.25)(y)
    y = layers.Dense(256, activation="relu")(y)
    y = layers.Dropout(0.3)(y)

    # ==================== 特征融合 ====================
    combined = layers.concatenate([x, y])
    combined = layers.Dense(128, activation="relu")(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Dropout(0.3)(combined)

    output = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="output",
    )(combined)

    return models.Model(
        inputs=[input_time, input_freq],
        outputs=output,
    )
