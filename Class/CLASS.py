import numpy as np
import tensorflow as tf
import math
from tensorflow.keras import models
from tensorflow.keras import layers 

#
# 通道注意力 SE 自适应时域和频域
@tf.keras.utils.register_keras_serializable()
class SEBlock(layers.Layer):
    """
    Squeeze-and-Excitation Block with Conv layers
    - 支持 1D (B, T, C)
    - 支持 2D (B, H, W, C)
    """
    def __init__(self, r=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = r

        self.channels = None
        self.pool = None
        self.conv1 = None
        self.conv2 = None

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        c = input_shape[-1]
        if c is None:
            raise ValueError(
                f"SEBlock requires known channel dimension, got {input_shape}"
            )

        self.channels = int(c)
        reduced_channels = max(self.channels // self.reduction, 1)

        rank = len(input_shape)
        if rank == 3:
            self.pool = layers.GlobalAveragePooling1D(keepdims=True)
            self.conv1 = layers.Conv1D(reduced_channels, 1, activation="relu")
            self.conv2 = layers.Conv1D(self.channels, 1, activation="sigmoid")
        elif rank == 4:
            self.pool = layers.GlobalAveragePooling2D(keepdims=True)
            self.conv1 = layers.Conv2D(reduced_channels, 1, activation="relu")
            self.conv2 = layers.Conv2D(self.channels, 1, activation="sigmoid")
        else:
            raise ValueError(
                f"SEBlock expects rank 3 or 4 input, got rank={rank}"
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        # Ensure dtype consistency
        dtype = inputs.dtype

        x = self.pool(inputs)

        x = self.conv1(x)
        x = self.conv2(x)  # (B, C), already sigmoid

        # Re-scale
        if len(inputs.shape) == 3:
            x = x[:, None, :]  # (B, 1, C)
        else:
            x = x[:, None, None, :]  # (B, 1, 1, C)

        x = tf.cast(x, dtype)

        return inputs * x

    def get_config(self):
        config = super().get_config()
        config.update({
            "r": self.reduction,
        })
        return config
#
# 通道注意力 增强版ECA-Net 自适应时域和频域
@tf.keras.utils.register_keras_serializable()
class ECALayer(layers.Layer):
    """
    通道数大于512时，将gamma改为2.5
    """
    def __init__(self, gamma=2, b=1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.b = b
        self.channels = None
        self.avg_pool = None
        self.max_pool = None
        self.conv = None

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        c = input_shape[-1]
        if c is None:
            raise ValueError(f"ECALayer requires a known channel dimension, got input_shape={input_shape}")
        self.channels = int(c)

        # 计算卷积核大小 k
        log_C = math.log2(self.channels)
        k = abs((log_C + self.b) / self.gamma)
        k = int(k)
        k = 2 * math.ceil(k / 2) - 1
        k = max(k, 3)

        rank = len(input_shape)
        if rank == 3:
            self.avg_pool = layers.GlobalAveragePooling1D(keepdims=True)
            self.max_pool = layers.GlobalMaxPooling1D(keepdims=True)
        elif rank == 4:
            self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
            self.max_pool = layers.GlobalMaxPooling2D(keepdims=True)
        else:
            raise ValueError(f"ECALayer expects rank 3 or 4 input, got rank={rank}, input_shape={input_shape}")

        # Conv1D: input (B, C, 1) -> output (B, C, 1)
        self.conv = layers.Conv1D(
            filters=1, kernel_size=k, padding="same",
            use_bias=False, activation="sigmoid"
        )

        super().build(input_shape)

    def call(self, inputs, training=None):
        avg_out = self.avg_pool(inputs)
        max_out = self.max_pool(inputs)

        if len(inputs.shape) == 3:       # (B,T,C) -> pool -> (B,1,C)
            avg_sqz = tf.squeeze(avg_out, axis=1)     # (B,C)
            max_sqz = tf.squeeze(max_out, axis=1)
        else:                            # (B,H,W,C) -> pool -> (B,1,1,C)
            avg_sqz = tf.squeeze(avg_out, axis=[1, 2])  # (B,C)
            max_sqz = tf.squeeze(max_out, axis=[1, 2])

        avg_resh = tf.expand_dims(avg_sqz, axis=-1)   # (B,C,1)
        max_resh = tf.expand_dims(max_sqz, axis=-1)

        y = (self.conv(avg_resh) + self.conv(max_resh)) / 2.0  # (B,C,1)
        y_sqz = tf.squeeze(y, axis=-1)                          # (B,C)

        if len(inputs.shape) == 3:
            y_back = tf.expand_dims(y_sqz, 1)                   # (B,1,C)
        else:
            y_back = tf.expand_dims(tf.expand_dims(y_sqz, 1), 2) # (B,1,1,C)

        return inputs * y_back

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "b": self.b})
        return config
#
# 坐标注意力(标准） 时频
@tf.keras.utils.register_keras_serializable()
class CoordinateAttention(layers.Layer):
    def __init__(self, reduction=32,activation="hard_swish", **kwargs):
        super(CoordinateAttention, self).__init__(**kwargs)
        self.reduction = reduction
        self.activation_name = activation
        self.conv1 = None  # 共享的降维卷积
        # self.bn = None
        self.conv_f = None  
        self.conv_t = None  

    def build(self, input_shape):
        channels = input_shape[-1]

        mip = max(8, channels // self.reduction)  # 最小通道数为8
        self.conv1 = layers.Conv2D(mip, kernel_size=1, strides=1, padding='same', use_bias=False)
        # self.bn = layers.BatchNormalization()
        self.conv_f = layers.Conv2D(channels, kernel_size=1, strides=1, padding='same', activation='sigmoid')
        self.conv_t = layers.Conv2D(channels, kernel_size=1, strides=1, padding='same', activation='sigmoid')

        super().build(input_shape)

    def call(self, inputs, training=None):
        # inputs: (B, F, T, C)

        # 分解池化
        avg_f = tf.reduce_mean(inputs, axis=2, keepdims=True)  # 沿T轴池化: (B, F, 1, C) - 频率全局
        avg_t = tf.reduce_mean(inputs, axis=1, keepdims=True)  # 沿F轴池化: (B, 1, T, C) - 时间全局
        x = tf.concat([avg_f, tf.transpose(avg_t, perm=[0, 2, 1, 3])], axis=1)  # 拼接成 (B, F+T, 1, C)

        # 共享降维
        x = self.conv1(x)  # 降维到 mip 通道 (B, F+T, 1, mip)
        # x = self.bn(x, training=training)
        if self.activation_name == "hard_swish":
            x = x * tf.nn.relu6(x + 3.0) / 6.0
        else:
            x = tf.keras.activations.get(self.activation_name)(x)

        # 拆分回F和T方向
        f = tf.shape(inputs)[1]   
        t = tf.shape(inputs)[2]

        # att_f: (B, F, 1, mip)
        # att_t: (B, T, 1, mip) -> (B, 1, T, mip)
        att_f, att_t = tf.split(x, num_or_size_splits=tf.stack([f, t]), axis=1)
        att_t = tf.transpose(att_t, perm=[0, 2, 1, 3])  # (B, 1, T, mip)

        # 生成sigmoid注意力
        att_f = self.conv_f(att_f)  # (B, F, 1, C)
        att_t = self.conv_t(att_t)  # (B, 1, T, C)

        # 组合注意力
        attention = att_f * att_t

        # 乘回输入
        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({
            "reduction": self.reduction,
            "activation": self.activation_name,
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
#
# 注意力池化(基于卷积层) 时序
@tf.keras.utils.register_keras_serializable()
class AttentionPooling_1D_CNN(layers.Layer):
    def __init__(self, hidden_units=None, return_attention=False, **kwargs):
        """
        基于1x1 Conv的注意力池化层（Attention Pooling）。
        
        参数:
            hidden_units: 中间隐藏维度，默认为 d_model（特征维度）。
            return_attention: 是否同时返回注意力权重。
        """
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.return_attention = return_attention
        self.supports_masking = True  # 支持序列 masking

    def build(self, input_shape):
        # input_shape: (batch_size, timesteps, channels)
        d_model = input_shape[-1]
        if d_model is None:
            raise ValueError("The last dimension of input (channel/feature dimension) must be defined.")

        # 默认隐藏维度为 d_model
        hidden = self.hidden_units if self.hidden_units is not None else d_model

        self.conv1 = layers.Conv1D(filters=hidden, kernel_size=1, activation='tanh')
        self.conv2 = layers.Conv1D(filters=1, kernel_size=1, activation=None)
        
        #  build
        super().build(input_shape)

    def call(self, x, mask=None, training=None):
        """
        x: (B, T, C)
        mask: (B, T) 的 bool tensor，可选
        """
        # 先通过两层 1x1 Conv 得到每个时间步的注意力分数
        scores = self.conv1(x, training=training)          # (B, T, hidden)
        scores = self.conv2(scores, training=training)    # (B, T, 1)

        # 为数值稳定，使用 float32 计算 softmax
        scores_f32 = tf.cast(scores, tf.float32)

        if mask is not None:
            # mask 扩展到 (B, T, 1)
            mask_f = tf.cast(mask, tf.float32)[..., tf.newaxis]  # (B, T, 1)
            # 被 mask 的位置分数设为极大的负数
            scores_f32 = scores_f32 + (1.0 - mask_f) * (-1e9)

        # softmax 得到注意力权重
        weights = tf.nn.softmax(scores_f32, axis=1)          # (B, T, 1)

        if mask is not None:
            # 直接把被 mask 的位置权重置 0（全 mask 的样本会得到全 0 权重，reduce_sum 后自然为 0，避免 NaN）
            weights = weights * mask_f

        # 转回原始 dtype（支持 float16 等混合精度训练）
        weights = tf.cast(weights, x.dtype)

        # 加权求和得到池化后的特征向量
        out = tf.reduce_sum(x * weights, axis=1)            # (B, C)

        if self.return_attention:
            # 返回池化结果 + 注意力权重 (B, T)
            return out, tf.squeeze(weights, axis=-1)
        return out

    def compute_mask(self, inputs, mask=None):
        # 池化后不再是序列，不需要传递 mask
        return None

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "return_attention": self.return_attention,
        })
        return config
#
# 注意力池化(基于卷积层) 时频
@tf.keras.utils.register_keras_serializable()
class AttentionPooling_2D_CNN(layers.Layer):
    def __init__(self, hidden_units=None, return_attention=False, **kwargs):
        """
        基于1x1 Conv2D的联合二维注意力池化层（Joint 2D Attention Pooling）。
        对时频图 (B, F, T, C) 在频率和时间两个维度上进行全局加权平均，得到 (B, C)。
        
        参数:
            hidden_units: 中间隐藏维度，默认为 d_model（特征维度 C）。
            return_attention: 是否同时返回注意力权重 (B, F, T)。
        """
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.return_attention = return_attention
        # 不处理 mask
        self.supports_masking = False

    def build(self, input_shape):
        # input_shape: (batch_size, freq, time, channels)
        if len(input_shape) != 4:
            raise ValueError("Input must be 4D tensor (B, F, T, C).")

        d_model = input_shape[-1]
        if d_model is None:
            raise ValueError("The last dimension of input (channel/feature dimension) must be defined.")

        # 默认隐藏维度为 d_model
        hidden = self.hidden_units if self.hidden_units is not None else d_model

        # 先创建子层
        self.conv1 = layers.Conv2D(filters=hidden, kernel_size=1, activation='tanh')
        self.conv2 = layers.Conv2D(filters=1, kernel_size=1, activation=None)

        super().build(input_shape)

    def call(self, x, training=None):
        """
        x: (B, F, T, C)
        """
        # 通过两层 1x1 Conv2D 得到每个 (f,t) 位置的注意力分数
        scores = self.conv1(x, training=training)          # (B, F, T, hidden)
        scores = self.conv2(scores, training=training)     # (B, F, T, 1)

        # 为数值稳定，使用 float32 计算 softmax
        scores_f32 = tf.cast(scores, tf.float32)

        # 在频率和时间维度上全局 softmax
        # 先展平 F 和 T 维度进行 softmax，再恢复形状
        shape = tf.shape(scores_f32)
        scores_flat = tf.reshape(scores_f32, [shape[0], -1, 1])  # (B, F*T, 1)
        weights_flat = tf.nn.softmax(scores_flat, axis=1)        # (B, F*T, 1)
        weights = tf.reshape(weights_flat, shape)               # (B, F, T, 1)

        # 转回原始 dtype
        weights = tf.cast(weights, x.dtype)

        # 加权求和：在 F 和 T 两个轴上聚合
        out = tf.reduce_sum(x * weights, axis=[1, 2])            # (B, C)

        if self.return_attention:
            # 返回池化结果 + 注意力权重 (B, F, T)
            return out, tf.squeeze(weights, axis=-1)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "return_attention": self.return_attention,
        })
        return config
#
# 多尺度扩张卷积 时序 (+通道注意力)
@tf.keras.utils.register_keras_serializable()
class MultiScaleDilatedConv_1D(layers.Layer):
    def __init__(self, filters, kernel_size=3, use_dilations=True, dilations=(1, 2, 4), 
                 use_se=True,r=16,use_eca=False, gamma=2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.r = r
        self.gamma = gamma
        self.use_dilations = use_dilations
        self.use_se = use_se
        self.use_eca = use_eca
        # 扩张卷积
        if self.use_dilations:
            self.dilations = tuple(dilations)
        else:
            self.dilations = (1,)

        self.branches = []
        for d in self.dilations:
            branch = tf.keras.Sequential([
                layers.Conv1D(filters, kernel_size, padding="same", dilation_rate=d, use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
            ])
            self.branches.append(branch)

        self.fuse = tf.keras.Sequential([
            layers.Conv1D(filters, 1, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        # SE / ECA
        if self.use_se:
            if self.use_eca:
                self.se = ECALayer(gamma)
            else:
                self.se = SEBlock(r)
        else:
            self.se = None
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)  # (B, T, C)
        if len(input_shape) != 3:
            raise ValueError(f"请检查输入数据张量是否为(B,T,C)")
        super().build(input_shape)

    def call(self, x, training=None):
        outs = []
        for branch in self.branches:
            outs.append(branch(x, training=training))

        x = tf.concat(outs, axis=-1)            # (B, T, filters * n)
        x = self.fuse(x, training=training)     # (B, T, filters)

        if self.se is not None:
            x = self.se(x, training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "use_dilations":self.use_dilations,
            "dilations": tuple(self.dilations),
            "use_se": self.use_se,
            "r":self.r,
            "use_eca": self.use_eca,
            "gamma": self.gamma,
        })
        return cfg
#
# 多尺度扩张卷积 时频 (+通道注意力、坐标注意力)
@tf.keras.utils.register_keras_serializable()
class MultiScaleDilatedConv_2D(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 7), use_dilations=True, dilations=(1, 2, 4), 
                 use_se=True,r=16,use_eca=False, gamma=2, use_coord= True, reduction=32, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_freq, self.kernel_time = kernel_size
        self.r = r
        self.gamma = gamma
        self.use_dilations = use_dilations
        self.reduction = reduction
        self.use_se = use_se
        self.use_eca = use_eca
        self.use_coord = use_coord

        # 扩张卷积？
        if self.use_dilations:
            self.dilations = tuple(dilations)
        else:
            self.dilations = (1,)

        self.branches = []
        for dr in self.dilations:
            branch = tf.keras.Sequential([
                layers.Conv2D(
                    self.filters,
                    (self.kernel_freq, self.kernel_time),
                    dilation_rate=(1, dr),
                    padding="same",
                    use_bias=False
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
            ])
            self.branches.append(branch)

        self.concat = layers.Concatenate(axis=-1)

        self.fuse = tf.keras.Sequential([
            layers.Conv2D(self.filters, (1, 1), padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        # 通道注意力 SE / ECA
        if self.use_se:
            if self.use_eca:
                self.se = ECALayer(gamma)
            else:
                self.se = SEBlock(r)
        else:
            self.se = None
        # 坐标注意力
        if self.use_coord:
            self.coordinateAttention = CoordinateAttention(reduction)
        else:
            self.coordinateAttention = None
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)  # (B, F, T, C)
        if len(input_shape) != 4:
            raise ValueError(f"请检查输入数据张量是否为(B,H,T,C)")
        super().build(input_shape)

    def call(self, x, training=None):
        # multi-scale branches
        outs = []
        for conv  in self.branches:
            outs.append(conv(x ,training = training))

        x = self.concat(outs)
        x = self.fuse(x, training = training)

        if self.se is not None:
            x = self.se(x, training=training)
        if self.coordinateAttention is not None:
            x = self.coordinateAttention(x, training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "filters": self.filters,
            "kernel_size": (self.kernel_freq, self.kernel_time),
            "use_dilations":self.use_dilations,
            "dilations": tuple(self.dilations),
            "use_se": self.use_se,
            "r":self.r,
            "use_eca": self.use_eca,
            "gamma": self.gamma,
            "use_coord":self.use_coord,
            "reduction":self.reduction,
        })
        return cfg
#
