import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from scipy.signal import detrend, butter, filtfilt
import traceback
# 较 ST_1.2 修改：
# h5文件保存逻辑、绘图逻辑
# 处理流程：
# TXT信号→ 长度统一→ 去平均值→ 带通滤波(10–1500Hz)→ 去趋势→ S变换→ 三通道特征→ min-max归一化（保存时频图）
#                                                   → z-score（保存时域数据）
# 保存的H5文件：时域数据、 三通道时频图 (16, 4000, 3)、label
# 三通道：对数幅度谱 、频率方向梯度谱 、时间方向梯度谱

# 全局定义的类别名称
CLASS_NAMES = {
    0: "岩石破裂",
    1: "爆破",
    2: "噪声"
}

# 信号采样率参数设置
FREQ = 4000  
# 每个H5文件的数据数量
CHUNK_SIZE = 1000
# 是否保存图片
SAVE_PLOT = True
# 全局配置
PLOT_MODE = 'detailed'  # 'detailed'或'compact'
FREQ_SAMPLING_MODE = 'custom'  # 'custom'或'uniform'

# 微震信号频率范围配置
MIN_FREQ = 10     # 微震信号最低有效频率
MAX_FREQ = FREQ // 2  # 微震信号最高有效频率 (2000Hz)
SAFETY_MARGIN = 0.98  # 安全边际系数
DEFAULT_FREQ_STEP = (MAX_FREQ * SAFETY_MARGIN) // 15  # 均匀采样下频率步长
MODE_POINTS = 16    # 采样点数

# 带通滤波参数
LOWCUT = 10      # Hz
HIGHCUT = 1500   # Hz
FILTER_ORDER = 4

def load_single_data(file_path):
    """
    加载单分量微震数据文件
    格式：每行一个浮点数
    """
    try:
        # data = np.loadtxt(file_path)
        # 读取数据（过滤空行和非法字符）
        clean_data = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:    
            for line in f:
                line = line.strip()
                    
                # 跳过空行
                if not line:
                    continue
                # 跳过包含NULL字符的行
                if '\x00' in line:
                    continue
                try:
                    value = float(line)
                    clean_data.append(value)
                except ValueError:
                    # 遇到无法转换的行直接跳过
                    continue    

        data = np.array(clean_data)    

        if len(data) < 4000:
            print(f"警告: 文件 {file_path} 只有 {len(data)} 行数据，不足4000行，填充零值。")
            padded_data = np.zeros(4000)
            padded_data[:len(data)] = data
            data = padded_data
        else:
            data = data[:4000]

        time_s = np.arange(0, 1.0, 1/FREQ)
        return time_s, data, FREQ

    except Exception as e:
        print(f"错误加载文件 {file_path}: {str(e)}")
        return None, None, None

def frequency_sampling(Fs, N, min_freq=MIN_FREQ, max_freq=MAX_FREQ, mode=FREQ_SAMPLING_MODE):
    """
    频率采样选择器 - 支持自定义分辨率和标准均匀分辨率
    """
    # 确保不超过奈奎斯特频率
    max_freq = min(max_freq, Fs / 2 * SAFETY_MARGIN)  # 保留安全边际
    
    if mode == 'custom':
        return custom_frequency_sampling(min_freq, max_freq, Fs, N)
    else:
        return uniform_frequency_sampling(min_freq, max_freq, Fs, N)

def custom_frequency_sampling(min_freq, max_freq,Fs, N, mode_points=MODE_POINTS):
    """
    自定义频率采样 - 改为固定稀疏频率方案 (支持8点或16点)
    mode_points: 8 或 16，控制采样点数量
    """
    # ========== 固定频率方案 ==========
    if mode_points == 8:
        # 8点频率(Hz)
        freqs_arr = np.array([10, 200, 400, 600, 800, 1100, 1300, 1500], dtype=float)
    elif mode_points == 16:
        # 16点频率(Hz)
        freqs_arr = np.array([10,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500
        ], dtype=float)
    else:
        raise ValueError("mode_points 必须是 8 或 16")

    # 安全边际：不能超过Fs/2 * SAFETY_MARGIN
    max_freq_allowed = Fs / 2 * SAFETY_MARGIN
    freqs_arr = freqs_arr[freqs_arr <= max_freq_allowed]

    # ========== 将频率转换为FFT索引 ==========
    k_indices_arr = np.round(freqs_arr * N / Fs).astype(int)

    # 提示信息
    print(f"[Custom Sampling] 模式: {mode_points}点 | 频率范围: {freqs_arr[0]}-{freqs_arr[-1]} Hz")
    print(f"[Custom Sampling] 频率点数: {len(freqs_arr)} | 频率列表: {freqs_arr}")

    return freqs_arr, k_indices_arr

def uniform_frequency_sampling(min_freq, max_freq, Fs, N):
    """
    标准均匀频率采样 - 确保包含最小和最大频率点
    """
    # 创建频率索引映射表：freq -> k
    freq_k_map = {}
    
    # 确保不超过奈奎斯特频率（保留安全边际）
    max_freq = min(max_freq, Fs / 2 * SAFETY_MARGIN)
    
    # 生成频率点 - 确保包含最小和最大频率点
    freqs = np.arange(min_freq, max_freq + DEFAULT_FREQ_STEP, DEFAULT_FREQ_STEP)
    
    # 确保不超过最大频率
    freqs = freqs[freqs <= max_freq]
    
    # 特别添加0Hz（零频分量）
    if min_freq > 0 and 0 not in freqs:
        freqs = np.insert(freqs, 0, 0)
    
    # 过滤超出奈奎斯特频率的点
    freqs = freqs[freqs < Fs/2]
    
    # 计算每个频率对应的索引
    k_indices = np.round(freqs * N / Fs).astype(int)
    
    # 创建频率到索引的映射表
    for freq, k in zip(freqs, k_indices):
        if 0 <= k < N:
            freq_k_map[freq] = k
    
    # 确保0Hz被包含（零频分量）
    if 0 not in freq_k_map:
        freq_k_map[0] = 0
        freqs = np.insert(freqs, 0, 0)
        k_indices = np.insert(k_indices, 0, 0)
    
    freqs_arr = np.array(sorted(freq_k_map.keys()))
    k_indices_arr = [freq_k_map[f] for f in freqs_arr]
    
    # 显示配置信息
    num_points = len(freqs_arr)
    if num_points > 1:
        resolution = (freqs_arr[-1] - freqs_arr[0]) / (num_points - 1)
    else:
        resolution = 0
    
    print(f"  均匀频率采样: 步长={DEFAULT_FREQ_STEP:.1f}Hz, 点数={num_points}, "
          f"频率范围={freqs_arr[0]:.1f}-{freqs_arr[-1]:.1f}Hz, "
          f"平均分辨率={resolution:.2f}Hz/点")
    
    return freqs_arr, k_indices_arr

def bandpass_filter(x, fs, lowcut=LOWCUT, highcut=HIGHCUT, order=FILTER_ORDER):
    """
    Butterworth 零相位带通滤波（filtfilt）
    """
    x = np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0

    # 边界保护：避免 0 或 >= Nyquist
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999999)

    if low >= high:
        raise ValueError(f"带通滤波参数非法: lowcut={lowcut}, highcut={highcut}, fs={fs}")

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, x)

def preprocess_signal(raw, fs):
    """
    流程：
    去平均值 → 带通滤波(10–1500Hz) → 去趋势
    返回：预处理后的信号（用于 ST），以及 z-score 后的时域信号（用于保存）
    """
    x = np.asarray(raw, dtype=np.float64)

    # 1) 去平均值（去 DC）
    x = x - np.mean(x)

    # 2) 带通滤波
    x = bandpass_filter(x, fs, LOWCUT, HIGHCUT, FILTER_ORDER)

    # 3) 去趋势（线性）
    x = detrend(x, type='linear')

    # 4) z-score（仅用于保存时域信号）
    mean = x.mean()
    std = x.std()
    x_z = (x - mean) / (std + 1e-12)

    return x, x_z.astype(np.float32)

def enhanced_stockwell(signal_data, Fs, add_hanning=False):
    """
    标准离散 S-transform（Stockwell et al., 1996）的频域实现（稀疏频率也支持）
    返回:
        st_complex: (M, N)  M=frequency_sampling 给出的频率点数
        freqs_hz:   (M,)    与 st 行一一对应的“实际 FFT bin 频率”(Hz)
    """
    x = np.asarray(signal_data)
    N = x.size

    # 1) 可选预处理（注意：这会改变严格的“标准定义”，但工程上可用）
    if add_hanning:
        x = x * np.hanning(N)

    # 2) FFT（NumPy 约定：ifft 内含 1/N 归一化）
    X = np.fft.fft(x)

    # 3) 取你自定义的稀疏频率采样
    freqs_target_hz, k_indices = frequency_sampling(Fs, N)
    k_indices = np.asarray(k_indices, dtype=int)

    # 只做正频率（若你只关心 0..Fs/2，这是最常见的做法）
    k_indices = np.clip(k_indices, 0, N // 2)

    # 去重（保序），避免重复 voice 或 round 造成重复
    seen = set()
    k_clean = []
    for k in k_indices.tolist():
        k = int(k)
        if k not in seen:
            seen.add(k)
            k_clean.append(k)
    k_indices = np.array(k_clean, dtype=int)

    # 输出频率用“实际 bin 频率”，避免标注偏差
    freqs_hz = k_indices * Fs / N

    # 4) m：频率偏移索引，顺序要与 numpy FFT 的 bin 排列一致
    # numpy FFT/ifft 的频率 bin 顺序：0, +1..+N/2, -N/2+1..-1
    half = N // 2
    m = (np.arange(N) + half) % N - half  # [0,1,2,...,half-1,-half,...,-1]

    st_complex = np.zeros((len(k_indices), N), dtype=np.complex128)

    for i, n in enumerate(k_indices):
        if n == 0:
            # Eq.(12): DC voice 是均值 
            st_complex[i, :] = np.mean(x)
            continue

        # Eq.(11)(14): voice Gaussian（零中心）
        gauss0 = np.exp(-2.0 * (np.pi**2) * (m**2) / (n**2))

        # Eq.(11): H[m+n] -> 频谱循环平移 + 乘窗 + IFFT 
        st_complex[i, :] = np.fft.ifft(np.roll(X, -n) * gauss0)

    return st_complex, freqs_hz

def create_single_spectrogram(st_complex):
    """
    创建单分量微震信号三通道时频图
    通道1: 对数幅度谱
    通道2: 频率梯度幅度谱
    通道3: 时间梯度幅度谱
    """
    mag = np.abs(st_complex)                         
    log_mag = np.log1p(mag)                            

    # 时间方向梯度
    grad_t = np.gradient(log_mag, axis=1)
    grad_t = np.abs(grad_t)  # 取绝对值，更鲁棒

    # 频率方向梯度
    grad_f = np.gradient(log_mag, axis=0)
    grad_f = np.abs(grad_f)


    channel1 = log_mag
    channel2 = grad_f
    channel3 = grad_t

    # 归一化
    def minmax_norm(arr, eps=1e-12):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min > eps:
            return (arr - arr_min) / (arr_max - arr_min)
        else:
            return np.zeros_like(arr)

    channel1 = minmax_norm(channel1)
    channel2 = minmax_norm(channel2)
    channel3 = minmax_norm(channel3)

    # (16,4000,3)
    st_image = np.stack([channel1, channel2, channel3], axis=-1)  
    return st_image.astype(np.float32)

def plot_record_with_spectrogram(time_s, raw_signal, proc_signal, st_image, freqs, label, save_path=None):
    """
    绘制原始信号、预处理信号以及三通道时频图
    """

    # label_name = CLASS_NAMES.get(label, "未知标签")
    file_name = os.path.basename(save_path) if save_path else "record"

    extent = [time_s[0], time_s[-1], 0, len(freqs) - 1]
    yticks = np.arange(len(freqs))
    yticklabels = [f"{f:.0f}" for f in freqs]

    if PLOT_MODE == 'detailed':

        fig, axes = plt.subplots(4, 1, figsize=(18, 16), constrained_layout=True)

        fig.suptitle(f"{file_name} ", fontsize=16)

        # ===== 原始 vs 预处理信号 =====
        axes[0].plot(time_s, raw_signal, alpha=0.7)
        # axes[0].plot(time_s, proc_signal, label='Preprocessed Signal', alpha=0.8)

        axes[0].set_title('Original Signal vs Preprocess Signal')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Velocity(m/s)')
        axes[0].legend()
        axes[0].grid(True)

        # ===== 通道1: 对数幅度谱 =====
        im0 = axes[1].imshow(
            st_image[:, :, 0],
            aspect='auto',
            origin='lower',
            extent=extent,
            interpolation='nearest',
            cmap='viridis'
        )

        axes[1].set_yticks(yticks)
        axes[1].set_yticklabels(yticklabels)

        axes[1].set_title('Logarithmic Amplitude Spectrum log(1+|S|)')
        axes[1].set_ylabel('Frequency (Hz)')

        fig.colorbar(im0, ax=axes[1])

        # ===== 通道2: 频率梯度谱 =====
        im1 = axes[2].imshow(
            st_image[:, :, 1],
            aspect='auto',
            origin='lower',
            extent=extent,
            interpolation='nearest',
            cmap='inferno'
        )

        axes[2].set_yticks(yticks)
        axes[2].set_yticklabels(yticklabels)

        axes[2].set_title('Frequency Gradient Spectrum |∂/∂f log|S||')
        axes[2].set_ylabel('Frequency (Hz)')

        fig.colorbar(im1, ax=axes[2])

        # ===== 通道3: 时间梯度谱 =====
        im2 = axes[3].imshow(
            st_image[:, :, 2],
            aspect='auto',
            origin='lower',
            extent=extent,
            interpolation='nearest',
            cmap='magma'
        )

        axes[3].set_yticks(yticks)
        axes[3].set_yticklabels(yticklabels)

        axes[3].set_title('Time Gradient Spectrum |∂/∂t log|S||')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Frequency (Hz)')

        fig.colorbar(im2, ax=axes[3])

    else:

        plt.figure(figsize=(18, 10))
        plt.suptitle(f"{file_name}", fontsize=16)

        plt.subplot(2, 2, 1)

        plt.plot(time_s, raw_signal, label='Raw Signal', alpha=0.7)
        plt.plot(time_s, proc_signal, label='Preprocessed Signal', alpha=0.8)

        plt.title('original signal vs Preprocess signal')
        plt.xlabel('Time (s)')
        plt.ylabel('amplitude')
        plt.legend()
        plt.grid(True)

        for i, (title, cmap) in enumerate(
            zip(
                ['Logarithmic amplitude spectrum', 'Frequency gradient spectrum', 'Time gradient spectrum'],
                ['viridis', 'twilight', 'inferno']
            ), 2
        ):

            plt.subplot(2, 2, i)

            plt.imshow(
                st_image[:, :, i - 2],
                aspect='auto',
                origin='lower',
                extent=[time_s[0], time_s[-1], freqs[0], freqs[-1]],
                cmap=cmap
            )

            plt.title(title)
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar()

        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

def get_label_from_user(file_name):
    """
    从用户获取单条记录的标签
    """

    print("\n" + "="*50)
    print(f"请为文件 {file_name} 分配标签:")
    print("0:岩石破裂, 1:爆破, 2:噪声")
    print("="*50)
    
    # 自动分配标签
    if INPUT_DIR==INPUT_DIR2:
        label = 2
        print(f"输入文件路径为噪音文件,自动分配标签: {label}")
        print("="*50)
        return label
    elif INPUT_DIR==INPUT_DIR0:
        label = 0
        print(f"输入文件路径为岩石破裂文件，自动分配标签: {label}")
        print("="*50)
        return label
    elif INPUT_DIR==INPUT_DIR1:
        label = 1
        print(f"输入文件路径为岩石爆破文件，自动分配标签: {label}")
        print("="*50)
        return label
    
    
    # 手动输入标签
    while True:
        user_input = input("请输入标签 (0,1或2): ")
        try:
            label = int(user_input.strip())
            if label in [0, 1, 2]:
                print(f"标签分配成功: {CLASS_NAMES[label]}")
                return label
            else:
                print("错误: 标签值只能是0,1或2")
        except ValueError:
            print("错误: 输入格式不正确，请输入数字")

def open_new_h5(out_dir, class_name, chunk_id, freqs):
    h5_path = os.path.join(out_dir, f"{class_name}_{chunk_id:03d}.h5")
    hf = h5py.File(h5_path, "w")

    # 三通道时频图
    image_ds = hf.create_dataset(
        "st_image",
        shape=(CHUNK_SIZE, 16, 4000, 3),
        maxshape=(None, 16, 4000, 3),
        dtype="float32",
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        chunks=(1, 16, 4000, 3)
    )
    # 标签
    label_ds = hf.create_dataset(
        "label",
        shape=(CHUNK_SIZE,),
        maxshape=(None,),
        dtype="int64",
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        chunks=(1024,)
    )
    # 预处理后的时域信号
    x_ds = hf.create_dataset(
        "st_component",
        shape=(CHUNK_SIZE, 4000),
        maxshape=(None, 4000),
        dtype="float32",
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        chunks=(1, 4000)
    )

    # 保存原始文件名
    str_dt = h5py.string_dtype(encoding="utf-8")
    fn_ds = hf.create_dataset(
        "file_name",
        shape=(CHUNK_SIZE,),
        maxshape=(None,),
        dtype=str_dt,
        chunks=(1024,)
    )

    # 元信息
    hf.attrs["sampling_rate"] = FREQ
    hf.attrs["channels"] = "0:log|S|, 1:|d/df log|S||, 2:|d/dt log|S||"

    hf.create_dataset("freqs", data=freqs.astype(np.float32))
    return hf, image_ds, label_ds, x_ds, fn_ds, h5_path

def close_h5_file(hf, image_ds, label_ds, x_ds, fn_ds, write_idx):
    """
    将当前HDF5文件裁剪到实际写入长度并关闭
    """
    if hf is not None:
        image_ds.resize((write_idx, 16, 4000, 3))
        label_ds.resize((write_idx,))
        x_ds.resize((write_idx, 4000))
        fn_ds.resize((write_idx,))
        hf.close()

def save_record_to_h5(record, output_dir, hf, image_ds, label_ds, x_ds, fn_ds, chunk_id, write_idx, current_h5_path):
    """
    将单条record写入HDF5。
    若当前HDF5为空或已写满，则自动关闭旧文件并创建新文件。

    返回：
        hf, image_ds, label_ds, x_ds, fn_ds, chunk_id, write_idx, current_h5_path, result_info
    """
    # 达到上限后关旧开新
    if hf is None or write_idx >= CHUNK_SIZE:
        if hf is not None:
            close_h5_file(hf, image_ds, label_ds, x_ds, fn_ds, write_idx)

        label = record["label"]
        class_name = CLASS_NAMES.get(label, f"class{label}")
        hf, image_ds, label_ds, x_ds, fn_ds, current_h5_path = open_new_h5(
            output_dir, class_name, chunk_id, record["freqs"]
        )
        chunk_id += 1
        write_idx = 0

    # 写入当前样本
    image_ds[write_idx] = record["st_image"]
    label_ds[write_idx] = record["label"]
    x_ds[write_idx] = record["st_component"]
    fn_ds[write_idx] = record["file_name"]

    result_info = {
        "file_name": record["file_name"],
        "label": record["label"],
        "h5_file": os.path.basename(current_h5_path),
        "index_in_h5": write_idx
    }

    write_idx += 1

    return hf, image_ds, label_ds, x_ds, fn_ds, chunk_id, write_idx, current_h5_path, result_info

def process_single_file(file_path, output_dir):
    """
    处理单个TXT文件，包含一条单分量记录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载文件数据
    time_s, data, Fs = load_single_data(file_path)
    
    if time_s is None:
        print(f"文件加载失败，跳过: {file_path}")
        return None
    
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]
    
    # 从用户获取标签
    label = get_label_from_user(file_name)
    print(f"处理文件: {file_name} - 标签: {CLASS_NAMES.get(label, '未知')}")
    
   # ===== 预处理 =====
    proc, proc_z = preprocess_signal(data, Fs)

    # ===== S变换 =====
    print("  S变换...")
    st_complex, freqs = enhanced_stockwell(proc, Fs)

    print("  创建三通道时频图...")
    st_image = create_single_spectrogram(st_complex)    
    
    # 绘制结果
    if SAVE_PLOT:
        save_path = os.path.join(output_dir, f"{base_name}.png")
        plot_record_with_spectrogram(time_s, data, proc, st_image, freqs, label, save_path)
    
    return {
    'file_name': file_name,
    'time_s': time_s,
    'signal': data,                     # 原始数据
    'st_component': proc_z,     #预处理后的时域信号
    'label': label,
    'st_image': st_image,
    'freqs': freqs
}

def batch_process_directory(input_dir, output_dir):
    """
    批量处理目录中的所有TXT文件 单分量
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有TXT文件
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"在 {input_dir} 中未找到TXT文件!")
        return
    
    results = []
    chunk_id = 0
    write_idx = 0
    hf = image_ds = label_ds = x_ds = fn_ds = None
    current_h5_path = None

    # 处理每个文件
    for i, file in enumerate(txt_files):
        file_path = os.path.join(input_dir, file)
        print(f"\n处理文件 {i+1}/{len(txt_files)}: {file}")
        print(f"使用微震信号频率范围: {MIN_FREQ}-{MAX_FREQ}Hz")
                
        try:
            record = process_single_file(file_path, output_dir)

            if record is None:
                continue

            hf, image_ds, label_ds, x_ds, fn_ds, chunk_id, write_idx, current_h5_path, result_info = save_record_to_h5(
                record=record,
                output_dir=output_dir,
                hf=hf,
                image_ds=image_ds,
                label_ds=label_ds,
                x_ds=x_ds,
                fn_ds=fn_ds,
                chunk_id=chunk_id,
                write_idx=write_idx,
                current_h5_path=current_h5_path
            )

            results.append(result_info)

            if (i + 1) % 50 == 0:
                print(f"[进度] 已完成 {i+1}/{len(txt_files)}，当前H5: {current_h5_path}，write_idx={write_idx}")

        except Exception as e:
            print(f"[错误] 处理文件失败: {file_path}")
            print(f"[错误信息] {e}")
            traceback.print_exc()
            continue

    # 收尾关闭最后一个HDF5文件
    close_h5_file(hf, image_ds, label_ds, x_ds, fn_ds, write_idx)

    # 保存汇总元数据
    summary_file = os.path.join(output_dir, "batch_summary.csv")
    with open(summary_file, 'w') as sf:
        sf.write("file_name,label,h5_file,index_in_h5,sampling_mode,min_freq,max_freq\n")
        for res in results:
            sf.write(f"{res['file_name']},{res['label']},{res['h5_file']},{res['index_in_h5']},"
            f"{FREQ_SAMPLING_MODE},{MIN_FREQ},{MAX_FREQ}\n")

    
    print(f"批量处理完成! 共处理 {len(results)} 个文件.")
    return results

# 岩石破裂文件目录
INPUT_DIR0 = os.environ.get("CLASSIFY_RAW_ROCK", "data/raw/rock")
# 爆破文件目录
INPUT_DIR1 = os.environ.get("CLASSIFY_RAW_BLAST", "data/raw/blast")
# 噪音文件目录
INPUT_DIR2 = os.environ.get("CLASSIFY_RAW_NOISE", "data/raw/noise")

OUTPUT_DIR0 = os.environ.get("CLASSIFY_H5_ROCK", "data/processed/rock")
OUTPUT_DIR1 = os.environ.get("CLASSIFY_H5_BLAST", "data/processed/blast")
OUTPUT_DIR2 = os.environ.get("CLASSIFY_H5_NOISE", "data/processed/noise")

if __name__ == "__main__":
    # 配置路径
    INPUT_DIR = os.environ.get("CLASSIFY_INPUT_DIR", INPUT_DIR2)
    OUTPUT_DIR = os.environ.get("CLASSIFY_PREPROCESS_OUTPUT_DIR", OUTPUT_DIR2)

    # 显示微震信号处理配置
    print("="*50)
    print(f"微震信号处理配置")
    print(f"频率采样模式: {FREQ_SAMPLING_MODE}")
    print(f"最小频率: {MIN_FREQ}Hz")
    print(f"最大频率: {MAX_FREQ}Hz")
    print("="*50)
    
    # 运行批处理
    batch_process_directory(INPUT_DIR, OUTPUT_DIR)