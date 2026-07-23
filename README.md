# Nhận dạng 4 loại điều chế tín hiệu vô tuyến bằng mạng MLP lượng tử hóa trên PYNQ-Z2

> **Tên repository GitHub đề xuất:** `pynq-z2-quantized-mlp-amc-4class`  
> **Tên dự án Vivado:** `AMC_MLP_16_32_16_4_FPGA`  
> **Bo mạch:** PYNQ-Z2 — Zynq-7000 XC7Z020  
> **Vivado:** 2023.1  
> **Luồng tích hợp:** Đóng gói RTL thành **Custom IP**, đưa IP vào Block Design, tạo bitstream và chạy bằng PYNQ/Jupyter.

---

## 1. Giới thiệu

Dự án xây dựng một hệ thống **Automatic Modulation Classification — AMC** để nhận dạng bốn loại điều chế tín hiệu vô tuyến:

| Mã lớp | Loại điều chế |
|---:|---|
| `0` | BPSK |
| `1` | QPSK |
| `2` | QAM16 |
| `3` | GFSK |

Mô hình học máy là mạng **Multilayer Perceptron — MLP** có kiến trúc cố định:

```text
16 đặc trưng
    │
    ▼
Dense 1: 16 → 32
    │
    ▼
ReLU
    │
    ▼
Dense 2: 32 → 16
    │
    ▼
ReLU
    │
    ▼
Dense 3: 16 → 4
    │
    ▼
Argmax
    │
    ▼
class_id ∈ {0, 1, 2, 3}
```

Mạng được huấn luyện trên máy tính, fuse Batch Normalization, lượng tử hóa sang fixed-point, chuyển trọng số thành các file `.mem`, sau đó tổng hợp thành phần cứng trên FPGA.

---

## 2. Mục tiêu kỹ thuật

Dự án thực hiện toàn bộ chuỗi sau:

```text
RadioML 2016.10A
        │
        ▼
Lọc 4 modulation và SNR > 0 dB
        │
        ▼
Trích xuất 16 đặc trưng
        │
        ▼
StandardScaler
        │
        ▼
Huấn luyện MLP 16–32–16–4
        │
        ▼
Fuse BatchNorm vào Dense
        │
        ▼
Lượng tử fixed-point
        │
        ▼
Xuất 6 file .mem
        │
        ▼
RTL Verilog + AXI4-Lite
        │
        ▼
Đóng gói thành Custom IP
        │
        ▼
Vivado Block Design
        │
        ▼
.bit + .hwh + .bin
        │
        ▼
PYNQ Overlay + Jupyter
```

---

## 3. Đặc điểm chính

- Nhận dạng bốn loại điều chế: **BPSK, QPSK, QAM16, GFSK**.
- Đầu vào FPGA là vector **16 đặc trưng**, không phải trực tiếp 256 mẫu I/Q.
- MLP cố định: **16 → 32 → 16 → 4**.
- Batch Normalization được fuse trước khi lượng tử hóa.
- ReLU được triển khai bằng logic so sánh dấu.
- Không dùng Dropout khi inference.
- Không cần Softmax trên FPGA; kết quả được lấy bằng Argmax.
- Weight dùng signed 16-bit.
- Bias dùng signed 32-bit.
- Accumulator dùng signed 48-bit.
- Điều khiển từ Processing System qua AXI4-Lite.
- Trọng số được nạp vào ROM bằng `$readmemh()` và nhúng vào bitstream.
- Kết quả phân lớp có thể hiển thị bằng bốn LED trên PYNQ-Z2.

---

# PHẦN I — MÔ HÌNH MLP

## 4. Cấu trúc mạng neural network

### 4.1 Kích thước từng lớp

| Lớp | Kích thước vào | Kích thước ra | Activation |
|---|---:|---:|---|
| Dense 1 | 16 | 32 | ReLU |
| Dense 2 | 32 | 16 | ReLU |
| Dense 3 | 16 | 4 | Không |
| Output | 4 logits | 1 class ID | Argmax |

### 4.2 Số lượng tham số

#### Dense 1

```text
Weight: 32 × 16 = 512
Bias  : 32
```

#### Dense 2

```text
Weight: 16 × 32 = 512
Bias  : 16
```

#### Dense 3

```text
Weight: 4 × 16 = 64
Bias  : 4
```

#### Tổng cộng

```text
Tổng weight = 512 + 512 + 64 = 1088
Tổng bias   = 32 + 16 + 4    = 52
Tổng tham số                        = 1140
```

Với weight 16-bit và bias 32-bit:

```text
Dung lượng weight = 1088 × 2 byte = 2176 byte
Dung lượng bias   =   52 × 4 byte =  208 byte
Tổng dữ liệu tham số              = 2384 byte
```

---

## 5. Công thức suy luận MLP

Gọi vector đặc trưng đã chuẩn hóa là:

```math
\mathbf{x} \in \mathbb{R}^{16}
```

### 5.1 Dense 1

```math
\mathbf{z}_1 = \mathbf{W}_1\mathbf{x} + \mathbf{b}_1
```

```math
\mathbf{W}_1 \in \mathbb{R}^{32 \times 16},\qquad
\mathbf{b}_1 \in \mathbb{R}^{32}
```

Sau ReLU:

```math
\mathbf{a}_1 = \operatorname{ReLU}(\mathbf{z}_1)
```

```math
\operatorname{ReLU}(u)=\max(0,u)
```

### 5.2 Dense 2

```math
\mathbf{z}_2 = \mathbf{W}_2\mathbf{a}_1 + \mathbf{b}_2
```

```math
\mathbf{W}_2 \in \mathbb{R}^{16 \times 32}
```

```math
\mathbf{a}_2 = \operatorname{ReLU}(\mathbf{z}_2)
```

### 5.3 Dense 3

```math
\mathbf{z}_3 = \mathbf{W}_3\mathbf{a}_2 + \mathbf{b}_3
```

```math
\mathbf{W}_3 \in \mathbb{R}^{4 \times 16}
```

Vector đầu ra:

```math
\mathbf{z}_3 = [z_{3,0},z_{3,1},z_{3,2},z_{3,3}]
```

### 5.4 Argmax

```math
\hat{y} = \underset{k \in \{0,1,2,3\}}{\operatorname{argmax}} \; z_{3,k}
```

Không cần Softmax vì Softmax không làm thay đổi vị trí phần tử lớn nhất:

```math
\operatorname{argmax}(\mathbf{z})
=
\operatorname{argmax}(\operatorname{softmax}(\mathbf{z}))
```

---

## 6. Fuse Batch Normalization

Trong mô hình huấn luyện, BatchNorm có dạng:

```math
y =
\gamma
\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}
+\beta
```

Giả sử đầu vào BatchNorm đến từ Dense:

```math
x = Wa + b
```

Có thể fuse BatchNorm vào Dense bằng:

```math
W_{\text{fused}}
=
\frac{\gamma}{\sqrt{\sigma^2+\varepsilon}}W
```

```math
b_{\text{fused}}
=
\frac{\gamma}{\sqrt{\sigma^2+\varepsilon}}(b-\mu)+\beta
```

Sau khi fuse:

```text
Dense → BatchNorm
```

được thay bằng:

```text
Dense với W_fused và b_fused
```

FPGA không cần triển khai riêng phép chia hoặc căn bậc hai của BatchNorm.

---

# PHẦN II — TIỀN XỬ LÝ VÀ LƯỢNG TỬ

## 7. Dữ liệu đầu vào

Nguồn dữ liệu:

```text
RML2016.10a_dict.pkl
```

Mỗi mẫu thường có dạng:

```text
[2, 128]
```

Trong đó:

```text
Kênh 0: I — In-phase
Kênh 1: Q — Quadrature
```

Dự án lọc:

```text
Modulation: BPSK, QPSK, QAM16, GFSK
SNR       : > 0 dB
```

Phân chia dữ liệu:

```text
Training   : 70%
Validation : 15%
Testing    : 15%
```

Cần sử dụng cùng:

- Thứ tự dữ liệu.
- Class mapping.
- `random_state`.
- Phương pháp `stratify`.
- Pipeline trích xuất đặc trưng.
- File StandardScaler.

Nếu thay đổi một trong các thành phần trên, tập test hoặc đầu vào FPGA có thể không còn khớp với mô hình đã huấn luyện.

---

## 8. Vector 16 đặc trưng

FPGA nhận đúng 16 giá trị:

```text
x[0], x[1], ..., x[15]
```

Thứ tự vector phải giống tuyệt đối thứ tự đã dùng khi huấn luyện. Pipeline dự án bắt đầu với đặc trưng `mean_amplitude` và kết thúc bằng `circularity_coefficient`.

Không được:

- Đổi thứ tự đặc trưng.
- Bỏ một đặc trưng.
- Thêm đặc trưng mới.
- Dùng scaler khác.
- Gửi trực tiếp dữ liệu I/Q vào AXI MLP.

Luồng đúng:

```text
I/Q [2,128]
   │
   ▼
Hàm extract_16_features()
   │
   ▼
Vector FP32 [16]
   │
   ▼
StandardScaler
   │
   ▼
Vector scaled [16]
   │
   ▼
Fixed-point int16
```

---

## 9. StandardScaler

Với đặc trưng thứ `i`:

```math
x_i^{(\text{scaled})}
=
\frac{x_i-\mu_i}{s_i}
```

Trong đó:

- `μᵢ`: mean của đặc trưng trong tập train.
- `sᵢ`: standard deviation hoặc scale do StandardScaler lưu.

File scaler cố định:

```text
amc_feature_scaler_v2_16_snr_gt0.npz
```

Không tính lại scaler từ tập test.

---

## 10. Lượng tử fixed-point

Thiết kế không được mô tả là INT8. Định dạng phần cứng hiện tại:

```text
Input/activation : signed int16
Weight           : signed int16
Bias             : signed int32
Accumulator      : signed 48-bit
```

### 10.1 Lượng tử input

Với `F_x` bit phần lẻ:

```math
x_q = \operatorname{round}\left(x_f 2^{F_x}\right)
```

Sau đó saturate:

```math
x_q = \operatorname{clip}(x_q,-32768,32767)
```

Ví dụ Python:

```python
INPUT_FRAC_BITS = 8
INPUT_SCALE = 2 ** INPUT_FRAC_BITS

features_q = np.rint(features_scaled * INPUT_SCALE)
features_q = np.clip(features_q, -32768, 32767).astype(np.int16)
```

Giá trị `INPUT_FRAC_BITS` thực tế phải khớp với:

```text
amc_quantization_params.vh
```

### 10.2 Lượng tử weight

Với `F_w` bit phần lẻ:

```math
w_q = \operatorname{round}\left(w_f 2^{F_w}\right)
```

### 10.3 Lượng tử bias

Bias phải cùng scale với accumulator:

```math
b_q = \operatorname{round}\left(b_f 2^{F_x+F_w}\right)
```

### 10.4 MAC fixed-point

```math
acc_q = b_q + \sum_i x_{q,i}w_{q,i}
```

Sau đó dịch phải số học:

```math
y_q = acc_q \gg F_w
```

Trong RTL:

```verilog
output_value = accumulator >>> OUTPUT_SHIFT;
```

`OUTPUT_SHIFT` phải khớp với số bit phần lẻ của weight tại từng layer.

### 10.5 Saturation

Sau dịch phải, giá trị được giới hạn về signed 16-bit:

```math
y_q =
\begin{cases}
32767, & y_q > 32767\\
-32768, & y_q < -32768\\
y_q, & \text{ngược lại}
\end{cases}
```

---

## 11. Mã bù hai trong file `.mem`

Weight 16-bit:

```text
1    → 0001
-1   → FFFF
-2   → FFFE
```

Bias 32-bit:

```text
1    → 00000001
-1   → FFFFFFFF
```

Mỗi dòng của `.mem` chứa một word hexadecimal.

---

## 12. Sáu file trọng số

```text
layer1_weights.mem
layer1_bias.mem
layer2_weights.mem
layer2_bias.mem
layer3_weights.mem
layer3_bias.mem
```

| File | Số dòng | Độ rộng mỗi dòng |
|---|---:|---:|
| `layer1_weights.mem` | 512 | 4 ký tự hex |
| `layer1_bias.mem` | 32 | 8 ký tự hex |
| `layer2_weights.mem` | 512 | 4 ký tự hex |
| `layer2_bias.mem` | 16 | 8 ký tự hex |
| `layer3_weights.mem` | 64 | 4 ký tự hex |
| `layer3_bias.mem` | 4 | 8 ký tự hex |

Thứ tự weight:

```text
address = neuron_index × INPUT_SIZE + input_index
```

Tương đương NumPy:

```python
weight_q.reshape(-1, order="C")
```

---

# PHẦN III — KIẾN TRÚC RTL VERILOG

## 13. Sơ đồ phân cấp RTL

```text
AMC_MLP_16_32_16_4_AXI_v1_0
│
├── axi_slave_regs
│   ├── AXI4-Lite write channel
│   ├── AXI4-Lite read channel
│   ├── CONTROL
│   ├── FEATURE_INDEX
│   ├── FEATURE_DATA
│   ├── STATUS
│   └── RESULT
│
└── amc_mlp_core
    │
    ├── input_buffer
    │
    ├── layer1_weight_rom
    ├── dense1_mac
    ├── relu1
    │
    ├── layer2_weight_rom
    ├── dense2_mac
    ├── relu2
    │
    ├── layer3_weight_rom
    ├── dense3_mac
    │
    ├── argmax4
    └── mlp_controller
```

---

## 14. Danh sách module Verilog

| STT | Module | Chức năng |
|---:|---|---|
| 1 | `input_buffer.v` | Lưu 16 feature đầu vào |
| 2 | `layer1_weight_rom.v` | ROM 512 weight và 32 bias của Dense 1 |
| 3 | `dense1_mac.v` | Tính Dense 1: 16 → 32 |
| 4 | `relu1.v` | ReLU đầu ra Dense 1 |
| 5 | `layer2_weight_rom.v` | ROM 512 weight và 16 bias của Dense 2 |
| 6 | `dense2_mac.v` | Tính Dense 2: 32 → 16 |
| 7 | `relu2.v` | ReLU đầu ra Dense 2 |
| 8 | `layer3_weight_rom.v` | ROM 64 weight và 4 bias của Dense 3 |
| 9 | `dense3_mac.v` | Tính Dense 3: 16 → 4 |
| 10 | `argmax4.v` | Chọn logit lớn nhất |
| 11 | `mlp_controller.v` | FSM điều khiển toàn bộ inference |
| 12 | `axi_slave_regs.v` | Thanh ghi và giao thức AXI4-Lite |
| 13 | `amc_mlp_core.v` | Ghép datapath MLP |
| 14 | `AMC_MLP_16_32_16_4_AXI_v1_0.v` | Top-level Custom IP |

---

## 15. Input buffer

`input_buffer.v` chứa:

```text
16 × signed 16-bit
```

Phần mềm ghi lần lượt:

```text
feature_index = 0
feature_data  = x[0]

feature_index = 1
feature_data  = x[1]

...

feature_index = 15
feature_data  = x[15]
```

Sau khi đủ 16 phần tử:

```text
input_full = 1
```

---

## 16. Weight ROM

Ví dụ:

```verilog
initial begin
    $readmemh(WEIGHT_FILE, weight_mem);
    $readmemh(BIAS_FILE, bias_mem);
end
```

Tên mặc định:

```verilog
parameter WEIGHT_FILE = "layer1_weights.mem";
parameter BIAS_FILE   = "layer1_bias.mem";
```

Các file `.mem` phải được thêm vào Vivado dưới dạng:

```text
Memory Initialization Files
```

và bật:

```text
USED_IN_SYNTHESIS = 1
```

Weight được nhúng vào netlist và bitstream trong quá trình synthesis.

---

## 17. Dense MAC

Mỗi neuron thực hiện:

```math
acc = bias[n] + \sum_{i=0}^{N-1} input[i] \times weight[n][i]
```

FSM hoặc counter của MAC duyệt:

```text
neuron_index
input_index
```

Khi xử lý xong một neuron:

```text
output_valid = 1
```

Khi xử lý xong toàn bộ layer:

```text
layer_done = 1
```

---

## 18. ReLU

Logic signed:

```verilog
assign relu_out =
    data_in[DATA_WIDTH-1]
    ? {DATA_WIDTH{1'b0}}
    : data_in;
```

Nếu bit dấu bằng `1`, đầu ra bằng `0`.

---

## 19. Argmax4

So sánh bốn logits:

```text
logit[0]
logit[1]
logit[2]
logit[3]
```

Kết quả:

```text
class_id[1:0]
```

Mapping:

```text
00 = BPSK
01 = QPSK
10 = QAM16
11 = GFSK
```

---

## 20. MLP controller

FSM tổng quát:

```text
IDLE
  │ START
  ▼
CHECK_INPUT
  │
  ▼
DENSE1
  │
  ▼
RELU1
  │
  ▼
DENSE2
  │
  ▼
RELU2
  │
  ▼
DENSE3
  │
  ▼
ARGMAX
  │
  ▼
DONE
  │ CLEAR_DONE
  └──────────────► IDLE
```

Controller phát các tín hiệu:

- `start`.
- `busy`.
- `done`.
- `layer_start`.
- `layer_done`.
- Chỉ số đọc/ghi buffer.
- State debug.

---

# PHẦN IV — AXI4-LITE

## 21. Register map

### Base address của phiên bản Custom IP cũ

```text
0x43C00000
```

### Range

```text
64 KiB
```

> Một số biến thể project sau này dùng `0x40000000`. Không trộn hai bản đồ địa chỉ. README này mô tả **phiên bản Custom IP cũ** sử dụng `0x43C00000`.

| Offset | Tên | Quyền | Nội dung |
|---:|---|---|---|
| `0x00` | `CONTROL` | W | bit 0 START, bit 1 CLEAR_DONE |
| `0x04` | `FEATURE_INDEX` | W | Chỉ số feature 0–15 |
| `0x08` | `FEATURE_DATA` | W | signed 16-bit |
| `0x0C` | `STATUS` | R | BUSY, DONE, INPUT_FULL, FSM state |
| `0x10` | `RESULT` | R | class ID 0–3 |

### CONTROL

```text
bit 0 = START
bit 1 = CLEAR_DONE
```

Các bit điều khiển được dùng theo kiểu xung write-one.

### STATUS

```text
bit 0     = BUSY
bit 1     = DONE
bit 2     = INPUT_FULL
bits 7:4  = controller_state
```

### RESULT

```text
bits 1:0 = class_id
```

---

## 22. Địa chỉ tuyệt đối

Với base `0x43C00000`:

```text
CONTROL       = 0x43C00000
FEATURE_INDEX = 0x43C00004
FEATURE_DATA  = 0x43C00008
STATUS        = 0x43C0000C
RESULT        = 0x43C00010
```

---

# PHẦN V — CẤU TRÚC REPOSITORY

## 23. Cây thư mục đề xuất

```text
pynq-z2-quantized-mlp-amc-4class/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── data/
│   ├── README.md
│   └── .gitkeep
│
├── training/
│   ├── train_amc_mlp.py
│   ├── extract_features.py
│   ├── fuse_batchnorm.py
│   ├── evaluate_model.py
│   └── export_amc_test_15pct.py
│
├── model/
│   ├── amc_mlp_bn_do_16_32_16_4_snr_gt0.pth
│   ├── amc_mlp_bn_do_fused_fp32_snr_gt0.npz
│   ├── amc_feature_scaler_v2_16_snr_gt0.npz
│   └── amc_classification_report_snr_gt0.txt
│
├── quantization/
│   ├── export_fused_npz_to_mem.py
│   ├── amc_quantization_params.vh
│   └── mem/
│       ├── layer1_weights.mem
│       ├── layer1_bias.mem
│       ├── layer2_weights.mem
│       ├── layer2_bias.mem
│       ├── layer3_weights.mem
│       └── layer3_bias.mem
│
├── rtl/
│   ├── AMC_MLP_16_32_16_4_AXI_v1_0.v
│   ├── axi_slave_regs.v
│   ├── amc_mlp_core.v
│   ├── input_buffer.v
│   ├── layer1_weight_rom.v
│   ├── dense1_mac.v
│   ├── relu1.v
│   ├── layer2_weight_rom.v
│   ├── dense2_mac.v
│   ├── relu2.v
│   ├── layer3_weight_rom.v
│   ├── dense3_mac.v
│   ├── argmax4.v
│   └── mlp_controller.v
│
├── constraints/
│   └── pynq_z2_leds_4bits.xdc
│
├── ip_repo/
│   └── AMC_MLP_16_32_16_4_AXI_1.0/
│
├── vivado/
│   ├── add_amc_mlp_ip_complete.tcl
│   ├── build_and_export.tcl
│   └── reports/
│
├── pynq/
│   ├── amc_mlp_16_32_16_4.bit
│   ├── amc_mlp_16_32_16_4.hwh
│   ├── amc_mlp_16_32_16_4.bin
│   └── test_amc_mlp.ipynb
│
└── tests/
    ├── test_fixed_point_model.py
    ├── test_mem_files.py
    └── test_axi_registers.py
```

Không nên commit file dataset RadioML gốc nếu dung lượng lớn hoặc điều khoản phân phối không cho phép. Có thể cung cấp hướng dẫn tải và checksum riêng.

---

# PHẦN VI — HUẤN LUYỆN VÀ XUẤT TRỌNG SỐ

## 24. Môi trường Python đề xuất

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Cài thư viện:

```bash
pip install numpy scipy scikit-learn torch matplotlib jupyter
```

Nên lưu phiên bản:

```bash
pip freeze > requirements.txt
```

---

## 25. Huấn luyện

Quy trình:

```text
1. Đọc RML2016.10a_dict.pkl
2. Lọc BPSK, QPSK, QAM16, GFSK
3. Lọc SNR > 0 dB
4. Trích xuất 16 đặc trưng
5. Chia train/validation/test
6. Fit StandardScaler trên train
7. Transform cả ba tập
8. Huấn luyện MLP
9. Đánh giá
10. Fuse BatchNorm
11. Lưu fused FP32 NPZ
```

File mô hình huấn luyện:

```text
amc_mlp_bn_do_16_32_16_4_snr_gt0.pth
```

File fused cho FPGA:

```text
amc_mlp_bn_do_fused_fp32_snr_gt0.npz
```

FPGA không đọc trực tiếp `.pth` hoặc `.npz`.

---

## 26. Chuyển NPZ sang MEM

Chạy:

```bash
python export_fused_npz_to_mem.py
```

Đầu ra:

```text
layer1_weights.mem
layer1_bias.mem
layer2_weights.mem
layer2_bias.mem
layer3_weights.mem
layer3_bias.mem
```

Kiểm tra:

```text
Layer 1 weights: 512 dòng × 4 hex
Layer 1 bias   :  32 dòng × 8 hex
Layer 2 weights: 512 dòng × 4 hex
Layer 2 bias   :  16 dòng × 8 hex
Layer 3 weights:  64 dòng × 4 hex
Layer 3 bias   :   4 dòng × 8 hex
```

Cập nhật `amc_quantization_params.vh` bằng các giá trị fixed-point đúng của lần export.

---

# PHẦN VII — TẠO VIVADO PROJECT

## 27. Tạo project

Mở Vivado 2023.1:

```text
Create Project
```

Thiết lập:

```text
Project name : AMC_MLP_16_32_16_4_FPGA
Project type : RTL Project
Part         : xc7z020clg400-1
Board        : PYNQ-Z2 nếu đã cài board files
```

Thư mục khuyến nghị:

```text
C:/VIVADO/AMC_MLP_16_32_16_4_FPGA
```

Không tham chiếu source từ:

```text
Downloads
Desktop
OneDrive
```

Nên đặt toàn bộ source trong cấu trúc project ổn định.

---

## 28. Thêm RTL

```text
Add Sources
→ Add or Create Design Sources
→ Add Files
```

Thêm 14 file `.v` và:

```text
amc_quantization_params.vh
```

Bật compile order tự động:

```tcl
set_property source_mgmt_mode All [current_project]
update_compile_order -fileset sources_1
```

---

## 29. Thêm Memory Initialization Files

```text
Add Sources
→ Add or Create Design Sources
→ Add Files
```

Chọn sáu file `.mem`.

Trong Tcl Console:

```tcl
foreach f [get_files -quiet -all *.mem] {
    set_property FILE_TYPE {Memory Initialization Files} $f
    set_property USED_IN_SYNTHESIS true $f
    set_property USED_IN_SIMULATION true $f
}
```

Kiểm tra:

```tcl
foreach f [get_files -quiet -all *.mem] {
    puts "[file tail $f] : [get_property USED_IN_SYNTHESIS $f]"
}
```

Mỗi file phải trả về:

```text
1
```

---

## 30. Thêm XDC LED

File:

```text
pynq_z2_leds_4bits.xdc
```

Mapping:

```tcl
set_property PACKAGE_PIN R14 [get_ports {leds_4bits[0]}]
set_property PACKAGE_PIN P14 [get_ports {leds_4bits[1]}]
set_property PACKAGE_PIN N16 [get_ports {leds_4bits[2]}]
set_property PACKAGE_PIN M14 [get_ports {leds_4bits[3]}]

set_property IOSTANDARD LVCMOS33 [get_ports {leds_4bits[*]}]
```

---

# PHẦN VIII — ĐÓNG GÓI CUSTOM IP

## 31. Chuẩn bị trước khi package

Module top của IP:

```text
AMC_MLP_16_32_16_4_AXI_v1_0
```

Các port AXI bắt buộc:

```text
s00_axi_aclk
s00_axi_aresetn
s00_axi_awaddr
s00_axi_awprot
s00_axi_awvalid
s00_axi_awready
s00_axi_wdata
s00_axi_wstrb
s00_axi_wvalid
s00_axi_wready
s00_axi_bresp
s00_axi_bvalid
s00_axi_bready
s00_axi_araddr
s00_axi_arprot
s00_axi_arvalid
s00_axi_arready
s00_axi_rdata
s00_axi_rresp
s00_axi_rvalid
s00_axi_rready
```

Thông số:

```text
AXI data width    : 32
AXI address width : 6
```

---

## 32. Package IP bằng Vivado

Trong Vivado:

```text
Tools
→ Create and Package New IP
```

Chọn một trong hai cách phù hợp:

```text
Package your current project
```

hoặc:

```text
Package a specified directory
```

Thư mục IP repository đề xuất:

```text
<repo>/ip_repo/AMC_MLP_16_32_16_4_AXI_1.0
```

### 32.1 Identification

Đặt:

```text
Vendor  : user.org
Library : user
Name    : AMC_MLP_16_32_16_4_AXI
Version : 1.0
```

VLNV dự kiến:

```text
user.org:user:AMC_MLP_16_32_16_4_AXI:1.0
```

### 32.2 File Groups

Phải có:

- 14 file RTL.
- `amc_quantization_params.vh`.
- Sáu file `.mem`.
- Synthesis files.
- Simulation files nếu cần.

Không package file `.pth` hoặc `.npz` vào IP.

### 32.3 Ports and Interfaces

Xác nhận Vivado nhận:

```text
S00_AXI
```

Loại interface:

```text
AXI4-Lite Slave
```

Associated clock:

```text
s00_axi_aclk
```

Associated reset:

```text
s00_axi_aresetn
```

Reset polarity:

```text
ACTIVE_LOW
```

### 32.4 Address Block

Địa chỉ nội bộ của IP:

```text
0x00 – 0x3F
```

Register thực tế sử dụng đến:

```text
0x10
```

### 32.5 Review and Package

Chạy:

```text
Review and Package
→ Re-Package IP
```

Sau khi package, thư mục IP phải có:

```text
component.xml
hdl/
xgui/
```

và các file nguồn cần thiết.

---

## 33. Thêm IP repository vào project

```text
Tools
→ Settings
→ IP
→ Repository
→ Add
```

Chọn:

```text
<repo>/ip_repo
```

Sau đó:

```text
Rescan Repositories
```

Hoặc Tcl:

```tcl
set_property ip_repo_paths {
    C:/PATH_TO_REPOSITORY/ip_repo
} [current_project]

update_ip_catalog -rebuild
```

Kiểm tra:

```tcl
get_ipdefs -all *AMC_MLP_16_32_16_4_AXI*
```

Phải thấy VLNV của Custom IP.

---

# PHẦN IX — BLOCK DESIGN SAU KHI CREATE IP

## 34. Các block chính

```text
processing_system7_0
rst_ps7_0_100M
smartconnect_0
axi_gpio_0
AMC_MLP_16_32_16_4_AXI_0
```

Sơ đồ:

```text
                       ┌─────────────────────────┐
                       │ processing_system7_0   │
                       │                         │
                       │ M_AXI_GP0               │
                       │ FCLK_CLK0               │
                       │ FCLK_RESET0_N           │
                       └──────┬────────┬─────────┘
                              │        │
                         AXI  │        │ Clock/Reset
                              │        │
                              ▼        ▼
                     ┌──────────────────────┐
                     │ smartconnect_0       │
                     │ S00_AXI              │
                     │ M00_AXI      M01_AXI │
                     └─────┬──────────┬─────┘
                           │          │
                           ▼          ▼
                ┌──────────────┐  ┌────────────────────────────┐
                │ axi_gpio_0   │  │ AMC_MLP_16_32_16_4_AXI_0  │
                │ LEDs[3:0]    │  │ S00_AXI                    │
                └──────────────┘  └────────────────────────────┘
```

---

## 35. Kết nối AXI

```text
processing_system7_0/M_AXI_GP0
→ smartconnect_0/S00_AXI
```

```text
smartconnect_0/M00_AXI
→ axi_gpio_0/S_AXI
```

```text
smartconnect_0/M01_AXI
→ AMC_MLP_16_32_16_4_AXI_0/S00_AXI
```

SmartConnect:

```text
NUM_SI = 1
NUM_MI = 2
```

---

## 36. Kết nối clock

```text
processing_system7_0/FCLK_CLK0
→ smartconnect_0/aclk
→ axi_gpio_0/s_axi_aclk
→ AMC_MLP_16_32_16_4_AXI_0/s00_axi_aclk
→ rst_ps7_0_100M/slowest_sync_clk
```

Tần số:

```text
100 MHz
```

---

## 37. Kết nối reset

```text
processing_system7_0/FCLK_RESET0_N
→ rst_ps7_0_100M/ext_reset_in
```

```text
rst_ps7_0_100M/interconnect_aresetn
→ smartconnect_0/aresetn
```

```text
rst_ps7_0_100M/peripheral_aresetn
→ axi_gpio_0/s_axi_aresetn
→ AMC_MLP_16_32_16_4_AXI_0/s00_axi_aresetn
```

---

## 38. Address Editor

Phiên bản Custom IP cũ:

```text
AMC MLP : 0x43C00000 / 64K
AXI GPIO: 0x41200000 / 64K
```

Không để hai segment chồng lấn.

Sau khi gán địa chỉ:

```text
Validate Design
```

Kết quả cần:

```text
Validation successful
```

---

## 39. Script thêm Custom IP

File:

```text
vivado/add_amc_mlp_ip_complete.tcl
```

Trước khi chạy, sửa:

```tcl
set IP_REPO_PATH "C:/PATH_TO_REPOSITORY/ip_repo"
```

Script thực hiện:

- Refresh IP Catalog.
- Mở `amc_mlp_system_bd`.
- Tìm Custom IP.
- Tạo cell `AMC_MLP_16_32_16_4_AXI_0`.
- Mở rộng SmartConnect thành hai master interface.
- Nối AXI.
- Nối clock.
- Nối reset.
- Gán `0x43C00000 / 64K`.
- Validate.
- Generate output products.
- Xuất Tcl và báo cáo.

Chạy:

```tcl
source {C:/PATH_TO_REPOSITORY/vivado/add_amc_mlp_ip_complete.tcl}
```

---

## 40. Tạo HDL wrapper

Trong Sources:

```text
Right-click amc_mlp_system_bd
→ Create HDL Wrapper
→ Let Vivado manage wrapper
```

Top bắt buộc:

```text
amc_mlp_system_bd_wrapper
```

Không đặt:

```text
AMC_MLP_16_32_16_4_AXI_v1_0
```

làm top của toàn hệ thống, vì khi đó các port AXI sẽ bị đưa trực tiếp ra chân FPGA và gây lỗi UCIO-1.

---

# PHẦN X — SYNTHESIS, IMPLEMENTATION VÀ BITSTREAM

## 41. Run Synthesis

```text
Flow Navigator
→ Run Synthesis
```

Hoặc Tcl:

```tcl
reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
puts [get_property STATUS [get_runs synth_1]]
```

Kết quả:

```text
synth_design Complete!
```

Kiểm tra log không có:

```text
Cannot open file
Failed to read memory initialization file
File not found
```

---

## 42. Run Implementation

```tcl
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
puts [get_property STATUS [get_runs impl_1]]
```

Kết quả:

```text
write_bitstream Complete!
```

Các cảnh báo DSP pipeline có thể là cảnh báo hiệu năng. Tuy nhiên, lỗi DRC liên quan top module, port chưa gán chân hoặc clock/reset phải được sửa trước khi tạo bitstream.

---

## 43. Xuất BIT và BIN

Mở implementation:

```tcl
open_run impl_1
```

Xuất:

```tcl
write_bitstream -force -bin_file {
    C:/PATH_TO_REPOSITORY/pynq/amc_mlp_16_32_16_4.bit
}
```

Kết quả:

```text
amc_mlp_16_32_16_4.bit
amc_mlp_16_32_16_4.bin
```

---

## 44. Xuất HWH

File HWH thường được sinh từ Block Design output products trong thư mục `hw_handoff`.

Sao chép và đổi tên thành:

```text
amc_mlp_16_32_16_4.hwh
```

Ba file cuối:

```text
amc_mlp_16_32_16_4.bit
amc_mlp_16_32_16_4.hwh
amc_mlp_16_32_16_4.bin
```

PYNQ Overlay sử dụng chủ yếu:

```text
.bit + .hwh
```

Hai file phải có cùng tên gốc và nằm cạnh nhau.

---

# PHẦN XI — CHẠY TRÊN PYNQ-Z2

## 45. Tải file lên PYNQ

Chép vào cùng thư mục Notebook:

```text
amc_mlp_16_32_16_4.bit
amc_mlp_16_32_16_4.hwh
amc_feature_scaler_v2_16_snr_gt0.npz
amc_test_15pct_iq_snr_gt0.npz
```

`.bin` là file tùy chọn đối với luồng `Overlay()` thông thường.

---

## 46. Load Overlay

```python
from pynq import Overlay

overlay = Overlay("amc_mlp_16_32_16_4.bit")
print(overlay.ip_dict.keys())
```

Với HWH hợp lệ, PYNQ có thể đọc metadata và register map của thiết kế.

---

## 47. Truy cập bằng MMIO

```python
from pynq import MMIO

MLP_BASE = 0x43C00000
MLP_RANGE = 0x10000

mmio = MMIO(MLP_BASE, MLP_RANGE)
```

Register offset:

```python
CONTROL       = 0x00
FEATURE_INDEX = 0x04
FEATURE_DATA  = 0x08
STATUS        = 0x0C
RESULT        = 0x10
```

---

## 48. Ghi một vector đặc trưng

```python
import numpy as np
import time

def write_feature_vector(mmio, features_q):
    features_q = np.asarray(features_q, dtype=np.int16)

    if features_q.shape != (16,):
        raise ValueError("features_q phải có đúng shape (16,)")

    # Clear DONE cũ.
    mmio.write(CONTROL, 0x2)

    for index, value in enumerate(features_q):
        mmio.write(FEATURE_INDEX, index)

        # Ghi pattern 16-bit bù hai vào bus 32-bit.
        value_u16 = int(np.uint16(value))
        mmio.write(FEATURE_DATA, value_u16)
```

---

## 49. Bắt đầu inference

```python
def run_inference(mmio, timeout_s=1.0):
    mmio.write(CONTROL, 0x1)

    start_time = time.time()

    while True:
        status = mmio.read(STATUS)

        busy = (status >> 0) & 0x1
        done = (status >> 1) & 0x1

        if done:
            class_id = mmio.read(RESULT) & 0x3
            return class_id

        if time.time() - start_time > timeout_s:
            raise TimeoutError(
                f"FPGA timeout, STATUS=0x{status:08X}, BUSY={busy}"
            )
```

Sử dụng:

```python
write_feature_vector(mmio, features_q)
class_id = run_inference(mmio)

class_names = ["BPSK", "QPSK", "QAM16", "GFSK"]

print("Class ID:", class_id)
print("Prediction:", class_names[class_id])
```

Kết quả ví dụ:

```text
Class ID: 0
Prediction: BPSK
```

---

## 50. Điều khiển LED

AXI GPIO base:

```text
0x41200000
```

Ví dụ:

```python
GPIO_BASE = 0x41200000
GPIO_RANGE = 0x10000

gpio = MMIO(GPIO_BASE, GPIO_RANGE)

GPIO_DATA = 0x00
GPIO_TRI  = 0x04

gpio.write(GPIO_TRI, 0x0)
gpio.write(GPIO_DATA, 1 << class_id)
```

Mapping:

```text
BPSK  → LED0
QPSK  → LED1
QAM16 → LED2
GFSK  → LED3
```

---

# PHẦN XII — KIỂM THỬ VÀ ĐỐI CHIẾU

## 51. Ba mức xác nhận trọng số

### Mức 1 — File hợp lệ

- Đúng số dòng.
- Đúng độ rộng hex.
- Được thêm vào project.
- `USED_IN_SYNTHESIS = 1`.

### Mức 2 — Synthesis thành công

- Không có lỗi `$readmemh`.
- Không có lỗi file missing.
- ROM được đưa vào netlist.

### Mức 3 — Kết quả phần cứng đúng

So sánh:

```text
PyTorch FP32
Python fixed-point
RTL simulation
PYNQ FPGA
```

Cần đo:

- Tỷ lệ dự đoán giống nhau.
- Accuracy.
- Confusion matrix.
- Sai số logits.
- Số mẫu timeout.
- Số mẫu input saturation.

---

## 52. Test từng lớp

Nên kiểm tra riêng:

```text
Dense 1 output
ReLU 1 output
Dense 2 output
ReLU 2 output
Dense 3 logits
Argmax result
```

Dùng cùng một vector input và so sánh với mô phỏng fixed-point trong Python.

---

# PHẦN XIII — LỖI THƯỜNG GẶP

## 53. Failed to resolve reference

```text
Failed to resolve reference:
AMC_MLP_16_32_16_4_AXI_v1_0
```

Nguyên nhân:

- File RTL bị di chuyển.
- Project giữ đường dẫn cũ.
- Compile order manual.
- Thiếu submodule.
- Tên module trong file không khớp.

Khắc phục:

- Đưa toàn bộ RTL vào thư mục repository.
- Add Sources lại.
- Bật source management tự động.
- Kiểm tra:

```tcl
can_resolve_reference AMC_MLP_16_32_16_4_AXI_v1_0
```

---

## 54. Custom IP không xuất hiện

Kiểm tra:

```tcl
get_ipdefs -all *AMC_MLP_16_32_16_4_AXI*
```

Nếu trống:

- Kiểm tra `component.xml`.
- Kiểm tra `ip_repo_paths`.
- Chạy:

```tcl
update_ip_catalog -rebuild
```

---

## 55. Cannot open `.mem`

Nguyên nhân:

- File chưa được package.
- Tên parameter khác tên file.
- Đường dẫn file cũ.
- Memory file không dùng cho synthesis.

Kiểm tra:

```tcl
puts [get_files -all *.mem]
```

Không dùng đường dẫn tuyệt đối trong `$readmemh()`.

---

## 56. UCIO-1 với hàng chục port AXI

Nguyên nhân:

```text
AMC_MLP_16_32_16_4_AXI_v1_0
```

đang bị đặt làm top của toàn project.

Top đúng:

```text
amc_mlp_system_bd_wrapper
```

Không gán chân vật lý cho các tín hiệu AXI nội bộ.

---

## 57. Overlay không thấy IP

Kiểm tra:

- `.bit` và `.hwh` cùng tên.
- Hai file nằm cạnh nhau.
- HWH được sinh từ đúng Block Design.
- IP có address segment.
- Không dùng HWH của bitstream cũ.

---

## 58. Prediction luôn cùng một lớp

Kiểm tra:

- 16 feature có đúng thứ tự không.
- Đã dùng đúng StandardScaler chưa.
- Input có bị saturate không.
- `INPUT_FRAC_BITS` có khớp không.
- `OUTPUT_SHIFT` có khớp không.
- `.mem` có phải bản mới không.
- Bitstream đã build lại sau khi thay `.mem` chưa.
- START và CLEAR_DONE có đúng thứ tự không.

---

# PHẦN XIV — QUY TẮC TÁI LẬP

## 59. Không được thay đổi tùy ý

Các tên cố định:

```text
AMC_MLP_16_32_16_4_FPGA
amc_mlp_system_bd
amc_mlp_system_bd_wrapper
AMC_MLP_16_32_16_4_AXI_v1_0
```

Class mapping cố định:

```text
0 BPSK
1 QPSK
2 QAM16
3 GFSK
```

Kiến trúc cố định:

```text
16 → 32 → 16 → 4
```

Register map cố định:

```text
0x00 CONTROL
0x04 FEATURE_INDEX
0x08 FEATURE_DATA
0x0C STATUS
0x10 RESULT
```

---

## 60. Trọng số tĩnh

Kiến trúc hiện tại dùng ROM khởi tạo bằng `.mem`.

Do đó:

```text
Thay trọng số
→ tạo lại .mem
→ chạy lại synthesis
→ chạy lại implementation
→ tạo bitstream mới
```

Upload `.npz` lên Jupyter không cập nhật trọng số bên trong FPGA.

Muốn nạp weight khi runtime cần thiết kế lại:

- BRAM writable.
- AXI master/slave để ghi weight.
- Register hoặc DMA cho weight loading.
- Cơ chế xác nhận dữ liệu hoàn chỉnh.

---

# PHẦN XV — ĐƯA DỰ ÁN LÊN GITHUB

## 61. Khởi tạo repository

```bash
git init
git add .
git commit -m "Initial commit: quantized MLP AMC on PYNQ-Z2"
git branch -M main
```

Tạo repository GitHub tên:

```text
pynq-z2-quantized-mlp-amc-4class
```

Sau đó:

```bash
git remote add origin https://github.com/<username>/pynq-z2-quantized-mlp-amc-4class.git
git push -u origin main
```

---

## 62. `.gitignore` đề xuất

```gitignore
# Python
__pycache__/
*.pyc
.venv/
.ipynb_checkpoints/

# Large datasets
data/*.pkl
data/*.npz

# PyTorch checkpoints — bỏ dòng này nếu muốn phát hành model
*.pth

# Vivado generated directories
*.cache/
*.gen/
*.hw/
*.ip_user_files/
*.runs/
*.sim/
*.srcs/
.Xil/

# Vivado temporary files
*.jou
*.log
*.str
*.tmp
*.backup.jou
*.backup.log

# OS/editor
.DS_Store
Thumbs.db
.vscode/
```

Không ignore source cần tái lập:

```text
rtl/
quantization/mem/
constraints/
vivado/*.tcl
ip_repo/
pynq/*.bit
pynq/*.hwh
```

Có thể không commit `.bin` nếu không dùng.

---

# PHẦN XVI — TÀI LIỆU THAM KHẢO CHÍNH THỨC

- [AMD Vivado Design Suite User Guide: Synthesis — UG901](https://docs.amd.com/r/en-US/ug901-vivado-synthesis/Loading-Memory-Contents-With-File-I/O-Tasks)
- [AMD Vivado Design Suite User Guide: Designing IP Subsystems Using IP Integrator — UG994](https://docs.amd.com/r/en-US/ug994-vivado-ip-subsystems/Creating-a-Block-Design)
- [AMD UG994 — Adding RTL Modules to the Block Design](https://docs.amd.com/r/en-US/ug994-vivado-ip-subsystems/Adding-RTL-Modules-to-the-Block-Design)
- [PYNQ — Loading an Overlay](https://pynq.readthedocs.io/en/v3.1/pynq_overlays/loading_an_overlay.html)
- [PYNQ Overlay source documentation](https://pynq.readthedocs.io/en/latest/_modules/pynq/overlay.html)

---

# PHẦN XVII — TRẠNG THÁI DỰ ÁN

| Hạng mục | Trạng thái |
|---|---|
| Chọn bốn loại modulation | ✅ |
| Kiến trúc MLP 16–32–16–4 | ✅ |
| BatchNorm fusion | ✅ |
| 14 module RTL | ✅ |
| AXI4-Lite register map | ✅ |
| Xuất sáu file `.mem` | ✅ |
| Kiểm tra kích thước `.mem` | ✅ |
| Package Custom IP | Cần thực hiện trên bản repository sạch |
| Block Design | Cần tạo hoặc phục hồi từ Tcl |
| Synthesis | Chạy lại sau khi package IP |
| Implementation | Chưa xác nhận bản cuối |
| Bitstream chứa trọng số mới | Chưa xác nhận bản cuối |
| PYNQ test toàn bộ tập test | Chưa hoàn tất |

---

## 63. Kết luận

Dự án trình bày đầy đủ một pipeline TinyML/FPGA:

```text
Tín hiệu vô tuyến
→ xử lý đặc trưng
→ học máy
→ fuse BatchNorm
→ lượng tử fixed-point
→ RTL Verilog
→ Custom AXI4-Lite IP
→ Vivado Block Design
→ bitstream
→ PYNQ/Jupyter
```

Điểm quan trọng nhất để kết quả FPGA khớp mô hình Python là bảo đảm đồng nhất:

```text
Thứ tự 16 đặc trưng
StandardScaler
Class mapping
Q-format
OUTPUT_SHIFT
Thứ tự weight trong ROM
Register map
Bitstream và HWH
```

Khi thay model hoặc trọng số, phải tạo lại các file `.mem` và build lại bitstream.
