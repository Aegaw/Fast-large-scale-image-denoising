# 自适应引导滤波去噪


单张 3968×2944 图像稳定速度约145 ms


---

## 1. 用法（cmd 进入所在路径，直接运行）

```cmd
cd 所在路径
bin\agf.exe
```

不需要任何参数、不需要改路径。程序会：

1. 自动找到本目录的 `run_config.ini`；
2. 读 **`noisy_images\`** 里的噪声图（按文件名升序排序）；
3. 与 **`truth_images\`** 里的真值图（同样按文件名升序排序）**逐张一一对应**；
4. 去噪结果写到 **`out_images\`**（文件名沿用噪声图文件名）；
5. 每张打印耗时与 PSNR，最后打印均值。

---

## 2. 目录结构

```
\
├─ bin\agf.exe                已编译好的程序
├─ noisy_images\              噪声图（按文件名排序）
├─ truth_images\              真值图（按文件名排序，与上面逐张对应）
├─ out_images\                去噪结果（运行后生成）
├─ run_config.ini             ★ 入口配置：读/写路径、是否有真值、去噪参数
├─ agf.cpp / agf.h            算法原语 + agf_denoise（全图）+ agf2_denoise_tiled（分块）
├─ pipeline.cpp / pipeline.h  ★ 对外唯一入口 agf_denoise_rgb() + agf_run/agf_run_batch
├─ config.cpp / config.h      配置结构体 + 默认值 + ini 解析
├─ image_io.cpp / image_io.h  PNG 读写（GDI+）+ PSNR
├─ main_agf.cpp               命令行（ini 模式 / --in / --sidd / 自检）
├─ build.ps1                  MSVC 构建脚本
└─ README.md
```

**对应规则**：`noisy_images` 与 `truth_images` 各自按**文件名升序**排序后，第 N 个噪声图配第 N 个真值图。
只要两个文件夹里文件数量相同、排序后语义一致即可。

---

## 3. 构建

```powershell
cd 所在路径
.\build.ps1 agf          # 需要 VS2022 的 vcvars64.bat
```

编译选项：`/O2 /arch:AVX2 /fp:precise /EHsc /std:c++17`。

---

## 4. 修改配置（不用重新编译）

编辑 `run_config.ini`：

```ini
# 读取路径：目录 -> 批量；文件 -> 单张
input  = noisy_images

# 写入路径：目录（留空 = 只算不写盘）
output = out_images

# 是否存在真值图像
has_gt = true
gt     = truth_images

# 去噪参数
tiled = true
tw    = 384
th    = 384
halo  = 16
nt    = 8
thr   = 0.5
```

> 路径规则：相对路径相对 **ini 所在目录** 解析，最后规范化为绝对路径；
> 也可以直接写绝对路径，例如 `input = D:\data\my_noisy.png`。
> `run_config.ini` 的查找顺序：当前目录 → exe 目录 → exe 的上级目录。

---

## 5. 代码里怎么用（整个去噪 = 一个函数）

```cpp
#include "pipeline.h"

// 输入/输出都是 H*W*3 交织 u8；W、H 需为 128 的倍数
std::vector<uint8_t> out((size_t)W * H * 3);
agf_denoise_rgb(in_rgb.data(), out.data(), W, H);   // 默认参数 = 当前配置

// 想改参数就传一个 AgfParams：
AgfParams prm;
prm.tiled = true;  prm.tileW = prm.tileH = 384;  prm.halo = 16;  prm.nthreads = 8;
agf_denoise_rgb(in_rgb.data(), out.data(), W, H, prm);
```

`AgfParams` 全部可调项（`agf.h`，默认值即当前配置）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `maskThr` | 0.5 | 硬 mask 阈值（0.5 ≈ 中值，比 0 高约 3 dB） |
| `nthreads` | 6 | 线程数；实测 8 线程打满内存带宽，再多因 SMT 争抢反而慢 |
| `tiled` | false | 分块滑窗（推荐开）；`false` = 全图平面式 |
| `tileW` / `tileH` | 384 | tile 尺寸（8 的倍数） |
| `halo` | 16 | 分块 halo（8 的倍数） |
| `half` | false | 半分辨率快速路径（2×2 降采样→去噪→双线性放大） |
| `stream` | false | 引导图 1× 层流式（实验性，更慢，仅对照） |
| `verbose` | false | 打印各相位耗时 |

文件/批量封装：

```cpp
#include "config.h"
#include "pipeline.h"

AgfConfig cfg = DefaultConfig(".");          // 或 LoadConfigFromFile("run_config.ini", cfg)
int rc = agf_run_auto(cfg);                  // 自动判断 input 是文件还是目录
```

---

## 6. 命令行（覆盖 ini，用于单张/实验）

```powershell
# 单张
bin\agf.exe --in noisy.png --out out.png --gt gt.png --tiled --tw 384 --th 384 --halo 16 --nt 8

# 半分辨率快速路径
bin\agf.exe --in noisy.png --gt gt.png --half --tiled --tw 256 --th 256 --nt 8

# 指定其它数据目录做批量（目录内按名称排序配对）
bin\agf.exe --sidd <dir> --n 160 --tiled --tw 384 --th 384 --halo 16 --nt 8

# 自检
bin\agf.exe --unit            # 原语单元测试（cov_box / upsample / block_noise ...）
bin\agf.exe --selfcheck       # 语义自检
bin\agf.exe --gftest          # 融合内核 vs 平面内核
bin\agf.exe --difftiled --in x.png    # 分块 vs 全图对拍
```

---

## 7. 说明

- `W`、`H` 必须是 **128 的倍数**（算法按 8×8 块与 2× 金字塔对齐）。
- 批量时若两个文件夹张数不同，按较小值配对并给出提示。
- `stream` 路径为实验性实现（单遍流式引导图 1× 层），正确但比平面版慢，默认关闭。
- 本项目在 DeepSeek Harness 辅助下完成。
