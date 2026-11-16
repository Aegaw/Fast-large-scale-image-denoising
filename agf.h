// =============================================================================
//  agf.h -- adaptive guided filter（adapt_guided.m 的 C++ 版）
//  ---------------------------------------------------------------------------
//  与 MATLAB 的差异（有意为之，见 README）：
//    * mask 是硬的：m = (v > thr) ? 1 : 0，v 是 GF(t) 的输出（平方之前）
//    * 中间量不 clamp；只有最终输出 clamp 到 [0,255]
//    * 引导图 0..255 时做 -128 中心化；mask 当引导图（值域 0..1）时不中心化
//    * 逐像素 eps（est_noise 平面），不是标量
// =============================================================================
#pragma once
#include <cstdint>

struct AgfParams {
    // 硬 mask 阈值，作用在 m_k = GF(t_k) 上（值域 [0,1]）：
    //  0.0    -> 任何正值都算结构（膨胀 ~r 像素）
    //  0.5    -> 等价中值（10 张 SIDD 实测比 0.0 高 3.2 dB）<- 默认
    //  0.9999 -> 只有饱和处才算（腐蚀 ~r 像素）
    float maskThr = 0.5f;
    int   nthreads = 6;     // 实测 6 线程即贴内存墙，12 线程无收益
    int   tile = 0;         // 0 = 不分块
    bool  fused = true;     // false -> 走平面式参考实现（对拍用）
    bool  verbose = false;
    // ---- 分块滑窗（tiled streaming）参数 ----
    bool  tiled = false;    // true -> 走 agf2_denoise_tiled（读一次写一次）
    int   tileW = 384;      // tile 宽（8 的倍数；384 + 2*halo 贴近缓存且 halo 开销小）
    int   tileH = 384;      // tile 高（8 的倍数）
    int   halo  = 16;       // 全分辨率 halo（8 的倍数；实测 16 已足够正确，8 会退化）
    // ---- 半分辨率加速（实验）----
    bool  half = false;     // true -> 先 2x2 降采样到半分辨率去噪，再双线性放大回原尺寸
    // ---- 单遍流式（mega-kernel，实验）----
    bool  stream = false;   // true -> 引导图 1× 层流式化（消 g/e1/up1/m2up/t1）
    // 注意：分块版把 sub-image 补到 32 的倍数（复制边缘）来规避 agf.cpp 原语里
    // 对“非 128 倍数宽度”的越界假设。但原语仍存在一处堆越界（写入约在 8×8 块噪声
    // 输出大小处），个别 tile 尺寸（如 128/512/768）会触发 0xC0000374 崩溃；
    // 384/256 在本数据集上稳定。若遇到崩溃，可把 denoise_tile 里 align32i 换成
    // align128i（完全稳健但更慢，~277ms）。
};

// 输入：uint8 RGB 交织，H*W*3，H、W 必须是 128 的倍数
// 输出：float RGB 平面式（3 个 H*W 平面连续），值域 0..255
void agf_denoise(const uint8_t* rgb, float* out3, int W, int H, const AgfParams& prm);

// 分块滑窗版：整图切成 tile，每 tile 读一次/写一次，中间量留缓存。语义与 agf_denoise 一致。
void agf2_denoise_tiled(const uint8_t* rgb, float* out3, int W, int H, const AgfParams& prm);

// 单元测试：同一组输入分别跑融合/平面实现，返回 max|d|（0 表示一致）
float agf_test_gf(const float* guide, const float* p, const float* eps,
                  float* out_fused, float* out_plane,
                  int W, int H, int r1, int r2, float epsScale, int nthr);

// 单独暴露出来做单元测试用
namespace agf_detail {
    void block_noise(const float* img, float* outVar, int W, int H);   // 8x8 块, 频带 {2,3,4,5}^2
    void downsample2(const float* src, float* dst, int W, int H);
    void upsample(const float* src, int sw, int sh, float* dst, int dw, int dh, bool bicubic);
    void upsample_u8(const float* src, int sw, int sh, uint8_t* dst, int dw, int dh, float scale = 1.f);
    void bitonic(const float* src, float* dst, int W, int H);
    void box_mean(const float* src, float* dst, int W, int H, int r);
    // pSub != null 时用 (p - pSub) 当作 p；eps != null 时输出 clip01((|cov| - keps*eps^2)/nrm)
    void cov_box(const float* p, const float* q, float* dst, int W, int H, int r,
                 const float* pSub = nullptr, const float* eps = nullptr,
                 float keps = 0.f, float nrm = 1.f,
                 const uint8_t* pSubU8 = nullptr, const uint8_t* epsU8 = nullptr, float epsScale = 1.f);
}
