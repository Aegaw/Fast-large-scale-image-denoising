// =============================================================================
//  pipeline.cpp -- 整条去噪流水线（单函数入口）+ 文件级封装
// =============================================================================
#include "pipeline.h"

#include <immintrin.h>
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "image_io.h"

// -----------------------------------------------------------------------------
//  复用的 scratch 缓冲：避免每张图都重新分配几十~上百 MB 的向量
//  （分配本身 + 首次触碰的缺页开销在 11.7MP 图上可达 ~30 ms）
//  thread_local：多个线程各持一份，互不干扰。
// -----------------------------------------------------------------------------
static std::vector<float>& bufF(int slot, size_t n) {
    static thread_local std::vector<float> v[4];
    std::vector<float>& x = v[slot & 3];
    if (x.size() < n) x.resize(n);
    return x;
}
static std::vector<uint8_t>& bufB(int slot, size_t n) {
    static thread_local std::vector<uint8_t> v[4];
    std::vector<uint8_t>& x = v[slot & 3];
    if (x.size() < n) x.resize(n);
    return x;
}
static std::vector<uint16_t>& bufU(int slot, size_t n) {
    static thread_local std::vector<uint16_t> v[2];
    std::vector<uint16_t>& x = v[slot & 1];
    if (x.size() < n) x.resize(n);
    return x;
}

// -----------------------------------------------------------------------------
//  整个去噪过程：RGB 交织 u8 进 -> RGB 交织 u8 出
//  内部按 prm 选择三条路径：
//    prm.half  -> 2x2 降采样到半分辨率去噪，再双线性放大回原尺寸
//    prm.tiled -> 分块滑窗（每 tile 读一次写一次，中间量留 L3）
//    否则       -> 全图平面式
// -----------------------------------------------------------------------------
void agf_denoise_rgb(const uint8_t* rgb, uint8_t* out, int W, int H,
                     const AgfParams& prm) {
    const size_t N = (size_t)W * H;

    if (prm.half) {
        // ---- 半分辨率：2x2 box 降采样（SIMD 垂直 + 标量水平） ----
        const int hw = W / 2, hh = H / 2;
        const size_t NH = (size_t)hw * hh;
        uint8_t* half = bufB(0, NH * 3).data();
        uint16_t* vsum = bufU(0, (size_t)W * 3).data();
        for (int y = 0; y < hh; ++y) {
            const uint8_t* r0 = rgb + (size_t)(2 * y) * W * 3;
            const uint8_t* r1 = r0 + (size_t)W * 3;
            uint8_t* d = half + (size_t)y * hw * 3;
            const int nbytes = W * 3;
            const int n16 = nbytes & ~15;
            for (int i = 0; i < n16; i += 16) {
                __m128i a = _mm_loadu_si128((const __m128i*)(r0 + i));
                __m128i b = _mm_loadu_si128((const __m128i*)(r1 + i));
                _mm256_storeu_si256((__m256i*)(vsum + i),
                                    _mm256_add_epi16(_mm256_cvtepu8_epi16(a), _mm256_cvtepu8_epi16(b)));
            }
            for (int i = n16; i < nbytes; ++i) vsum[i] = (uint16_t)(r0[i] + r1[i]);
            for (int x = 0; x < hw; ++x) {
                const uint16_t* p = vsum + (size_t)x * 6;
                d[x * 3 + 0] = (uint8_t)((p[0] + p[3] + 2) >> 2);
                d[x * 3 + 1] = (uint8_t)((p[1] + p[4] + 2) >> 2);
                d[x * 3 + 2] = (uint8_t)((p[2] + p[5] + 2) >> 2);
            }
        }
        float* fhalf = bufF(0, NH * 3).data();
        if (prm.tiled) agf2_denoise_tiled(half, fhalf, hw, hh, prm);
        else           agf_denoise(half, fhalf, hw, hh, prm);
        // ---- 放大直出 u8 平面，再交织 ----
        uint8_t* u8p = bufB(1, N * 3).data();
        for (int c = 0; c < 3; ++c)
            agf_detail::upsample_u8(fhalf + (size_t)c * NH, hw, hh,
                                    u8p + (size_t)c * N, W, H, 1.f);
        for (size_t i = 0; i < N; ++i)
            for (int c = 0; c < 3; ++c)
                out[i * 3 + c] = u8p[c * N + i];
        return;
    }

    float* f = bufF(1, N * 3).data();
    if (prm.tiled) agf2_denoise_tiled(rgb, f, W, H, prm);
    else           agf_denoise(rgb, f, W, H, prm);
    // 输出是平面式（R/G/B 三个平面），转回交织 RGB
    for (size_t i = 0; i < N; ++i)
        for (int c = 0; c < 3; ++c) {
            float v = f[c * N + i] + 0.5f;
            v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
            out[i * 3 + c] = (uint8_t)v;
        }
}

// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
//  单张核心：读 -> 去噪 -> 写 ->（可选）PSNR
// -----------------------------------------------------------------------------
static DenoiseResult run_core(const std::string& inPath, const std::string& outPath,
                              const std::string& gtPath, bool hasGt,
                              const AgfParams& prm, bool printDiag) {
    DenoiseResult r;
    Img in;
    if (!load_png(inPath, in)) {
        std::printf("load failed: %s\n", inPath.c_str());
        return r;
    }
    if ((in.w % 128) || (in.h % 128)) {
        std::printf("skip %s: size %dx%d is not a multiple of 128\n",
                    base_of(inPath).c_str(), in.w, in.h);
        return r;
    }
    r.w = in.w;
    r.h = in.h;

    Img out;
    out.w = in.w;
    out.h = in.h;
    out.rgb.resize((size_t)in.w * in.h * 3);

    const auto t0 = std::chrono::steady_clock::now();
    agf_denoise_rgb(in.rgb.data(), out.rgb.data(), in.w, in.h, prm);
    const auto t1 = std::chrono::steady_clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.ok = true;

    if (!outPath.empty()) {
        const std::string od = dir_of(outPath);
        if (!od.empty() && od != ".") CreateDirectoryA(od.c_str(), nullptr);   // 目录不存在则创建
        if (!save_png(outPath, out.rgb.data(), out.w, out.h)) {
            std::printf("save failed: %s\n", outPath.c_str());
            r.ok = false;
        }
    }

    if (hasGt && !gtPath.empty()) {
        Img gt;
        if (load_png(gtPath, gt) && gt.w == in.w && gt.h == in.h) {
            const size_t n = (size_t)in.w * in.h * 3;
            r.hasGt = true;
            r.psnrOutGt = psnr(out.rgb.data(), gt.rgb.data(), n);
            r.psnrNoisyGt = psnr(in.rgb.data(), gt.rgb.data(), n);
            r.psnrOutNoisy = psnr(out.rgb.data(), in.rgb.data(), n);
        } else {
            std::printf("warning: cannot read GT or size mismatch, PSNR skipped: %s\n", gtPath.c_str());
        }
    }
    if (printDiag) {
        double si = 0, si2 = 0, so = 0, so2 = 0;
        const size_t n = in.rgb.size();
        for (size_t i = 0; i < n; ++i) {
            si += in.rgb[i]; si2 += (double)in.rgb[i] * in.rgb[i];
            so += out.rgb[i]; so2 += (double)out.rgb[i] * out.rgb[i];
        }
        const double mi = si / n, mo = so / n;
        std::printf("  in : mean %7.2f  std %6.2f | out: mean %7.2f  std %6.2f\n",
                    mi, std::sqrt(si2 / n - mi * mi), mo, std::sqrt(so2 / n - mo * mo));
    }
    return r;
}

// -----------------------------------------------------------------------------
//  单张：读 -> 去噪 -> 写 -> PSNR
// -----------------------------------------------------------------------------
DenoiseResult agf_run(const AgfConfig& cfg) {
    return run_core(cfg.inputPath, cfg.outputPath, cfg.gtPath, cfg.hasGroundTruth,
                    cfg.params, true);
}

// -----------------------------------------------------------------------------
//  批量：inputPath 是目录。noisy 与 gt 各自按文件名排序后逐张一一对应。
// -----------------------------------------------------------------------------
BatchResult agf_run_batch(const AgfConfig& cfg) {
    BatchResult b;
    const std::vector<std::string> noisy = list_png_sorted(cfg.inputPath);
    if (noisy.empty()) {
        std::printf("no *.png found in %s\n", cfg.inputPath.c_str());
        return b;
    }
    std::vector<std::string> gts;
    if (cfg.hasGroundTruth && !cfg.gtPath.empty()) gts = list_png_sorted(cfg.gtPath);

    if (!cfg.outputPath.empty()) CreateDirectoryA(cfg.outputPath.c_str(), nullptr);

    const size_t nPair = noisy.size();
    if (b.hasGt = !gts.empty(); b.hasGt && gts.size() != nPair)
        std::printf("note: noisy=%zu, gt=%zu, pairing by sorted index up to min\n",
                    nPair, gts.size());
    const size_t nRun = b.hasGt ? (std::min)(nPair, gts.size()) : nPair;

    double sumP = 0, sumMs = 0;
    b.minPsnr = 99.0;
    b.maxPsnr = 0.0;
    for (size_t i = 0; i < nRun; ++i) {
        const std::string outPath = cfg.outputPath.empty()
            ? std::string()
            : cfg.outputPath + "\\" + base_of(noisy[i]);
        const DenoiseResult r = run_core(noisy[i], outPath,
                                         b.hasGt ? gts[i] : std::string(), b.hasGt,
                                         cfg.params, false);
        if (!r.ok) { ++b.skipped; continue; }
        ++b.count;
        sumMs += r.ms;
        if (r.hasGt) {
            sumP += r.psnrOutGt;
            b.minPsnr = (std::min)(b.minPsnr, r.psnrOutGt);
            b.maxPsnr = (std::max)(b.maxPsnr, r.psnrOutGt);
        }
        std::printf("  [%3d] %-20s %dx%d  %7.1f ms", b.count,
                    base_of(noisy[i]).c_str(), r.w, r.h, r.ms);
        if (r.hasGt) std::printf("  PSNR=%6.3f dB", r.psnrOutGt);
        std::printf("\n");
    }
    if (b.count) {
        b.meanMs = sumMs / b.count;
        b.totalS = sumMs / 1000.0;
        if (b.hasGt) b.meanPsnr = sumP / b.count;
        std::printf("\n%d images", b.count);
        if (b.hasGt) std::printf(": mean PSNR = %.3f dB  (min %.3f / max %.3f)", b.meanPsnr, b.minPsnr, b.maxPsnr);
        std::printf(",  mean %.1f ms/image\n", b.meanMs);
        std::printf("total %.2f s\n", b.totalS);
        if (b.skipped) std::printf("skipped %d\n", b.skipped);
    }
    return b;
}

// -----------------------------------------------------------------------------
//  自动分派：input 是文件 -> 单张；是目录 -> 批量
// -----------------------------------------------------------------------------
int agf_run_auto(const AgfConfig& cfg) {
    if (is_directory(cfg.inputPath)) {
        const BatchResult b = agf_run_batch(cfg);
        return (b.count > 0) ? 0 : 1;
    }
    const DenoiseResult r = agf_run(cfg);
    if (!r.ok) return 1;
    if (r.hasGt)
        std::printf("  PSNR(out, gt) = %.3f dB   PSNR(noisy, gt) = %.3f dB   PSNR(out, noisy) = %.3f dB\n",
                    r.psnrOutGt, r.psnrNoisyGt, r.psnrOutNoisy);
    return 0;
}
