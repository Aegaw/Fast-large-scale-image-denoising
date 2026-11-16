// =============================================================================
//  main_agf.cpp -- CLI：单图 / SIDD 批量，PNG 读写走 GDI+，输出 PSNR
// =============================================================================
#include "agf.h"
#include "config.h"
#include "image_io.h"
#include "pipeline.h"

#include <immintrin.h>

#include <algorithm>
using std::min;
using std::max;
#ifdef _DEBUG
#include <crtdbg.h>
#endif

extern int g_nthreads;   // 定义在 agf.cpp，供 --boxbench 设置线程数
#include <windows.h>
#include <gdiplus.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace Gdiplus;   // 仅 main() 里的 GdiplusStartup 需要
using Clock = std::chrono::steady_clock;

static double run_one(const Img& in, Img& out, const AgfParams& prm, double* ms) {
    out.w = in.w; out.h = in.h;
    out.rgb.resize((size_t)in.w * in.h * 3);
    const auto t0 = Clock::now();
    agf_denoise_rgb(in.rgb.data(), out.rgb.data(), in.w, in.h, prm);
    const auto t1 = Clock::now();
    if (ms) *ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return 0.0;
}

int main(int argc, char** argv) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#ifdef _DEBUG
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF);
#endif
    GdiplusStartupInput gsi;
    ULONG_PTR token = 0;
    if (GdiplusStartup(&token, &gsi, nullptr) != Ok) { printf("GDI+ init failed\n"); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);

    std::string in, gt, outPath, siddDir;
    AgfParams prm;
    int nmax = 1000000;
    bool verbose = false;

    // ---------------- 无参数：读 run_config.ini 跑一张 ----------------
    if (argc == 1) {
        const std::string exeDir = dir_of(argv[0]);
        AgfConfig cfg = DefaultConfig(exeDir);
        std::string err;
        // 依次找：当前目录 -> exe 目录 -> exe 的上级目录（工程根，ini 放在这里）
        const std::string cands[3] = { "run_config.ini",
                                       exeDir + "\\run_config.ini",
                                       exeDir + "\\..\\run_config.ini" };
        std::string used;
        for (int k = 0; k < 3; ++k)
            if (LoadConfigFromFile(cands[k], cfg, &err)) { used = cands[k]; break; }
        if (!used.empty()) printf("config: %s\n", used.c_str());
        else {
            printf("config: 内置默认（未找到 run_config.ini）\n");
            printf("        reason: %s\n", err.c_str());
        }

        printf("  input  : %s%s\n", cfg.inputPath.c_str(),
               is_directory(cfg.inputPath) ? "  [目录: 批量]" : "");
        printf("  output : %s\n", cfg.outputPath.empty() ? "(不写盘)" : cfg.outputPath.c_str());
        printf("  has_gt : %s%s%s\n", cfg.hasGroundTruth ? "true" : "false",
               (cfg.hasGroundTruth && !cfg.gtPath.empty()) ? "  gt: " : "",
               (cfg.hasGroundTruth && !cfg.gtPath.empty()) ? cfg.gtPath.c_str() : "");
        printf("  params : %s  nt=%d  tile=%dx%d halo=%d  thr=%.2f%s\n",
               cfg.params.half ? "half" : (cfg.params.tiled ? "tiled" : "full"),
               cfg.params.nthreads, cfg.params.tileW, cfg.params.tileH, cfg.params.halo,
               cfg.params.maskThr, cfg.params.verbose ? "  verbose" : "");
        printf("\n");

        const int rc = agf_run_auto(cfg);
        GdiplusShutdown(token);
        return rc;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (k == "--in") in = next();
        else if (k == "--out") outPath = next();
        else if (k == "--gt") gt = next();
        else if (k == "--sidd") siddDir = next();
        else if (k == "--thr") prm.maskThr = (float)atof(next().c_str());
        else if (k == "--n") nmax = atoi(next().c_str());
        else if (k == "--nt") prm.nthreads = atoi(next().c_str());
        else if (k == "--nofused") prm.fused = false;
        else if (k == "--tile") prm.tile = atoi(next().c_str());
        else if (k == "--tiled") prm.tiled = true;
        else if (k == "--half") prm.half = true;
        else if (k == "--stream") prm.stream = true;
        else if (k == "--tw") prm.tileW = atoi(next().c_str());
        else if (k == "--th") prm.tileH = atoi(next().c_str());
        else if (k == "--halo") prm.halo = atoi(next().c_str());
        else if (k == "--verbose") { prm.verbose = true; verbose = true; }
        else if (k == "--difftiled" || k == "--boxbench" || k == "--selfcheck"
                 || k == "--unit" || k == "--gftest") { /* 模式开关，无参数 */ }
        else { printf("unknown arg: %s\n", k.c_str()); }
    }
    // ---------------- 分块滑窗 vs 全图 对拍 ----------------
    if (argc > 1 && std::string(argv[1]) == "--difftiled") {
        Img a;
        if (!load_png(in, a)) { printf("load failed: %s\n", in.c_str()); GdiplusShutdown(token); return 1; }
        const size_t N = (size_t)a.w * a.h;
        std::vector<float> f1(N * 3), f2(N * 3);
        AgfParams full = prm; full.tiled = false; full.verbose = false;
        AgfParams tld  = prm; tld.tiled  = true;  tld.verbose  = false;
        agf_denoise(a.rgb.data(), f1.data(), a.w, a.h, full);
        agf2_denoise_tiled(a.rgb.data(), f2.data(), a.w, a.h, tld);
        double md = 0, se = 0; size_t cnt = 0; int mx = 0, my = 0, mc = 0;
        for (int y = 0; y < a.h; ++y)
            for (int x = 0; x < a.w; ++x)
                for (int c = 0; c < 3; ++c) {
                    const size_t i = ((size_t)c * N) + (size_t)y * a.w + x;
                    const double d = std::fabs((double)f1[i] - f2[i]);
                    if (d > md) { md = d; mx = x; my = y; mc = c; }
                    se += d * d; ++cnt;
                }
        const double mse = se / (double)cnt;
        const double p = mse <= 1e-12 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
        printf("difftiled: max|d| = %.6g at (%d,%d,c%d)  PSNR(tiled vs full) = %.3f dB  %s\n",
               md, mx, my, mc, p, md < 1e-4 ? "IDENTICAL" : "DIFFERS");
        GdiplusShutdown(token);
        return 0;
    }
    // ---------------- box kernel ALU 吞吐基准（缓存内） ----------------
    if (argc > 1 && std::string(argv[1]) == "--boxbench") {
        using namespace agf_detail;
        int nthr = 1;
        for (int i = 2; i + 1 < argc; ++i) if (std::string(argv[i]) == "--nt") nthr = atoi(argv[i + 1]);
        g_nthreads = nthr;
        const int W = 512, H = 512, r = 5, iters = 120;
        const size_t N = (size_t)W * H;
        std::vector<float> p(N), q(N), d(N);
        for (size_t i = 0; i < N; ++i) { p[i] = (float)(i % 251); q[i] = (float)((i * 7) % 241); }
        cov_box(p.data(), q.data(), d.data(), W, H, r);
        const auto t0 = Clock::now();
        for (int k = 0; k < iters; ++k) cov_box(p.data(), q.data(), d.data(), W, H, r);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        const double bpx = 3.0 * (double)N * iters;
        printf("cov_box %dx%d r=%d x%d @%dT: %.1f ms, %.2f Gbox-px/s (%.3f box-px/cycle @4.4GHz)\n",
               W, H, r, iters, nthr, ms, bpx / (ms * 1e6), bpx / (ms * 1e6) / 4.4);
        return 0;
    }
    // ---------------- 语义自检（干净图 / 常量 / 强噪声 / 边缘位置） ----------------
    if (argc > 1 && std::string(argv[1]) == "--selfcheck") {
        const int W = 512, H = 384;
        const size_t N = (size_t)W * H;
        auto run = [&](const std::vector<uint8_t>& in, std::vector<uint8_t>& out, double* ms) {
            std::vector<float> f(N * 3);
            AgfParams ap; ap.maskThr = 0.5f; ap.nthreads = 6;
            const auto t0 = Clock::now();
            agf_denoise(in.data(), f.data(), W, H, ap);
            if (ms) *ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            out.assign(in.size(), 0);
            for (size_t i = 0; i < N; ++i)
                for (int c = 0; c < 3; ++c) {
                    float v = f[c * N + i] + 0.5f;
                    v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
                    out[i * 3 + c] = (uint8_t)v;
                }
        };
        auto psnr = [](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
            double se = 0;
            for (size_t i = 0; i < a.size(); ++i) { const double d = (double)a[i] - b[i]; se += d * d; }
            const double mse = se / (double)a.size();
            return mse <= 1e-12 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
        };
        // 合成干净图：平滑渐变 + 矩形块 + 一条 x=256 的竖直硬边
        std::vector<uint8_t> clean(N * 3);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < 3; ++c) {
                    double v = 40 + 120.0 * x / (W - 1) + 30.0 * std::sin(6.0 * y / H) + 20.0 * c;
                    if (x >= 256) v += 70;
                    if (y > 100 && y < 200 && x > 60 && x < 180) v -= 50;
                    clean[((size_t)y * W + x) * 3 + c] = (uint8_t)std::min(255.0, std::max(0.0, v));
                }
        auto best_shift = [&](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
            double best = 1e30; int bx = 0, by = 0;
            for (int dy = -3; dy <= 3; ++dy)
                for (int dx = -3; dx <= 3; ++dx) {
                    double se = 0; size_t cnt = 0;
                    for (int y = 8; y < H - 8; y += 2)
                        for (int x = 8; x < W - 8; x += 2)
                            for (int c = 0; c < 3; ++c) {
                                const double d = (double)b[((size_t)y * W + x) * 3 + c] -
                                                 a[((size_t)(y + dy) * W + (x + dx)) * 3 + c];
                                se += d * d; ++cnt;
                            }
                    if (se < best) { best = se; bx = dx; by = dy; }
                }
            return std::make_pair(bx, by);
        };
        int fails = 0;
        std::vector<uint8_t> out;
        double ms = 0;

        // A) 干净图输入 -> 输出应 ≈ 输入
        run(clean, out, &ms);
        { bool ok = psnr(out, clean) > 45.0;
          printf("  [A] clean-in  : PSNR(out,clean)=%6.2f dB  shift=(%d,%d)  %s\n",
                 psnr(out, clean), best_shift(clean, out).first, best_shift(clean, out).second, ok ? "OK" : "FAIL");
          if (!ok) ++fails; }

        // B) 常量图 -> 输出应保持常量
        {
            std::vector<uint8_t> cst(N * 3, 128);
            run(cst, out, &ms);
            int mn = 999, mx = -1;
            for (size_t i = 0; i < out.size(); ++i) { mn = std::min(mn, (int)out[i]); mx = std::max(mx, (int)out[i]); }
            const bool ok = (mn >= 127 && mx <= 129);
            printf("  [B] constant  : out range [%d,%d] (expect 128)  %s\n", mn, mx, ok ? "OK" : "FAIL");
            if (!ok) ++fails;
        }

        // C) 干净图 + sigma=30 噪声 -> 应明显优于噪声基线，且不能引入偏移
        {
            std::vector<uint8_t> nz(clean);
            uint32_t st = 12345u;
            for (size_t i = 0; i < nz.size(); ++i) {
                st = st * 1664525u + 1013904223u;
                const double g = ((double)((st >> 8) & 0xFFFF) / 65535.0 - 0.5) * 2.0 * 30.0 * 1.7320508;
                nz[i] = (uint8_t)std::min(255.0, std::max(0.0, (double)nz[i] + g));
            }
            run(nz, out, &ms);
            const double pIn = psnr(nz, clean), pOut = psnr(out, clean);
            const auto sh = best_shift(clean, out);
            const bool ok = (pOut > pIn + 3.0) && (sh.first == 0 && sh.second == 0);
            printf("  [C] noise s=30: PSNR in %6.2f -> out %6.2f dB (+%.2f)  shift=(%d,%d)  %s %.0f ms\n",
                   pIn, pOut, pOut - pIn, sh.first, sh.second, ok ? "OK" : "FAIL", ms);
            if (!ok) ++fails;
        }

        // D) 竖直硬边位置：输入与输出的最大横向梯度列必须相同
        {
            std::vector<uint8_t> stp(N * 3);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    const uint8_t v = (x < 300) ? 60 : 190;
                    for (int c = 0; c < 3; ++c) stp[((size_t)y * W + x) * 3 + c] = v;
                }
            run(stp, out, &ms);
            auto edge_col = [&](const std::vector<uint8_t>& im) {
                int bestc = -1; double bestg = 0;
                for (int x = 0; x + 1 < W; ++x) {
                    double s = 0;
                    for (int y = 20; y < H - 20; y += 4) s += std::fabs((double)im[((size_t)y * W + x + 1) * 3] - im[((size_t)y * W + x) * 3]);
                    if (s > bestg) { bestg = s; bestc = x; }
                }
                return bestc;
            };
            const int ei = edge_col(stp), eo = edge_col(out);
            const bool ok = (std::abs(ei - eo) <= 1);
            printf("  [D] step edge : input col %d, output col %d  %s\n", ei, eo, ok ? "OK" : "FAIL");
            if (!ok) ++fails;
        }
        printf("\n%s\n", fails ? "*** SELFCHECK FAILED ***" : "semantic selfcheck passed");
        GdiplusShutdown(token);
        return fails ? 2 : 0;
    }

    // ---------------- 原语回归测试 ----------------
    if (argc > 1 && std::string(argv[1]) == "--unit") {
        using namespace agf_detail;
        int fails = 0;
        // 1) cov_box vs 朴素
        {
            const int W = 96, H = 64, r = 5;
            std::vector<float> p((size_t)W * H), q((size_t)W * H), o((size_t)W * H, 0.f);
            for (size_t i = 0; i < p.size(); ++i) {
                p[i] = (float)((int)((i * 2654435761u) >> 11) % 200);
                q[i] = (float)((int)((i * 40503u) >> 9) % 200);
            }
            cov_box(p.data(), q.data(), o.data(), W, H, r);
            double worst = 0, refmax = 1e-30;
            const double inv = 1.0 / ((2 * r + 1) * (2 * r + 1));
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    double sp = 0, sq = 0, spq = 0;
                    for (int dy = -r; dy <= r; ++dy) {
                        const int yy = std::min(std::max(y + dy, 0), H - 1);
                        for (int dx = -r; dx <= r; ++dx) {
                            const int xx = std::min(std::max(x + dx, 0), W - 1);
                            const double a = p[(size_t)yy * W + xx], b = q[(size_t)yy * W + xx];
                            sp += a; sq += b; spq += a * b;
                        }
                    }
                    const double ref = spq * inv - (sp * inv) * (sq * inv);
                    refmax = std::max(refmax, std::fabs(ref));
                    worst = std::max(worst, std::fabs(ref - o[(size_t)y * W + x]));
                }
            printf("  cov_box  vs naive: max|d| = %.6g (rel %.2e of |max %.1f|)  %s\n",
                   worst, worst / refmax, refmax, worst / refmax < 1e-5 ? "OK" : "FAIL");
            if (worst / refmax >= 1e-5) ++fails;
        }
        // 2) upsample 双线性/双三次 vs 朴素
        for (int bic = 0; bic < 2; ++bic) {
            const int sw = 48, sh = 32, dw = 96, dh = 64;
            std::vector<float> s((size_t)sw * sh), o((size_t)dw * dh, 0.f);
            for (size_t i = 0; i < s.size(); ++i) s[i] = (float)((int)((i * 7919u) >> 7) % 255);
            upsample(s.data(), sw, sh, o.data(), dw, dh, bic != 0);
            auto cw = [](double t) {
                const double at = std::fabs(t);
                if (at <= 1.0) return 1.5 * at * at * at - 2.5 * at * at + 1.0;
                if (at < 2.0) return -0.5 * at * at * at + 2.5 * at * at - 4.0 * at + 2.0;
                return 0.0;
            };
            double worst = 0;
            for (int y = 0; y < dh; ++y)
                for (int x = 0; x < dw; ++x) {
                    const double u = (x + 0.5) * (double)sw / dw - 0.5;
                    const double v = (y + 0.5) * (double)sh / dh - 0.5;
                    const int jx = (int)std::floor(u), jy = (int)std::floor(v);
                    double ref = 0;
                    for (int ky = -1; ky <= 2; ++ky)
                        for (int kx = -1; kx <= 2; ++kx) {
                            const double wx = bic ? cw(u - (jx + kx)) : (kx == 0 ? 1.0 - (u - jx) : (kx == 1 ? (u - jx) : 0.0));
                            const double wy = bic ? cw(v - (jy + ky)) : (ky == 0 ? 1.0 - (v - jy) : (ky == 1 ? (v - jy) : 0.0));
                            if (wx == 0.0 || wy == 0.0) continue;
                            const int sx = std::min(std::max(jx + kx, 0), sw - 1);
                            const int sy = std::min(std::max(jy + ky, 0), sh - 1);
                            ref += wx * wy * s[(size_t)sy * sw + sx];
                        }
                    worst = std::max(worst, std::fabs(ref - o[(size_t)y * dw + x]));
                }
            printf("  upsample %s vs naive: max|d| = %.6f  %s\n", bic ? "bicubic" : "bilinear", worst,
                   worst < 1e-3 ? "OK" : "FAIL");
            if (worst >= 1e-3) ++fails;
        }
        // 3) block_noise vs 暴力 DFT
        {
            const int W = 64, H = 48;
            std::vector<float> s((size_t)W * H), o((size_t)W * H, 0.f);
            for (size_t i = 0; i < s.size(); ++i) s[i] = (float)((int)((i * 2654435761u) >> 9) % 255);
            block_noise(s.data(), o.data(), W, H);
            double worst = 0;
            for (int by = 0; by < H / 8; ++by)
                for (int bx = 0; bx < W / 8; ++bx) {
                    double E = 0;
                    for (int k = 2; k <= 5; ++k)
                        for (int l = 2; l <= 5; ++l) {
                            double re = 0, im = 0;
                            for (int y = 0; y < 8; ++y)
                                for (int x = 0; x < 8; ++x) {
                                    const double a = -2.0 * 3.14159265358979323846 * (k * y + l * x) / 8.0;
                                    const double v = s[(size_t)(by * 8 + y) * W + bx * 8 + x];
                                    re += v * std::cos(a); im += v * std::sin(a);
                                }
                            E += re * re + im * im;
                        }
                    const double ref = std::max(0.01, 0.0017798 * E);
                    worst = std::max(worst, std::fabs(ref - o[(size_t)by * (W / 8) + bx]));
                }
            printf("  block_noise vs brute DFT: max|d| = %.6f  %s\n", worst, worst < 1e-2 ? "OK" : "FAIL");
            if (worst >= 1e-2) ++fails;
        }
        printf("\n%s\n", fails ? "*** SOME CHECKS FAILED ***" : "all primitive checks passed");
        GdiplusShutdown(token);
        return fails ? 2 : 0;
    }

    // ---------------- 内核隔离测试 ----------------
    if (argc > 1 && std::string(argv[1]) == "--gftest") {
        const int W = 256, H = 192;
        std::vector<float> g((size_t)W * H), p((size_t)W * H), e((size_t)W * H);
        std::vector<float> o1((size_t)W * H, -1e9f), o2((size_t)W * H, -1e9f);
        for (size_t i = 0; i < g.size(); ++i) {
            g[i] = (float)((int)((i * 2654435761u) >> 9) % 256);
            p[i] = (float)((int)((i * 40503u) >> 7) % 256);
            e[i] = 1.0f + (float)(i % 17) * 0.5f;
        }
        for (int nt : {1, 2, 3, 4, 6}) for (int r : {3, 5, 9}) {
            const float d = agf_test_gf(g.data(), p.data(), e.data(), o1.data(), o2.data(), W, H, r, r, 1.0f, nt);
            printf("  nt=%d r=%2d  max|fused - plane| = %.6f  %s\n", nt, r, d, d < 5e-3f ? "OK" : "FAIL");
        }
        GdiplusShutdown(token);
        return 0;
    }

    // ---------------- 单图 ----------------
    if (!in.empty()) {
        Img a, b;
        if (!load_png(in, a)) { printf("load failed: %s\n", in.c_str()); return 1; }
        if (a.w % 128 || a.h % 128) { printf("size %dx%d is not a multiple of 128\n", a.w, a.h); return 1; }
        double ms = 0;
        run_one(a, b, prm, &ms);
        if (!outPath.empty() && !save_png(outPath, b.rgb.data(), b.w, b.h)) {
            printf("save failed: %s\n", outPath.c_str()); return 1;
        }
        printf("%s  %dx%d  %.1f ms\n", base_of(in).c_str(), a.w, a.h, ms);
        {
            double si = 0, si2 = 0, so = 0, so2 = 0;
            const size_t n = a.rgb.size();
            for (size_t i = 0; i < n; ++i) {
                si += a.rgb[i]; si2 += (double)a.rgb[i] * a.rgb[i];
                so += b.rgb[i]; so2 += (double)b.rgb[i] * b.rgb[i];
            }
            const double mi = si / n, mo = so / n;
            printf("  in : mean %7.2f  std %6.2f | out: mean %7.2f  std %6.2f\n",
                   mi, std::sqrt(si2 / n - mi * mi), mo, std::sqrt(so2 / n - mo * mo));
            for (int t = 0; t < 5; ++t) {
                const int x = (int)(a.w * (t + 1) / 6), y = (int)(a.h * (t + 2) / 7);
                const size_t i = ((size_t)y * a.w + x) * 3;
                printf("  (%4d,%4d)  in %3d %3d %3d   out %3d %3d %3d\n",
                       x, y, a.rgb[i], a.rgb[i + 1], a.rgb[i + 2], b.rgb[i], b.rgb[i + 1], b.rgb[i + 2]);
            }
            // 行对齐诊断：out 的第 y 行最像 in 的哪一行
            printf("  row-match: ");
            for (int t = 1; t <= 6; ++t) {
                const int y = (int)((long long)a.h * t / 8);
                double best = 1e30; int by = -1;
                for (int yy = 0; yy < a.h; ++yy) {
                    double se = 0;
                    for (int x = 0; x < a.w; x += 97)
                        for (int c = 0; c < 3; ++c) {
                            const double d = (double)b.rgb[((size_t)y * a.w + x) * 3 + c] - a.rgb[((size_t)yy * a.w + x) * 3 + c];
                            se += d * d;
                        }
                    if (se < best) { best = se; by = yy; }
                }
                printf("y%d->%d ", y, by);
            }
            printf("\n");
        }
        if (!gt.empty()) {
            Img q;
            if (load_png(gt, q) && q.w == a.w && q.h == a.h) {
                const size_t n = (size_t)a.w * a.h * 3;
                double sg = 0, sg2 = 0;
                for (size_t i = 0; i < n; ++i) { sg += q.rgb[i]; sg2 += (double)q.rgb[i] * q.rgb[i]; }
                const double mg = sg / n;
                printf("  gt : mean %7.2f  std %6.2f\n", mg, std::sqrt(sg2 / n - mg * mg));
                printf("  PSNR(out, gt)  = %.3f dB\n", psnr(b.rgb.data(), q.rgb.data(), n));
                printf("  PSNR(noisy, gt)= %.3f dB   (基线)\n", psnr(a.rgb.data(), q.rgb.data(), n));
                printf("  PSNR(out, noisy) = %.3f dB\n", psnr(b.rgb.data(), a.rgb.data(), n));
                // 诊断：out 是否只是 in 的平移
                {
                    double best = 1e30; int bx = 0, by = 0;
                    for (int dy = -4; dy <= 4; ++dy)
                        for (int dx = -4; dx <= 4; ++dx) {
                            double se = 0; size_t cnt = 0;
                            for (int y = 8; y < a.h - 8; y += 3)
                                for (int x = 8; x < a.w - 8; x += 3) {
                                    const size_t i = ((size_t)y * a.w + x) * 3;
                                    const size_t j = ((size_t)(y + dy) * a.w + (x + dx)) * 3;
                                    for (int c = 0; c < 3; ++c) { const double d = (double)b.rgb[i + c] - a.rgb[j + c]; se += d * d; }
                                    cnt += 3;
                                }
                            const double mse = se / (double)cnt;
                            if (mse < best) { best = mse; bx = dx; by = dy; }
                        }
                    printf("  best shift(in -> out) = (%d, %d), mse %.2f\n", bx, by, best);
                }
            }
        }
        GdiplusShutdown(token);
        return 0;
    }

    // ---------------- SIDD 批量 ----------------
    if (!siddDir.empty()) {
        const std::string nd = siddDir + "\\noisy", gd = siddDir + "\\gt", od = siddDir + "\\agf_out";
        CreateDirectoryA(od.c_str(), nullptr);
        std::vector<std::string> names;
        WIN32_FIND_DATAA fd;
        HANDLE h = FindFirstFileA((nd + "\\*.png").c_str(), &fd);
        if (h == INVALID_HANDLE_VALUE) { printf("no noisy png in %s\n", nd.c_str()); return 1; }
        do { names.push_back(fd.cFileName); } while (FindNextFileA(h, &fd));
        FindClose(h);
        std::sort(names.begin(), names.end());
        double sum = 0, sumT = 0, sumMs = 0, worst = 99, best = 0;
        int cnt = 0;
        for (const auto& nm : names) {
            if (cnt >= nmax) break;
            std::string gn = nm;                       // sidd_6001.png -> asidd_6001.png
            const size_t p = gn.find("sidd_");
            if (p != std::string::npos) gn.replace(p, 5, "asidd_");
            Img a, b, q;
            if (!load_png(nd + "\\" + nm, a)) { printf("load fail %s\n", nm.c_str()); continue; }
            if ((a.w % 128) || (a.h % 128)) { printf("skip %s (%dx%d not /128)\n", nm.c_str(), a.w, a.h); continue; }
            double ms = 0;
            run_one(a, b, prm, &ms);
            save_png(od + "\\" + nm, b.rgb.data(), b.w, b.h);
            double pv = 99;
            if (file_exists(gd + "\\" + gn) && load_png(gd + "\\" + gn, q) && q.w == a.w && q.h == a.h)
                pv = psnr(b.rgb.data(), q.rgb.data(), (size_t)a.w * a.h * 3);
            sum += pv; sumT += pv; sumMs += ms; ++cnt;
            worst = std::min(worst, pv); best = std::max(best, pv);
            printf("  [%3d] %-16s %dx%d  %7.1f ms  PSNR=%6.3f dB\n", cnt, nm.c_str(), a.w, a.h, ms, pv);
        }
        if (cnt) {
            printf("\n%d images: mean PSNR = %.3f dB  (min %.3f / max %.3f),  mean %.1f ms/image\n",
                   cnt, sumT / cnt, worst, best, sumMs / cnt);
            printf("total %.2f s\n", sumMs / 1000.0);
            if (sumT / cnt < 37.0) printf("*** WARNING: mean PSNR < 37.0 dB ***\n");
        }
        GdiplusShutdown(token);
        return 0;
    }

    printf("usage:\n");
    printf("  main_agf                                # 读 run_config.ini 跑一张（推荐）\n");
    printf("  main_agf --in noisy.png [--out out.png] [--gt gt.png] [--thr 0]\n");
    printf("  main_agf --sidd <dir> [--n 10] [--thr 0] [--verbose]\n");
    GdiplusShutdown(token);
    return 0;
}
