//  agf.cpp -- adapt_guided.m 的 C++ 实现（v1: 正确优先，单线程，平面式）
//  所有平面都是 float32，行优先连续；H、W 必须是 128 的倍数。
// =============================================================================
#include "agf.h"

#include <immintrin.h>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <limits>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <malloc.h>
#include <chrono>

#ifdef _DEBUG
#include <crtdbg.h>
#endif

namespace {

// ---------------------------------------------------------------- SIMD 辅助
inline __m256 prefix8(__m256 v) {
    __m256i vi = _mm256_castps_si256(v);
    v = _mm256_add_ps(v, _mm256_castsi256_ps(_mm256_slli_si256(vi, 4)));
    vi = _mm256_castps_si256(v);
    v = _mm256_add_ps(v, _mm256_castsi256_ps(_mm256_slli_si256(vi, 8)));
    __m256 lo = _mm256_permute2f128_ps(v, v, 0x00);
    __m256 c4 = _mm256_shuffle_ps(lo, lo, _MM_SHUFFLE(3, 3, 3, 3));
    c4 = _mm256_blend_ps(_mm256_setzero_ps(), c4, 0xF0);
    return _mm256_add_ps(v, c4);
}
inline float last_lane(__m256 v) {
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    return _mm_cvtss_f32(_mm_shuffle_ps(hi, hi, 0xFF));
}
inline int align8i(int v) { return (v + 7) & ~7; }
inline int align128i(int v) { return ((v + 127) / 128) * 128; }
inline int align32i(int v) { return ((v + 31) / 32) * 32; }
inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// 线程私有 scratch（各用途必须用不同槽，否则嵌套调用会互相踩）
float* scratch(int which, size_t n) {
    static thread_local std::vector<float*> buf;
    static thread_local std::vector<size_t> cap;
    if ((size_t)which >= buf.size()) { buf.resize((size_t)which + 1, nullptr); cap.resize((size_t)which + 1, 0); }
    if (cap[(size_t)which] < n) {
        std::free(buf[(size_t)which]);
        buf[(size_t)which] = (float*)std::malloc(n * sizeof(float));
        cap[(size_t)which] = n;
    }
    return buf[(size_t)which];
}

// 平面池：按需扩容，不清零
struct Pool {
    std::vector<float*> buf;
    std::vector<size_t> cap;
    float* get(int i, size_t n) {
        if ((size_t)i >= buf.size()) { buf.resize((size_t)i + 1, nullptr); cap.resize((size_t)i + 1, 0); }
        if (cap[(size_t)i] < n) {
            std::free(buf[(size_t)i]);
            buf[(size_t)i] = (float*)std::malloc(n * sizeof(float));
            cap[(size_t)i] = n;
        }
        return buf[(size_t)i];
    }
    ~Pool() { for (float* p : buf) std::free(p); }
};

void mul_n(float* dst, const float* a, const float* b, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; ++i) dst[i] = a[i] * b[i];
}
void blend_n(float* dst, const float* a, const float* b, const uint8_t* mask, size_t n) {
    if (!mask) { if (dst != a) std::memcpy(dst, a, sizeof(float) * n); return; }
    size_t i = 0;
    const __m256i z = _mm256_setzero_si256();
    for (; i + 8 <= n; i += 8) {
        const __m256i mi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(mask + i)));
        const __m256 mf = _mm256_castsi256_ps(_mm256_cmpgt_epi32(mi, z));
        _mm256_storeu_ps(dst + i, _mm256_blendv_ps(_mm256_loadu_ps(a + i),
                                                  _mm256_loadu_ps(b + i), mf));
    }
    for (; i < n; ++i) dst[i] = mask[i] ? b[i] : a[i];
}
inline float clip01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }

} // namespace

// =============================================================================
int g_nthreads = 1;
int g_gf_tile = 0;   // 融合内核的列分块宽度（0 = 不分块）
bool g_stream = false; // true -> 引导图 1× 层用 stream_guide_1x（消 g/e1/up1/m2up/t1）

struct ThreadPool {
    std::vector<std::thread> th;
    std::mutex mu;
    std::condition_variable cv, cvDone;
    int gen = 0, doneCnt = 0, nw = 0;
    const std::function<void(int)>* job = nullptr;
    bool quit = false;

    int threads() const { return nw > 0 ? nw : 1; }

    void ensure(int n) {
        if (nw == n) return;
        stop();
        nw = n;
        for (int i = 1; i < n; ++i)
            th.emplace_back([this, i] {
                _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
                _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
                int seen = 0;
                for (;;) {
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv.wait(lk, [&] { return gen != seen || quit; });
                        if (quit) return;
                        seen = gen;
                    }
                    (*job)(i);
                    {
                        std::lock_guard<std::mutex> lk(mu);
                        ++doneCnt;
                    }
                    cvDone.notify_one();
                }
            });
    }
    void run(int n, const std::function<void(int)>& f) {
        if (n <= 1) { f(0); return; }
        ensure(n);
        {
            std::lock_guard<std::mutex> lk(mu);
            job = &f;
            doneCnt = 0;
            ++gen;
        }
        cv.notify_all();
        f(0);
        {
            std::unique_lock<std::mutex> lk(mu);
            cvDone.wait(lk, [&] { return doneCnt >= n - 1; });
            job = nullptr;
        }
    }
    void stop() {
        if (th.empty()) { nw = 0; return; }
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv.notify_all();
        for (auto& t : th) t.join();
        th.clear();
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = false; gen = 0; doneCnt = 0;
        }
        nw = 0;
    }
    ~ThreadPool() { stop(); }
};

ThreadPool& get_pool(int n) {
    static ThreadPool p;
    p.ensure(n);
    return p;
}

// 每线程工作区（环形缓冲）：支持 NC 通道 + 列分块
struct Fws {
    int W = -1, rmax = 0, NC = 0, tileW = -1;
    int PAD = 0, WP = 0, cD = 0, h1D = 0, h2D = 0, maxCD = 0, s1n = 0, s2n = 0;
    float* cRing = nullptr;
    float* prow = nullptr;   // NC * WP
    float* h1 = nullptr;     // s1n * maxCD * WP
    float* v1 = nullptr;     // s1n * WP
    float* h2 = nullptr;     // s2n * maxCD * WP
    float* v2 = nullptr;     // s2n * WP
    float* ra = nullptr;     // NC * WP
    float* rb = nullptr;     // NC * WP

    static float* A(size_t n) { return (float*)_aligned_malloc(n * sizeof(float), 64); }
    void release() {
        _aligned_free(cRing); _aligned_free(prow); _aligned_free(h1); _aligned_free(v1);
        _aligned_free(h2); _aligned_free(v2); _aligned_free(ra); _aligned_free(rb);
        cRing = prow = h1 = v1 = h2 = v2 = ra = rb = nullptr;
        W = -1; rmax = 0; NC = 0; tileW = -1;
    }
    void ensure(int w, int nc, int tilew, int a, int b) {
        const int need = (a > b ? a : b);
        if (cRing && W == w && NC == nc && tileW == tilew && rmax >= need) {
            cD = a + b + 1; h1D = 2 * a + 1; h2D = 2 * b + 1;
            return;
        }
        release();
        W = w; NC = nc; tileW = tilew;
        rmax = 20; if (need > rmax) rmax = need;    // 一次分配够用，避免每次调用重分配
        PAD = align8i(2 * rmax + 16);
        WP = align8i(tilew + 2 * PAD);
        maxCD = 2 * rmax + 1;
        cD = a + b + 1; h1D = 2 * a + 1; h2D = 2 * b + 1;
        s1n = 2 + 2 * nc; s2n = 2 * nc;
        cRing = A((size_t)maxCD * WP);
        prow  = A((size_t)nc * WP);
        h1    = A((size_t)s1n * maxCD * WP);
        v1    = A((size_t)s1n * WP);
        h2    = A((size_t)s2n * maxCD * WP);
        v2    = A((size_t)s2n * WP);
        ra    = A((size_t)nc * WP);
        rb    = A((size_t)nc * WP);
    }
    ~Fws() { release(); }
};

constexpr int AGF_NSTRIP = 6;
// 条带数只由 H 决定（与线程数无关），且小图不要切太碎
inline int nstrip_for(int H) {
    int n = H / 64;
    if (n < 1) n = 1;
    if (n > AGF_NSTRIP) n = AGF_NSTRIP;
    return n;
}

template <class F> void par_rows(int H, F&& f) {
    ThreadPool& p = get_pool(g_nthreads);
    const int nt = p.threads();
    const int ns = nstrip_for(H);
    p.run(nt, [&](int tid) {
        for (int k = tid; k < ns; k += nt) {
            const int y0 = (int)((long long)H * k / ns);
            const int y1 = (int)((long long)H * (k + 1) / ns);
            if (y1 > y0) f(y0, y1);
        }
    });
}
template <class F> void par_cols(int W, F&& f) {
    ThreadPool& p = get_pool(g_nthreads);
    const int nt = p.threads();
    const int ns = nstrip_for(1 << 20) ;  // 列方向固定用 6 段
    p.run(nt, [&](int tid) {
        for (int k = tid; k < ns; k += nt) {
            const int x0 = (int)((long long)W * k / ns) & ~7;
            int x1 = (int)((long long)W * (k + 1) / ns) & ~7;
            if (k == ns - 1) x1 = W;
            if (x1 > x0) f(x0, x1);
        }
    });
}

// f(stripIdx, a, b)：n 个下标切成固定 6 段，与线程数无关
template <class F> void par_n(size_t n, F&& f) {
    ThreadPool& p = get_pool(g_nthreads);
    const int nt = p.threads();
    const int ns = 6;
    p.run(nt, [&](int tid) {
        for (int k = tid; k < ns; k += nt) {
            const size_t a = n * (size_t)k / (size_t)ns;
            const size_t b = n * (size_t)(k + 1) / (size_t)ns;
            if (b > a) f(k, a, b);
        }
    });
}
namespace agf_detail {

static void box_h_range(const float* src, float* dst, int W, int y0, int y1, int r) {
    const int PAD = align8i(r + 8);
    const int WP  = align8i(W + 2 * PAD);
    float* rb = scratch(0, (size_t)WP);
    const float inv = 1.0f / (float)(2 * r + 1);
    const __m256 vinv = _mm256_set1_ps(inv);
    for (int y = y0; y < y1; ++y) {
        const float* s = src + (size_t)y * W;
        float* d = dst + (size_t)y * W;
        for (int i = 0; i < PAD; ++i) rb[i] = s[0];
        std::memcpy(rb + PAD, s, sizeof(float) * (size_t)W);
        for (int i = PAD + W; i < WP; ++i) rb[i] = s[W - 1];
        float q = 0.f;
        for (int k = -r; k <= r; ++k) q += rb[PAD - 1 + k];
        int x = PAD;
        for (; x + 8 <= PAD + W; x += 8) {
            const __m256 nc = _mm256_loadu_ps(rb + x + r);
            const __m256 oc = _mm256_loadu_ps(rb + x - r - 1);
            const __m256 hv = _mm256_add_ps(_mm256_set1_ps(q), prefix8(_mm256_sub_ps(nc, oc)));
            q = last_lane(hv);
            _mm256_storeu_ps(d + (x - PAD), _mm256_mul_ps(hv, vinv));
        }
        for (; x < PAD + W; ++x) {
            q += rb[x + r] - rb[x - r - 1];
            d[x - PAD] = q * inv;
        }
    }
}

static void box_v_range(const float* src, float* dst, int W, int H, int x0, int x1, int r) {
    float* acc = scratch(1, (size_t)W);
    const float inv = 1.0f / (float)(2 * r + 1);
    const __m256 vinv = _mm256_set1_ps(inv);
    const int xv = x1 & ~7;
    for (int x = x0; x < xv; x += 8) _mm256_storeu_ps(acc + x, _mm256_setzero_ps());
    for (int x = xv; x < x1; ++x) acc[x] = 0.f;
    for (int k = -r; k <= r; ++k) {
        const float* s = src + (size_t)clampi(k, 0, H - 1) * W;
        for (int x = x0; x < xv; x += 8)
            _mm256_storeu_ps(acc + x, _mm256_add_ps(_mm256_loadu_ps(acc + x), _mm256_loadu_ps(s + x)));
        for (int x = xv; x < x1; ++x) acc[x] += s[x];
    }
    for (int y = 0; y < H; ++y) {
        float* d = dst + (size_t)y * W;
        for (int x = x0; x < xv; x += 8)
            _mm256_storeu_ps(d + x, _mm256_mul_ps(_mm256_loadu_ps(acc + x), vinv));
        for (int x = xv; x < x1; ++x) d[x] = acc[x] * inv;
        if (y + 1 < H) {
            const float* na = src + (size_t)clampi(y + r + 1, 0, H - 1) * W;
            const float* oa = src + (size_t)clampi(y - r, 0, H - 1) * W;
            for (int x = x0; x < xv; x += 8)
                _mm256_storeu_ps(acc + x, _mm256_add_ps(_mm256_loadu_ps(acc + x),
                                 _mm256_sub_ps(_mm256_loadu_ps(na + x), _mm256_loadu_ps(oa + x))));
            for (int x = xv; x < x1; ++x) acc[x] += na[x] - oa[x];
        }
    }
}

void box_mean(const float* src, float* dst, int W, int H, int r) {
    float* tmp = scratch(2, (size_t)W * H);
    par_rows(H, [&](int a, int b) { box_h_range(src, tmp, W, a, b, r); });
    par_cols(W, [&](int a, int b) { box_v_range(tmp, dst, W, H, a, b, r); });
}

// cov_box 单遍融合：box(p)、box(q)、box(p*q) 三条流一次扫完（读 p、q 各一遍）
struct Cws {
    int W = -1, r = -1, PAD = 0, WP = 0, depth = 0;
    float* pr = nullptr;
    float* qr = nullptr;
    float* h[3] = { nullptr, nullptr, nullptr };
    float* v[3] = { nullptr, nullptr, nullptr };
    static float* A(size_t n) { return (float*)_aligned_malloc(n * sizeof(float), 64); }
    void release() {
        _aligned_free(pr); _aligned_free(qr);
        for (int i = 0; i < 3; ++i) { _aligned_free(h[i]); _aligned_free(v[i]); }
        pr = qr = nullptr;
        for (int i = 0; i < 3; ++i) h[i] = v[i] = nullptr;
        W = -1;
    }
    void ensure(int w, int rr) {
        if (pr && W == w && r == rr) return;
        release();
        W = w; r = rr;
        PAD = align8i(2 * 20 + 16);   // 56，与 Fws 对齐（cov 的 t1 halo 需要）
        WP = align8i(w + 2 * PAD);
        depth = 2 * rr + 1;
        pr = A((size_t)WP); qr = A((size_t)WP);
        for (int i = 0; i < 3; ++i) { h[i] = A((size_t)depth * WP); v[i] = A((size_t)WP); }
    }
    ~Cws() { release(); }
};
Cws& get_cws(int W, int r) { static thread_local Cws w; w.ensure(W, r); return w; }

inline void cov_step(const float* __restrict a, const float* __restrict b, int x, int r, bool warm,
                     float& q0, float& q1, float& q2,
                     float* __restrict h0, float* __restrict h1, float* __restrict h2,
                     float* __restrict v0, float* __restrict v1, float* __restrict v2) {
    const __m256 na = _mm256_loadu_ps(a + x + r);
    const __m256 oa = _mm256_loadu_ps(a + x - r - 1);
    const __m256 nb = _mm256_loadu_ps(b + x + r);
    const __m256 ob = _mm256_loadu_ps(b + x - r - 1);
    const __m256 h0v = _mm256_add_ps(_mm256_set1_ps(q0), prefix8(_mm256_sub_ps(na, oa)));
    const __m256 h1v = _mm256_add_ps(_mm256_set1_ps(q1), prefix8(_mm256_sub_ps(nb, ob)));
    const __m256 h2v = _mm256_add_ps(_mm256_set1_ps(q2),
        prefix8(_mm256_sub_ps(_mm256_mul_ps(na, nb), _mm256_mul_ps(oa, ob))));
    q0 = last_lane(h0v); q1 = last_lane(h1v); q2 = last_lane(h2v);
    const __m256 o0 = _mm256_load_ps(h0 + x);
    const __m256 o1 = _mm256_load_ps(h1 + x);
    const __m256 o2 = _mm256_load_ps(h2 + x);
    _mm256_store_ps(h0 + x, h0v); _mm256_store_ps(h1 + x, h1v); _mm256_store_ps(h2 + x, h2v);
    __m256 s0 = _mm256_load_ps(v0 + x), s1 = _mm256_load_ps(v1 + x), s2 = _mm256_load_ps(v2 + x);
    if (warm) { s0 = _mm256_sub_ps(s0, o0); s1 = _mm256_sub_ps(s1, o1); s2 = _mm256_sub_ps(s2, o2); }
    _mm256_store_ps(v0 + x, _mm256_add_ps(s0, h0v));
    _mm256_store_ps(v1 + x, _mm256_add_ps(s1, h1v));
    _mm256_store_ps(v2 + x, _mm256_add_ps(s2, h2v));
}

void cov_box(const float* p, const float* q, float* dst, int W, int H, int r,
                const float* pSub, const float* eps, float keps, float nrm,
                const uint8_t* pSubU8, const uint8_t* epsU8, float epsScale) {
    const float inv = 1.0f / (float)((2 * r + 1) * (2 * r + 1));
    const __m256 vinv = _mm256_set1_ps(inv);
    par_rows(H, [&](int ya, int yb) {
        Cws& w = get_cws(W, r);
        const int PAD = w.PAD, WP = w.WP, depth = w.depth;
        int slot = 0;
        for (int i = 0; i < 3; ++i) std::memset(w.v[i], 0, sizeof(float) * (size_t)WP);
        for (int t = ya - r; t < yb + r; ++t) {
            const int ty = clampi(t, 0, H - 1);
            const float* ps = p + (size_t)ty * W;
            const float* qs = q + (size_t)ty * W;
            const float* ss = pSub ? (pSub + (size_t)ty * W) : nullptr;
            const uint8_t* ss8 = pSubU8 ? (pSubU8 + (size_t)ty * W) : nullptr;
            if (ss || ss8) {                       // 行内直接求 p - pSub（s1 = g - up1），不额外 copy
                const float lv = ps[0] - (ss8 ? (float)ss8[0] : ss[0]), rv2 = ps[W - 1] - (ss8 ? (float)ss8[W - 1] : ss[W - 1]);
                for (int i = 0; i < PAD; ++i) w.pr[i] = lv;
                if (ss8) {
                    for (int i = PAD; i < PAD + W; i += 8) {
                        const __m256 sv = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(ss8 + (i - PAD)))));
                        _mm256_storeu_ps(w.pr + i, _mm256_sub_ps(_mm256_loadu_ps(ps + (i - PAD)), sv));
                    }
                } else {
                    for (int i = PAD; i < PAD + W; i += 8)
                        _mm256_storeu_ps(w.pr + i, _mm256_sub_ps(_mm256_loadu_ps(ps + (i - PAD)),
                                                                 _mm256_loadu_ps(ss + (i - PAD))));
                }
                for (int i = PAD + W; i < WP; ++i) w.pr[i] = rv2;
            } else {
                for (int i = 0; i < PAD; ++i) w.pr[i] = ps[0];
                std::memcpy(w.pr + PAD, ps, sizeof(float) * (size_t)W);
                const float pv = ps[W - 1];
                for (int i = PAD + W; i < WP; ++i) w.pr[i] = pv;
            }
            for (int i = 0; i < PAD; ++i) w.qr[i] = qs[0];
            std::memcpy(w.qr + PAD, qs, sizeof(float) * (size_t)W);
            const float qv = qs[W - 1];
            for (int i = PAD + W; i < WP; ++i) w.qr[i] = qv;
            const bool warm = (t - (ya - r)) >= depth;
            float q0 = 0.f, q1 = 0.f, q2 = 0.f;
            for (int j = -r; j <= r; ++j) {
                const float a = w.pr[PAD - 1 + j], b = w.qr[PAD - 1 + j];
                q0 += a; q1 += b; q2 += a * b;
            }
            float* h0 = w.h[0] + (size_t)slot * WP;
            float* h1 = w.h[1] + (size_t)slot * WP;
            float* h2 = w.h[2] + (size_t)slot * WP;
            for (int x = PAD; x < PAD + W; x += 8)
                cov_step(w.pr, w.qr, x, r, warm, q0, q1, q2, h0, h1, h2, w.v[0], w.v[1], w.v[2]);
            slot = (slot + 1) % depth;
            const int v = t - r;
            if (v >= ya) {
                const float* va = w.v[0];
                const float* vb = w.v[1];
                const float* vc = w.v[2];
                float* d = dst + (size_t)v * W;
                const float* er = eps ? (eps + (size_t)v * W) : nullptr;
                const uint8_t* er8 = epsU8 ? (epsU8 + (size_t)v * W) : nullptr;
                const __m256 veps = _mm256_set1_ps(epsScale);
                int x = PAD;
                for (; x + 8 <= PAD + W; x += 8) {
                    const __m256 a = _mm256_load_ps(va + x);
                    const __m256 b = _mm256_load_ps(vb + x);
                    const __m256 cc = _mm256_load_ps(vc + x);
                    __m256 cov = _mm256_mul_ps(_mm256_sub_ps(cc, _mm256_mul_ps(_mm256_mul_ps(a, b), vinv)), vinv);
                    if (er || er8) {
                        // t = clamp01((|cov| - keps*eps^2) / nrm)
                        const __m256 ev = er8 ? _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(er8 + (x - PAD))))), veps)
                                              : _mm256_loadu_ps(er + (x - PAD));
                        cov = _mm256_sub_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), cov),
                                            _mm256_mul_ps(_mm256_mul_ps(ev, ev), _mm256_set1_ps(keps)));
                        cov = _mm256_mul_ps(cov, _mm256_set1_ps(1.f / nrm));
                        cov = _mm256_min_ps(_mm256_max_ps(cov, _mm256_setzero_ps()), _mm256_set1_ps(1.f));
                    }
                    _mm256_storeu_ps(d + (x - PAD), cov);
                }
                for (; x < PAD + W; ++x) {
                    float cov = (vc[x] - va[x] * vb[x] * inv) * inv;
                    if (er || er8) {
                        const float e = er8 ? (float)er8[x - PAD] * epsScale : er[x - PAD];
                        cov = (std::fabs(cov) - keps * e * e) / nrm;
                        cov = cov < 0.f ? 0.f : (cov > 1.f ? 1.f : cov);
                    }
                    d[x - PAD] = cov;
                }
            }
        }
    });
}

// 取 p[2x], p[2x+2], ..., p[2x+14]（偶列）
static inline __m256 load_even(const float* p, int x) {
    const __m256 a = _mm256_loadu_ps(p + 2 * x);
    const __m256 b = _mm256_loadu_ps(p + 2 * x + 8);
    const __m256 t = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
    return _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(t), 0xD8));
}
// 取 p[2x+1], p[2x+3], ..., p[2x+15]（奇列）
static inline __m256 load_odd(const float* p, int x) {
    const __m256 a = _mm256_loadu_ps(p + 2 * x);
    const __m256 b = _mm256_loadu_ps(p + 2 * x + 8);
    const __m256 t = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));
    return _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(t), 0xD8));
}

void downsample2(const float* src, float* dst, int W, int H) {
    const int dw = W / 2, dh = H / 2;
    for (int y = 0; y < dh; ++y) {
        const float* s0 = src + (size_t)(2 * y) * W;
        const float* s1 = src + (size_t)(2 * y + 1) * W;
        float* d = dst + (size_t)y * dw;
        int x = 0;
        for (; x + 8 <= dw; x += 8) {
            const __m256 acc = _mm256_add_ps(_mm256_add_ps(load_even(s0, x), load_odd(s0, x)),
                                             _mm256_add_ps(load_even(s1, x), load_odd(s1, x)));
            _mm256_storeu_ps(d + x, _mm256_mul_ps(acc, _mm256_set1_ps(0.25f)));
        }
        for (; x < dw; ++x)
            d[x] = 0.25f * (s0[2 * x] + s0[2 * x + 1] + s1[2 * x] + s1[2 * x + 1]);
    }
}
// u8 -> float 的 2x2 均值下采样（p 用原始 u8，k=255/b=0）
void downsample2_u8f(const uint8_t* src, float* dst, int W, int H) {
    const int w2 = W / 2, h2 = H / 2;
    par_rows(h2, [&](int ya, int yb) {
        for (int y = ya; y < yb; ++y) {
            const uint8_t* r0 = src + (size_t)(2 * y) * W;
            const uint8_t* r1 = src + (size_t)(2 * y + 1) * W;
            float* d = dst + (size_t)y * w2;
            for (int x = 0; x < w2; ++x)
                d[x] = 0.25f * ((float)r0[2 * x] + r0[2 * x + 1] + r1[2 * x] + r1[2 * x + 1]);
        }
    });
}

// ---------------------------------------------------------------- 上采样
static inline float cubic_w(float t) {           // a = -0.5
    const float at = std::fabs(t);
    if (at <= 1.f) return ((1.5f * at - 2.5f) * at) * at + 1.f;
    if (at < 2.f)  return (((-0.5f * at + 2.5f) * at - 4.f) * at) + 2.f;
    return 0.f;
}

void upsample(const float* src, int sw, int sh, float* dst, int dw, int dh, bool bic) {
    const double sx = (double)sw / dw, sy = (double)sh / dh;
    float* tmp = scratch(6, (size_t)sh * dw);
    par_rows(sh, [&](int ya, int yb) {
    // 按输出列预计算索引/权重（每个线程一份，dw 量级，很便宜）
    thread_local std::vector<int> gi0, gi1, gi2, gi3;
    thread_local std::vector<float> gw0, gw1, gw2, gw3;
    const int NT = 4;
    if ((int)gi0.size() < dw) {
        gi0.resize(dw); gi1.resize(dw); gi2.resize(dw); gi3.resize(dw);
        gw0.resize(dw); gw1.resize(dw); gw2.resize(dw); gw3.resize(dw);
    }
    for (int x = 0; x < dw; ++x) {
        const double u = (x + 0.5) * sx - 0.5;
        const int j = (int)std::floor(u);
        if (!bic) {
            gi0[x] = clampi(j, 0, sw - 1); gi1[x] = clampi(j + 1, 0, sw - 1);
            gw0[x] = (float)(u - j);
        } else {
            int* ii[4] = { gi0.data(), gi1.data(), gi2.data(), gi3.data() };
            float* ww[4] = { gw0.data(), gw1.data(), gw2.data(), gw3.data() };
            for (int k = 0; k < NT; ++k) {
                ii[k][x] = clampi(j - 1 + k, 0, sw - 1);
                ww[k][x] = cubic_w((float)(u - (j - 1 + k)));
            }
        }
    }
    for (int y = ya; y < yb; ++y) {
        const float* s = src + (size_t)y * sw;
        float* d = tmp + (size_t)y * dw;
        const int x8 = dw & ~7;
        for (int x = 0; x < x8; x += 8) {
            const __m256i a0 = _mm256_loadu_si256((const __m256i*)(gi0.data() + x));
            const __m256i a1 = _mm256_loadu_si256((const __m256i*)(gi1.data() + x));
            if (!bic) {
                const __m256 sa = _mm256_i32gather_ps(s, a0, 4);
                const __m256 sb = _mm256_i32gather_ps(s, a1, 4);
                const __m256 t = _mm256_loadu_ps(gw0.data() + x);
                _mm256_storeu_ps(d + x, _mm256_add_ps(sa, _mm256_mul_ps(t, _mm256_sub_ps(sb, sa))));
            } else {
                const __m256i a2 = _mm256_loadu_si256((const __m256i*)(gi2.data() + x));
                const __m256i a3 = _mm256_loadu_si256((const __m256i*)(gi3.data() + x));
                __m256 acc = _mm256_mul_ps(_mm256_i32gather_ps(s, a0, 4), _mm256_loadu_ps(gw0.data() + x));
                acc = _mm256_fmadd_ps(_mm256_i32gather_ps(s, a1, 4), _mm256_loadu_ps(gw1.data() + x), acc);
                acc = _mm256_fmadd_ps(_mm256_i32gather_ps(s, a2, 4), _mm256_loadu_ps(gw2.data() + x), acc);
                acc = _mm256_fmadd_ps(_mm256_i32gather_ps(s, a3, 4), _mm256_loadu_ps(gw3.data() + x), acc);
                _mm256_storeu_ps(d + x, acc);
            }
        }
        for (int x = x8; x < dw; ++x) {
            if (!bic) {
                const float a = s[gi0[x]], b = s[gi1[x]], t = gw0[x];
                d[x] = a + t * (b - a);
            } else {
                d[x] = gw0[x] * s[gi0[x]] + gw1[x] * s[gi1[x]] + gw2[x] * s[gi2[x]] + gw3[x] * s[gi3[x]];
            }
        }
    }    });
    par_rows(dh, [&](int ya, int yb) {
    for (int y = ya; y < yb; ++y) {
        const double u = (y + 0.5) * sy - 0.5;
        const int j = (int)std::floor(u);
        float* d = dst + (size_t)y * dw;
        if (!bic) {
            const float t = (float)(u - j);
            const float* r0 = tmp + (size_t)clampi(j, 0, sh - 1) * dw;
            const float* r1 = tmp + (size_t)clampi(j + 1, 0, sh - 1) * dw;
            const int x8 = dw & ~7;
            for (int x = 0; x < x8; x += 8) {
                const __m256 a = _mm256_loadu_ps(r0 + x), b = _mm256_loadu_ps(r1 + x);
                _mm256_storeu_ps(d + x, _mm256_add_ps(a, _mm256_mul_ps(_mm256_set1_ps(t),
                                                                      _mm256_sub_ps(b, a))));
            }
            for (int x = x8; x < dw; ++x) d[x] = r0[x] + t * (r1[x] - r0[x]);
        } else {
            float* acc = scratch(7, (size_t)dw);
            const int x8 = dw & ~7;
            for (int x = 0; x < x8; x += 8) _mm256_storeu_ps(acc + x, _mm256_setzero_ps());
            for (int x = x8; x < dw; ++x) acc[x] = 0.f;
            for (int k = -1; k <= 2; ++k) {
                const float w = cubic_w((float)(u - (j + k)));
                if (w == 0.f) continue;
                const float* r = tmp + (size_t)clampi(j + k, 0, sh - 1) * dw;
                const __m256 vw = _mm256_set1_ps(w);
                for (int x = 0; x < x8; x += 8)
                    _mm256_storeu_ps(acc + x, _mm256_add_ps(_mm256_loadu_ps(acc + x),
                                                            _mm256_mul_ps(_mm256_loadu_ps(r + x), vw)));
                for (int x = x8; x < dw; ++x) acc[x] += w * r[x];
            }
            std::memcpy(d, acc, sizeof(float) * (size_t)dw);
        }
    }
    });
}

// 双线性上采样，输出 u8（round + clamp），省 DRAM（ε/↑d2 中转）。scale 用于 ε 的缩放（epsScale 取 1/scale）
void upsample_u8(const float* src, int sw, int sh, uint8_t* dst, int dw, int dh, float scale) {
    const double sx = (double)sw / dw, sy = (double)sh / dh;
    float* tmp = scratch(6, (size_t)sh * dw);
    par_rows(sh, [&](int ya, int yb) {
        thread_local std::vector<int> gi0, gi1;
        thread_local std::vector<float> gw0;
        if ((int)gi0.size() < dw) { gi0.resize(dw); gi1.resize(dw); gw0.resize(dw); }
        for (int x = 0; x < dw; ++x) {
            const double u = (x + 0.5) * sx - 0.5;
            const int j = (int)std::floor(u);
            gi0[x] = clampi(j, 0, sw - 1); gi1[x] = clampi(j + 1, 0, sw - 1);
            gw0[x] = (float)(u - (double)j);
        }
        for (int y = ya; y < yb; ++y) {
            const float* s = src + (size_t)y * sw;
            float* d = tmp + (size_t)y * dw;
            const int x8 = dw & ~7;
            for (int x = 0; x < x8; x += 8) {
                const __m256i a0 = _mm256_loadu_si256((const __m256i*)(gi0.data() + x));
                const __m256i a1 = _mm256_loadu_si256((const __m256i*)(gi1.data() + x));
                const __m256 sa = _mm256_i32gather_ps(s, a0, 4);
                const __m256 sb = _mm256_i32gather_ps(s, a1, 4);
                const __m256 t = _mm256_loadu_ps(gw0.data() + x);
                _mm256_storeu_ps(d + x, _mm256_add_ps(sa, _mm256_mul_ps(t, _mm256_sub_ps(sb, sa))));
            }
            for (int x = x8; x < dw; ++x) {
                const float a = s[gi0[x]], b = s[gi1[x]], t = gw0[x];
                d[x] = a + t * (b - a);
            }
        }
    });
    par_rows(dh, [&](int ya, int yb) {
        const __m256 v0 = _mm256_setzero_ps(), v255 = _mm256_set1_ps(255.f);
        for (int y = ya; y < yb; ++y) {
            const double u = (y + 0.5) * sy - 0.5;
            const int j = (int)std::floor(u);
            const float t = (float)(u - (double)j);
            const float* r0 = tmp + (size_t)clampi(j, 0, sh - 1) * dw;
            const float* r1 = tmp + (size_t)clampi(j + 1, 0, sh - 1) * dw;
            uint8_t* d = dst + (size_t)y * dw;
            const __m256 vt = _mm256_set1_ps(t), vs = _mm256_set1_ps(scale);
            const int x8 = dw & ~7;
            const __m256 vhalf = _mm256_set1_ps(0.5f);
            for (int x = 0; x < x8; x += 8) {
                const __m256 a = _mm256_loadu_ps(r0 + x), b = _mm256_loadu_ps(r1 + x);
                __m256 q = _mm256_add_ps(a, _mm256_mul_ps(vt, _mm256_sub_ps(b, a)));
                q = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(q, vs), vhalf), v0), v255);
                const __m256i qi = _mm256_cvttps_epi32(q);
                const __m128i lo = _mm256_castsi256_si128(qi);
                const __m128i hi = _mm256_extracti128_si256(qi, 1);
                const __m128i p16 = _mm_packs_epi32(lo, hi);
                const __m128i q8 = _mm_packus_epi16(p16, p16);
                _mm_storel_epi64((__m128i*)(d + x), q8);
            }
            for (int x = x8; x < dw; ++x) {
                const float q = r0[x] + t * (r1[x] - r0[x]);
                d[x] = (uint8_t)clampi((int)(q * scale + 0.5f), 0, 255);
            }
        }
    });
}

void bitonic(const float* src, float* dst, int W, int H) {
    auto refl = [](int v, int n) { return v < 0 ? (-v - 1) : (v >= n ? 2 * n - v - 1 : v); };
    std::vector<float> mn((size_t)W * H), r4((size_t)W * H);
    const int dx[5] = { 0, -1, 1, 0, 0 };
    const int dy[5] = { 0, 0, 0, -1, 1 };
    auto second_largest = [](float a, float b, float c, float d, float e) {
        const float m1 = std::max(std::max(std::max(a, b), std::max(c, d)), e);
        int cnt = (a == m1) + (b == m1) + (c == m1) + (d == m1) + (e == m1);
        if (cnt >= 2) return m1;
        float m2 = -std::numeric_limits<float>::max();
        if (a != m1) m2 = std::max(m2, a);
        if (b != m1) m2 = std::max(m2, b);
        if (c != m1) m2 = std::max(m2, c);
        if (d != m1) m2 = std::max(m2, d);
        if (e != m1) m2 = std::max(m2, e);
        return m2;
    };
    auto tap = [&](const float* p, int y, int x, int k) { return p[(size_t)refl(y + dy[k], H) * W + refl(x + dx[k], W)]; };
    par_rows(H, [&](int ya, int yb) {
    for (int y = ya; y < yb; ++y)
        for (int x = 0; x < W; ++x) {
            float m = 1e30f;
            for (int k = 0; k < 5; ++k) m = std::min(m, tap(src, y, x, k));
            mn[(size_t)y * W + x] = m;
            r4[(size_t)y * W + x] = second_largest(tap(src, y, x, 0), tap(src, y, x, 1), tap(src, y, x, 2),
                                                   tap(src, y, x, 3), tap(src, y, x, 4));
        }
    });

    par_rows(H, [&](int ya2, int yb2) {
    for (int y = ya2; y < yb2; ++y)
        for (int x = 0; x < W; ++x) {
            const float r1v = second_largest(tap(&mn[0], y, x, 0), tap(&mn[0], y, x, 1), tap(&mn[0], y, x, 2),
                                             tap(&mn[0], y, x, 3), tap(&mn[0], y, x, 4));
            float r2v = 1e30f;
            for (int k = 0; k < 5; ++k) r2v = std::min(r2v, tap(&r4[0], y, x, k));
            const float a = src[(size_t)y * W + x];
            const float w1 = std::fabs(r1v - a) + 0.01f;
            const float w2 = std::fabs(r2v - a) + 0.01f;
            dst[(size_t)y * W + x] = (w1 * r1v + w2 * r2v) / (w1 + w2);
        }
    });
}

template <class SRC> void block_noise_impl(const SRC* img, float* outVar, int W, int H) {
// 8 点 DFT 的 cos/sin 表（编译期常量）
    static constexpr float C[8][8] = {
        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f },
        { 1.0f, 0.7071067811865476f, 0.0f, -0.7071067811865475f, -1.0f, -0.7071067811865477f, 0.0f, 0.7071067811865474f },
        { 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f },
        { 1.0f, -0.7071067811865475f, 0.0f, 0.7071067811865477f, -1.0f, 0.7071067811865466f, 0.0f, -0.7071067811865474f },
        { 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f },
        { 1.0f, -0.7071067811865477f, 0.0f, 0.7071067811865466f, -1.0f, 0.7071067811865475f, 0.0f, -0.7071067811865461f },
        { 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f },
        { 1.0f, 0.7071067811865474f, 0.0f, -0.7071067811865474f, -1.0f, -0.7071067811865461f, 0.0f, 0.7071067811865505f },
    };
    static constexpr float S[8][8] = {
        { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
        { 0.0f, 0.7071067811865476f, 1.0f, 0.7071067811865476f, 0.0f, -0.7071067811865475f, -1.0f, -0.7071067811865477f },
        { 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f },
        { 0.0f, 0.7071067811865476f, -1.0f, 0.7071067811865474f, 0.0f, -0.7071067811865485f, 1.0f, -0.7071067811865477f },
        { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
        { 0.0f, -0.7071067811865475f, 1.0f, -0.7071067811865485f, 0.0f, 0.7071067811865476f, -1.0f, 0.7071067811865489f },
        { 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f },
        { 0.0f, -0.7071067811865477f, -1.0f, -0.7071067811865477f, 0.0f, 0.7071067811865489f, 1.0f, 0.7071067811865446f },
    };
    const int Hb = H / 8, Wb = W / 8;
    const __m256 vk = _mm256_set1_ps(0.0017798f), vlo = _mm256_set1_ps(0.01f);

    par_rows(Hb, [&](int b0, int b1) {
        for (int by = b0; by < b1; ++by) {
            int bx = 0;
            // ---- SoA-8：8 个水平相邻块同时算（lane = 块）
            for (; bx + 8 <= Wb; bx += 8) {
                __m256 Rre[4][8], Rim[4][8], E = _mm256_setzero_ps();
                for (int y = 0; y < 8; ++y) {
                    const SRC* rp = img + (size_t)(by * 8 + y) * W + (size_t)bx * 8;
                    for (int t = 0; t < 4; ++t) { Rre[t][y] = _mm256_setzero_ps(); Rim[t][y] = _mm256_setzero_ps(); }
                    for (int x = 0; x < 8; ++x) {
                        alignas(32) float v[8];
                        v[0] = (float)rp[x];       v[1] = (float)rp[8 + x];   v[2] = (float)rp[16 + x];  v[3] = (float)rp[24 + x];
                        v[4] = (float)rp[32 + x];  v[5] = (float)rp[40 + x];  v[6] = (float)rp[48 + x];  v[7] = (float)rp[56 + x];
                        const __m256 vx = _mm256_load_ps(v);
                        for (int t = 0; t < 4; ++t) {
                            Rre[t][y] = _mm256_fmadd_ps(vx, _mm256_set1_ps(C[t + 2][x]), Rre[t][y]);
                            Rim[t][y] = _mm256_fmadd_ps(vx, _mm256_set1_ps(-S[t + 2][x]), Rim[t][y]);
                        }
                    }
                }
                for (int t = 0; t < 4; ++t)
                    for (int u = 0; u < 4; ++u) {
                        __m256 fre = _mm256_setzero_ps(), fim = _mm256_setzero_ps();
                        for (int y = 0; y < 8; ++y) {
                            fre = _mm256_fmadd_ps(Rre[t][y], _mm256_set1_ps(C[u + 2][y]), fre);
                            fre = _mm256_fmadd_ps(Rim[t][y], _mm256_set1_ps(S[u + 2][y]), fre);
                            fim = _mm256_fmadd_ps(Rim[t][y], _mm256_set1_ps(C[u + 2][y]), fim);
                            fim = _mm256_fnmadd_ps(Rre[t][y], _mm256_set1_ps(S[u + 2][y]), fim);
                        }
                        E = _mm256_fmadd_ps(fre, fre, E);
                        E = _mm256_fmadd_ps(fim, fim, E);
                    }
                _mm256_storeu_ps(outVar + (size_t)by * Wb + bx,
                                 _mm256_max_ps(vlo, _mm256_mul_ps(E, vk)));
            }
            // ---- 尾部（Wb 不是 8 的倍数）用标量
            for (; bx < Wb; ++bx) {
                float Rre[4][8], Rim[4][8];
                for (int y = 0; y < 8; ++y) {
                    const SRC* v = img + (size_t)(by * 8 + y) * W + (size_t)bx * 8;
                    for (int t = 0; t < 4; ++t) {
                        const int l = t + 2;
                        float re = 0.f, im = 0.f;
                        for (int x = 0; x < 8; ++x) { const float vf = (float)v[x]; re += vf * C[l][x]; im -= vf * S[l][x]; }
                        Rre[t][y] = re; Rim[t][y] = im;
                    }
                }
                float E = 0.f;
                for (int t = 0; t < 4; ++t)
                    for (int u = 0; u < 4; ++u) {
                        const int k = u + 2;
                        float fre = 0.f, fim = 0.f;
                        for (int y = 0; y < 8; ++y) {
                            fre += Rre[t][y] * C[k][y] + Rim[t][y] * S[k][y];
                            fim += Rim[t][y] * C[k][y] - Rre[t][y] * S[k][y];
                        }
                        E += fre * fre + fim * fim;
                    }
                outVar[(size_t)by * Wb + bx] = std::max(0.01f, 0.0017798f * E);
            }
        }
    });
}
void block_noise(const float* img, float* outVar, int W, int H) { block_noise_impl(img, outVar, W, H); }
void block_noise_u8(const uint8_t* img, float* outVar, int W, int H) { block_noise_impl(img, outVar, W, H); }
} // namespace agf_detail

// =============================================================================
//  融合内核：一次调用只读/写各一遍平面（P0 优化）
//  ---------------------------------------------------------------------------
//  形状与 bench_box 验证过的完全一致：一条 "pair" 用一次 pair_step 产出
//  box(a) 与 box(a*b)，stage1 两组 pair，stage2 一组 pair。
//  行流水：读入 t 行 -> stage1(滞后 r1) -> a/b -> stage2(再滞后 r2) -> 输出 v 行。
// =============================================================================
namespace {



Fws& get_fws(int W, int NC, int tileW, int r1, int r2) {
    static thread_local Fws w;
    w.ensure(W, NC, tileW, r1, r2);
    return w;
}

// 填一行（带 blend + replicate 边）：按列分块 [x0, x0+wt8) 切片
inline void fill_tile(float* __restrict dst, const float* __restrict base,
                      const float* __restrict alt, const uint8_t* __restrict alt8,
                      const uint8_t* __restrict mk, const uint8_t* __restrict base8,
                      int W, int x0, int wt8, int PAD, int WP) {
    auto baseAt = [&](int i) -> float { return base8 ? (float)base8[i] : base[i]; };
    auto altAt = [&](int i) -> float { return alt8 ? (float)alt8[i] : alt[i]; };
    const int total = PAD + wt8 + PAD;
    const int gstart = x0 - PAD;
    int i0 = (gstart < 0) ? -gstart : 0;
    int i1 = (gstart + total > W) ? (W - gstart) : total;
    if (i1 < i0) i1 = i0;
    const float lv = mk ? (mk[0] ? altAt(0) : baseAt(0)) : baseAt(0);
    const float rv = mk ? (mk[W - 1] ? altAt(W - 1) : baseAt(W - 1)) : baseAt(W - 1);
    for (int i = 0; i < i0; ++i) dst[i] = lv;
    if (mk) {
        const __m256i z = _mm256_setzero_si256();
        for (int i = i0; i < i1; i += 8) {
            const __m256i mi = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(mk + gstart + i)));
            const __m256 mf = _mm256_castsi256_ps(_mm256_cmpgt_epi32(mi, z));
            const __m256 bv = base8 ? _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(base8 + gstart + i))))
                                    : _mm256_loadu_ps(base + gstart + i);
            const __m256 av = alt8 ? _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(alt8 + gstart + i))))
                                   : _mm256_loadu_ps(alt + gstart + i);
            _mm256_storeu_ps(dst + i, _mm256_blendv_ps(bv, av, mf));
        }
    } else {
        for (int i = i0; i < i1; i += 8) {
            const __m256 bv = base8 ? _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(base8 + gstart + i))))
                                    : _mm256_loadu_ps(base + gstart + i);
            _mm256_storeu_ps(dst + i, bv);
        }
    }
    for (int i = i1; i < WP; ++i) dst[i] = rv;
}
// 一个 pair 的 8 像素：box(a) 与 box(b)（两个独立数组，用于 stage 2）
inline void pair_step_ab(const float* __restrict ra, const float* __restrict rb, int x, int r, bool warm,
                         float& q1, float& q2,
                         float* __restrict h1, float* __restrict h2,
                         float* __restrict v1, float* __restrict v2) {
    const __m256 na = _mm256_loadu_ps(ra + x + r);
    const __m256 oa = _mm256_loadu_ps(ra + x - r - 1);
    const __m256 nb = _mm256_loadu_ps(rb + x + r);
    const __m256 ob = _mm256_loadu_ps(rb + x - r - 1);
    const __m256 h1v = _mm256_add_ps(_mm256_set1_ps(q1), prefix8(_mm256_sub_ps(na, oa)));
    const __m256 h2v = _mm256_add_ps(_mm256_set1_ps(q2), prefix8(_mm256_sub_ps(nb, ob)));
    q1 = last_lane(h1v);
    q2 = last_lane(h2v);
    const __m256 o1 = _mm256_load_ps(h1 + x);
    const __m256 o2 = _mm256_load_ps(h2 + x);
    _mm256_store_ps(h1 + x, h1v);
    _mm256_store_ps(h2 + x, h2v);
    __m256 s1 = _mm256_load_ps(v1 + x);
    __m256 s2 = _mm256_load_ps(v2 + x);
    if (warm) { s1 = _mm256_sub_ps(s1, o1); s2 = _mm256_sub_ps(s2, o2); }
    _mm256_store_ps(v1 + x, _mm256_add_ps(s1, h1v));
    _mm256_store_ps(v2 + x, _mm256_add_ps(s2, h2v));
}

// 一个 pair 的 8 像素：box(a) 与 box(a*b)
inline void pair_step(const float* __restrict ra, const float* __restrict rb, int x, int r, bool warm,
                      float& q1, float& q2,
                      float* __restrict h1, float* __restrict h2,
                      float* __restrict v1, float* __restrict v2) {
    const __m256 na = _mm256_loadu_ps(ra + x + r);
    const __m256 oa = _mm256_loadu_ps(ra + x - r - 1);
    const __m256 nb = _mm256_loadu_ps(rb + x + r);
    const __m256 ob = _mm256_loadu_ps(rb + x - r - 1);
    const __m256 h1v = _mm256_add_ps(_mm256_set1_ps(q1), prefix8(_mm256_sub_ps(na, oa)));
    const __m256 h2v = _mm256_add_ps(_mm256_set1_ps(q2),
        prefix8(_mm256_sub_ps(_mm256_mul_ps(na, nb), _mm256_mul_ps(oa, ob))));
    q1 = last_lane(h1v);
    q2 = last_lane(h2v);
    const __m256 o1 = _mm256_load_ps(h1 + x);
    const __m256 o2 = _mm256_load_ps(h2 + x);
    _mm256_store_ps(h1 + x, h1v);
    _mm256_store_ps(h2 + x, h2v);
    __m256 s1 = _mm256_load_ps(v1 + x);
    __m256 s2 = _mm256_load_ps(v2 + x);
    if (warm) { s1 = _mm256_sub_ps(s1, o1); s2 = _mm256_sub_ps(s2, o2); }
    _mm256_store_ps(v1 + x, _mm256_add_ps(s1, h1v));
    _mm256_store_ps(v2 + x, _mm256_add_ps(s2, h2v));
}

} // namespace

// =============================================================================
namespace {

struct GfArgs {
    const float* guide = nullptr;
    const float* guideAlt = nullptr;
    const uint8_t* guideMask = nullptr;
    const uint8_t* guideU8 = nullptr;        // guide 本体从 u8 取（省 DRAM）
    const uint8_t* mask = nullptr;          // 作用于 p
    const float* p[4] = {};
    const float* pAlt[4] = {};
    const uint8_t* pAltU8[4] = {};   // p 的 alt 直接从 u8 取（k=255/b=0 时 y==X）
    const uint8_t* pU8[4] = {};      // p 本体从 u8 取（省 DRAM）
    float* out[4] = {};
    const float* eps[4] = {};
    const uint8_t* epsU8[4] = {};    // eps 从 u8 取（省 DRAM）
    float epsScale = 1.0f;
    int NC = 1;
    int r1 = 3, r2 = 3;
    bool centerGuide = true;
    bool clampOut = false;
    bool fused = true;
    uint8_t* outMask = nullptr;
    float maskThr = 0.5f;
    int  tile = 0;      // 0 = 不分块
};

// 多通道 + 列分块的融合实现：每个平面只读一遍、输出只写一遍
void gf_fused(const GfArgs& A, int W, int H, int nthr) {
    const int NC = A.NC, r1 = A.r1, r2 = A.r2;
    ThreadPool& pool = get_pool(nthr);
    const int nt = pool.threads();
    const int ns = nstrip_for(H);
    const int tile = (g_gf_tile > 0 && g_gf_tile < W) ? g_gf_tile : W;
    const float inv1 = 1.0f / (float)((2 * r1 + 1) * (2 * r1 + 1));
    const float inv2 = 1.0f / (float)((2 * r2 + 1) * (2 * r2 + 1));
    const __m256 vinv1 = _mm256_set1_ps(inv1), vinv2 = _mm256_set1_ps(inv2);
    const __m256 vzero = _mm256_setzero_ps(), v255 = _mm256_set1_ps(255.f);
    const __m256 v128 = _mm256_set1_ps(128.f);
    const __m256 vscale = _mm256_set1_ps(A.epsScale);
    const int nS1 = 2 + 2 * NC, nS2 = 2 * NC;

    pool.run(nt, [&](int tid) {
        for (int k = tid; k < ns; k += nt) {
            const int y0 = (int)((long long)H * k / ns);
            const int y1 = (int)((long long)H * (k + 1) / ns);
            if (y1 <= y0) continue;
            for (int x0 = 0; x0 < W; x0 += tile) {
                const int wt  = (x0 + tile <= W) ? tile : (W - x0);
                const int wt8 = align8i(wt);
                Fws& w = get_fws(W, NC, tile, r1, r2);
                const int PAD = w.PAD, WP = w.WP, cD = w.cD, h1D = w.h1D, h2D = w.h2D;
                const int maxH1D = w.maxCD, maxH2D = w.maxCD;
                const int t0 = y0 - (r1 + r2), tEnd = y1 + (r1 + r2);
                const int uStart = (y0 - r2 > 0) ? (y0 - r2) : 0;
                const int xBeg = PAD, xEnd = PAD + wt8;
                int hx0 = (PAD - r2 - 8) & ~7;
                int hx1 = (PAD + wt8 + r2 + 7) & ~7;
                if (hx0 < r1 + 1) hx0 = align8i(r1 + 1);
                if (hx1 + r1 > WP) hx1 = (WP - r1) & ~7;
                const int hx2 = (hx1 + r2 > WP) ? ((WP - r2) & ~7) : hx1;
                int cSlot = 0, h1Slot = 0, h2Slot = 0;
                for (int s = 0; s < nS1; ++s) std::memset(w.v1 + (size_t)s * WP, 0, sizeof(float) * (size_t)WP);
                for (int s = 0; s < nS2; ++s) std::memset(w.v2 + (size_t)s * WP, 0, sizeof(float) * (size_t)WP);

                for (int t = t0; t < tEnd; ++t) {
                    const int ty = clampi(t, 0, H - 1);
                    const bool warm1 = (t - t0) >= h1D;

                    // ---- 引导行（blend + 可选中心化）
                    {
                        float* dst = w.cRing + (size_t)cSlot * WP;
                        fill_tile(dst, A.guide + (size_t)ty * W,
                                  A.guideAlt ? A.guideAlt + (size_t)ty * W : nullptr, nullptr,
                                  A.guideMask ? A.guideMask + (size_t)ty * W : nullptr,
                                  A.guideU8 ? A.guideU8 + (size_t)ty * W : nullptr,
                                  W, x0, wt8, PAD, WP);
                        if (A.centerGuide)
                            for (int i = 0; i < WP; i += 8)
                                _mm256_storeu_ps(dst + i, _mm256_sub_ps(_mm256_loadu_ps(dst + i), v128));
                    }
                    const float* cRow = w.cRing + (size_t)cSlot * WP;
                    cSlot = (cSlot + 1) % cD;

                    // ---- 各通道 p 行
                    for (int c = 0; c < NC; ++c)
                        fill_tile(w.prow + (size_t)c * WP, A.p[c] + (size_t)ty * W,
                                  A.pAlt[c] ? A.pAlt[c] + (size_t)ty * W : nullptr,
                                  A.pAltU8[c] ? A.pAltU8[c] + (size_t)ty * W : nullptr,
                                  A.mask ? A.mask + (size_t)ty * W : nullptr,
                                  A.pU8[c] ? A.pU8[c] + (size_t)ty * W : nullptr,
                                  W, x0, wt8, PAD, WP);

                    // ---- stage 1：2 条共享流（box(I), box(I^2)）+ 每通道 2 条
                    {
                        float q1 = 0.f, q2 = 0.f;
                        for (int j = -r1; j <= r1; ++j) {
                            const float v = cRow[hx0 - 1 + j];
                            q1 += v; q2 += v * v;
                        }
                        float* hI  = w.h1 + (size_t)h1Slot * WP;
                        float* hI2 = w.h1 + (size_t)(maxH1D + h1Slot) * WP;
for (int x = hx0; x < hx1; x += 8)
pair_step(cRow, cRow, x, r1, warm1, q1, q2, hI, hI2,
                                      w.v1, w.v1 + WP);
                        for (int c = 0; c < NC; ++c) {
                            const float* pRow = w.prow + (size_t)c * WP;
                            float qa = 0.f, qb = 0.f;
                            for (int j = -r1; j <= r1; ++j) {
                                const float pv = pRow[hx0 - 1 + j];
                                const float v = cRow[hx0 - 1 + j];
                                qa += pv; qb += pv * v;
                            }
                            const int s0 = 2 + 2 * c;
                            float* hP  = w.h1 + ((size_t)s0 * maxH1D + h1Slot) * WP;
                            float* hIP = w.h1 + ((size_t)(s0 + 1) * maxH1D + h1Slot) * WP;
for (int x = hx0; x < hx1; x += 8)
pair_step(pRow, cRow, x, r1, warm1, qa, qb, hP, hIP,
                                          w.v1 + (size_t)s0 * WP, w.v1 + (size_t)(s0 + 1) * WP);
                        }
                        h1Slot = (h1Slot + 1) % h1D;
                    }

                    // ---- a/b（算在 [hx0,hx1) 扩展区间，图像边界才 clamp）
                    const int u = t - r1;
                    if (u >= uStart) {
                        if (u < H) {
                            for (int c = 0; c < NC; ++c) {
                                float* ra = w.ra + (size_t)c * WP;
                                float* rb = w.rb + (size_t)c * WP;
                                const float* vI  = w.v1;
                                const float* vI2 = w.v1 + WP;
                                const float* vP  = w.v1 + (size_t)(2 + 2 * c) * WP;
                                const float* vIP = w.v1 + (size_t)(3 + 2 * c) * WP;
                                const float* epRow = A.eps[c] + (size_t)u * W;
                                const uint8_t* epRow8 = A.epsU8[c] ? A.epsU8[c] + (size_t)u * W : nullptr;
                                for (int x = hx0; x < hx1; x += 8) {
                                    const __m256 mI = _mm256_mul_ps(_mm256_load_ps(vI + x), vinv1);
                                    __m256 vv = _mm256_mul_ps(_mm256_load_ps(vI2 + x), vinv1);
                                    const __m256 var = _mm256_max_ps(_mm256_sub_ps(vv, _mm256_mul_ps(mI, mI)), vzero);
                                    const __m256 mp = _mm256_mul_ps(_mm256_load_ps(vP + x), vinv1);
                                    const __m256 cv = _mm256_sub_ps(_mm256_mul_ps(_mm256_load_ps(vIP + x), vinv1),
                                                                    _mm256_mul_ps(mI, mp));
                                    const __m256 epv = epRow8 ? _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(epRow8 + clampi(x0 + x - PAD, 0, W - 1)))))
                                                              : _mm256_loadu_ps(epRow + clampi(x0 + x - PAD, 0, W - 1));
                                    const __m256 den = _mm256_add_ps(var, _mm256_mul_ps(epv, vscale));
                                    const __m256 av = _mm256_div_ps(cv, den);
                                    _mm256_storeu_ps(ra + x, av);
                                    _mm256_storeu_ps(rb + x, _mm256_sub_ps(mp, _mm256_mul_ps(av, mI)));
                                }
                                if (x0 == 0) {
                                    const float a0 = ra[PAD], b0 = rb[PAD];
                                    for (int i = hx0; i < PAD; ++i) { ra[i] = a0; rb[i] = b0; }
                                }
                                if (x0 + wt >= W) {
                                    const float a1 = ra[PAD + wt - 1], b1 = rb[PAD + wt - 1];
                                    for (int i = PAD + wt; i < hx2; ++i) { ra[i] = a1; rb[i] = b1; }
                                }
                            }
                        }
                        const bool warm2 = (u - uStart) >= h2D;
                        for (int c = 0; c < NC; ++c) {
                            const float* ra = w.ra + (size_t)c * WP;
                            const float* rb = w.rb + (size_t)c * WP;
                            float qa = 0.f, qb = 0.f;
                            for (int j = -r2; j <= r2; ++j) { qa += ra[PAD - 1 + j]; qb += rb[PAD - 1 + j]; }
                            float* hA = w.h2 + ((size_t)(2 * c) * maxH2D + h2Slot) * WP;
                            float* hB = w.h2 + ((size_t)(2 * c + 1) * maxH2D + h2Slot) * WP;
                            for (int x = xBeg; x < xEnd; x += 8)
                                pair_step_ab(ra, rb, x, r2, warm2, qa, qb, hA, hB,
                                             w.v2 + (size_t)(2 * c) * WP, w.v2 + (size_t)(2 * c + 1) * WP);
                        }
                        h2Slot = (h2Slot + 1) % h2D;
                    }

                    // ---- 输出
                    const int v = t - r1 - r2;
                    if (v >= y0) {
                        const float* Iv = w.cRing + (size_t)cSlot * WP;
                        const int extra = (y0 == 0 && v < r2) ? (r2 - v) : 0;
                        const __m256 ve = _mm256_set1_ps((float)extra);
                        if (A.outMask) {
                            const float* va = w.v2;
                            const float* vb = w.v2 + WP;
                            const float* ha0 = w.h2;
                            const float* hb0 = w.h2 + (size_t)maxH2D * WP;
                            uint8_t* mo = A.outMask + (size_t)v * W + x0;
                            const __m256 vthr = _mm256_set1_ps(A.maskThr);
                            const int xOut = PAD + wt;
                            int x = xBeg;
                            for (; x + 8 <= xOut; x += 8) {
                                __m256 sa = _mm256_load_ps(va + x);
                                __m256 sb = _mm256_load_ps(vb + x);
                                if (extra) {
                                    sa = _mm256_fmadd_ps(ve, _mm256_load_ps(ha0 + x), sa);
                                    sb = _mm256_fmadd_ps(ve, _mm256_load_ps(hb0 + x), sb);
                                }
                                const __m256 q = _mm256_add_ps(
                                    _mm256_mul_ps(_mm256_mul_ps(sa, vinv2), _mm256_load_ps(Iv + x)),
                                    _mm256_mul_ps(sb, vinv2));
                                const __m256i m32 = _mm256_castps_si256(_mm256_cmp_ps(q, vthr, _CMP_GT_OQ));
                                const __m256i m16 = _mm256_packs_epi32(m32, m32);
                                const __m128i m8 = _mm_packs_epi16(_mm256_castsi256_si128(m16), _mm256_castsi256_si128(m16));
                                _mm_storel_epi64((__m128i*)(mo + (x - PAD)), _mm_and_si128(m8, _mm_set1_epi8(1)));
                            }
                            for (; x < xOut; ++x) {
                                float sa = va[x], sb = vb[x];
                                if (extra) { sa += (float)extra * ha0[x]; sb += (float)extra * hb0[x]; }
                                const float q = sa * inv2 * Iv[x] + sb * inv2;
                                mo[x - PAD] = (q > A.maskThr) ? 1 : 0;
                            }
                            continue;
                        }
                        for (int c = 0; c < NC; ++c) {
                            const float* va = w.v2 + (size_t)(2 * c) * WP;
                            const float* vb = w.v2 + (size_t)(2 * c + 1) * WP;
                            const float* ha0 = w.h2 + (size_t)(2 * c) * maxH2D * WP;
                            const float* hb0 = w.h2 + (size_t)(2 * c + 1) * maxH2D * WP;
                            float* od = A.out[c] + (size_t)v * W + x0;
                            const int xOut = PAD + wt;
                            int x = xBeg;
                            for (; x + 8 <= xOut; x += 8) {
                                __m256 sa = _mm256_load_ps(va + x);
                                __m256 sb = _mm256_load_ps(vb + x);
                                if (extra) {
                                    sa = _mm256_fmadd_ps(ve, _mm256_load_ps(ha0 + x), sa);
                                    sb = _mm256_fmadd_ps(ve, _mm256_load_ps(hb0 + x), sb);
                                }
                                __m256 q = _mm256_add_ps(
                                    _mm256_mul_ps(_mm256_mul_ps(sa, vinv2), _mm256_load_ps(Iv + x)),
                                    _mm256_mul_ps(sb, vinv2));
                                if (A.clampOut) { q = _mm256_max_ps(q, vzero); q = _mm256_min_ps(q, v255); }
                                _mm256_storeu_ps(od + (x - PAD), q);
                            }
                            for (; x < xOut; ++x) {
                                float sa = va[x], sb = vb[x];
                                if (extra) { sa += (float)extra * ha0[x]; sb += (float)extra * hb0[x]; }
                                float q = sa * inv2 * Iv[x] + sb * inv2;
                                if (A.clampOut) { q = q < 0.f ? 0.f : (q > 255.f ? 255.f : q); }
                                od[x - PAD] = q;
                            }
                        }
                    }
                }
            }
        }
    });
}
static bool g_fused = true;
void gf_call(const GfArgs& A, int W, int H, Pool& pool, int nthr = 1) {
    if (A.fused && g_fused) { gf_fused(A, W, H, nthr); return; }
    using namespace agf_detail;
    const int NC = A.NC;
    const size_t N = (size_t)W * H;
    float* I = pool.get(0, N);
    blend_n(I, A.guide, A.guideAlt, A.guideMask, N);
    if (A.centerGuide) {
        const __m256 v128 = _mm256_set1_ps(128.f);
        for (size_t i = 0; i < N; i += 8)
            _mm256_storeu_ps(I + i, _mm256_sub_ps(_mm256_loadu_ps(I + i), v128));
    }
    float* P[4];
    for (int k = 0; k < NC; ++k) {
        P[k] = pool.get(1 + k, N);
        blend_n(P[k], A.p[k], A.pAlt[k], A.mask, N);
    }
    float* mI = pool.get(6, N);
    float* vI = pool.get(7, N);
    float* t1 = pool.get(8, N);
    box_mean(I, mI, W, H, A.r1);
    mul_n(t1, I, I, N);
    box_mean(t1, vI, W, H, A.r1);
    for (size_t i = 0; i < N; i += 8) {
        const __m256 m = _mm256_loadu_ps(mI + i);
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(vI + i), _mm256_mul_ps(m, m));
        v = _mm256_max_ps(v, _mm256_setzero_ps());
        _mm256_storeu_ps(vI + i, v);
    }
    float* mp[4], *cov[4], *av[4], *bv[4];
    for (int k = 0; k < NC; ++k) {
        mp[k] = pool.get(9 + k, N);
        box_mean(P[k], mp[k], W, H, A.r1);
        cov[k] = pool.get(13 + k, N);
        mul_n(t1, I, P[k], N);
        box_mean(t1, cov[k], W, H, A.r1);
        av[k] = pool.get(17 + k, N);
        bv[k] = pool.get(21 + k, N);
        const float* ep = A.eps[k];
        const __m256 vscale = _mm256_set1_ps(A.epsScale);
        for (size_t i = 0; i < N; i += 8) {
            const __m256 mIv = _mm256_loadu_ps(mI + i);
            const __m256 mpv = _mm256_loadu_ps(mp[k] + i);
            const __m256 cv = _mm256_sub_ps(_mm256_loadu_ps(cov[k] + i), _mm256_mul_ps(mIv, mpv));
            const __m256 den = _mm256_add_ps(_mm256_loadu_ps(vI + i),
                                             _mm256_mul_ps(_mm256_loadu_ps(ep + i), vscale));
            const __m256 a = _mm256_div_ps(cv, den);
            _mm256_storeu_ps(av[k] + i, a);
            _mm256_storeu_ps(bv[k] + i, _mm256_sub_ps(mpv, _mm256_mul_ps(a, mIv)));
        }
    }
    float* mb = pool.get(25, N);
    for (int k = 0; k < NC; ++k) {
        box_mean(av[k], A.out[k], W, H, A.r2);
        box_mean(bv[k], mb, W, H, A.r2);
        const __m256 v0 = _mm256_setzero_ps(), v255 = _mm256_set1_ps(255.f);
        for (size_t i = 0; i < N; i += 8) {
            __m256 q = _mm256_add_ps(
                _mm256_mul_ps(_mm256_loadu_ps(A.out[k] + i), _mm256_loadu_ps(I + i)),
                _mm256_loadu_ps(mb + i));
            if (A.clampOut) { q = _mm256_max_ps(q, v0); q = _mm256_min_ps(q, v255); }
            _mm256_storeu_ps(A.out[k] + i, q);
        }
    }
}

inline uint8_t hard_mask(float v, float thr) {
    if (thr <= 0.f) return v > 0.f ? 1 : 0;
    return v >= thr ? 1 : 0;
}
void hard_mask_plane(const float* v, uint8_t* m, size_t n, float thr) {
    par_n(n, [&](int, size_t a, size_t b) { for (size_t i = a; i < b; ++i) m[i] = hard_mask(v[i], thr); });
}

struct U8Pool {
    std::vector<uint8_t*> buf;
    std::vector<size_t> cap;
    uint8_t* get(int i, size_t n) {
        if ((size_t)i >= buf.size()) { buf.resize((size_t)i + 1, nullptr); cap.resize((size_t)i + 1, 0); }
        if (cap[(size_t)i] < n) {
            std::free(buf[(size_t)i]);
            buf[(size_t)i] = (uint8_t*)std::malloc(n);
            cap[(size_t)i] = n;
        }
        return buf[(size_t)i];
    }
    ~U8Pool() { for (uint8_t* p : buf) std::free(p); }
};

} // namespace

// 单行双线性上采样（与 agf_detail::upsample 的 bilinear 映射逐位一致），算列 [xa,xb)（越界 clamp 到图边）
static inline void upsample_row_bilinear(const float* src, int sw, int sh, float* dst, int dw, int dh, int y, int xa, int xb) {
    using namespace agf_detail;
    const double sx = (double)sw / dw, sy = (double)sh / dh;
    const int n = xb - xa;
    thread_local int t_dw = -1, t_sw = -1, t_xa = 0, t_n = 0;
    thread_local std::vector<int> gi0, gi1;
    thread_local std::vector<float> gw0;
    if (t_dw != dw || t_sw != sw || t_xa != xa || t_n != n) {
        t_dw = dw; t_sw = sw; t_xa = xa; t_n = n;
        gi0.resize(n); gi1.resize(n); gw0.resize(n);
        for (int i = 0; i < n; ++i) {
            const int x = clampi(xa + i, 0, dw - 1);
            const double u = (x + 0.5) * sx - 0.5;
            const int j = (int)std::floor(u);
            gi0[i] = clampi(j, 0, sw - 1);
            gi1[i] = clampi(j + 1, 0, sw - 1);
            gw0[i] = (float)(u - (double)j);
        }
    }
    const double v = (y + 0.5) * sy - 0.5;
    const int j = (int)std::floor(v);
    const float t = (float)(v - (double)j);
    const float* r0 = src + (size_t)clampi(j, 0, sh - 1) * sw;
    const float* r1 = src + (size_t)clampi(j + 1, 0, sh - 1) * sw;

    int i = 0;
    const __m256 vt = _mm256_set1_ps(t);
    for (; i + 8 <= n; i += 8) {
        const __m256i a0 = _mm256_loadu_si256((const __m256i*)(gi0.data() + i));
        const __m256i a1 = _mm256_loadu_si256((const __m256i*)(gi1.data() + i));
        const __m256 w = _mm256_loadu_ps(gw0.data() + i);
        const __m256 sa = _mm256_i32gather_ps(r0, a0, 4);
        const __m256 sb = _mm256_i32gather_ps(r0, a1, 4);
        const __m256 ha = _mm256_add_ps(sa, _mm256_mul_ps(w, _mm256_sub_ps(sb, sa)));
        const __m256 sa2 = _mm256_i32gather_ps(r1, a0, 4);
        const __m256 sb2 = _mm256_i32gather_ps(r1, a1, 4);
        const __m256 hb = _mm256_add_ps(sa2, _mm256_mul_ps(w, _mm256_sub_ps(sb2, sa2)));
        _mm256_storeu_ps(dst + i, _mm256_add_ps(ha, _mm256_mul_ps(vt, _mm256_sub_ps(hb, ha))));
    }
    for (; i < n; ++i) {
        const int x = clampi(xa + i, 0, dw - 1);
        const double u = (x + 0.5) * sx - 0.5;
        const int k = (int)std::floor(u);
        const float f = (float)(u - (double)k);
        const int k0 = clampi(k, 0, sw - 1), k1 = clampi(k + 1, 0, sw - 1);
        const float a = r0[k0] + f * (r0[k1] - r0[k0]);
        const float b = r1[k0] + f * (r1[k1] - r1[k0]);
        dst[i] = a + t * (b - a);
    }
}

// 单行双三次上采样（与 agf_detail::upsample 的 bicubic 一致，a=-0.5），算列 [xa,xb)（越界 clamp）
static inline void upsample_row_bicubic(const float* src, int sw, int sh, float* dst, int dw, int dh, int y, int xa, int xb) {
    using namespace agf_detail;
    const double sx = (double)sw / dw, sy = (double)sh / dh;
    const int n = xb - xa;
    // 水平索引/权重按列 [xa,xb) 预计算（每线程一份，越界 clamp 到图边）
    thread_local int t_dw = -1, t_sw = -1, t_xa = 0, t_n = 0;
    thread_local std::vector<int> gi0, gi1, gi2, gi3;
    thread_local std::vector<float> gw0, gw1, gw2, gw3;
    if (t_dw != dw || t_sw != sw || t_xa != xa || t_n != n) {
        t_dw = dw; t_sw = sw; t_xa = xa; t_n = n;
        gi0.resize(n); gi1.resize(n); gi2.resize(n); gi3.resize(n);
        gw0.resize(n); gw1.resize(n); gw2.resize(n); gw3.resize(n);
        for (int i = 0; i < n; ++i) {
            const int x = clampi(xa + i, 0, dw - 1);
            const double u = (x + 0.5) * sx - 0.5;
            const int j = (int)std::floor(u);
            gi0[i] = clampi(j - 1, 0, sw - 1); gw0[i] = cubic_w((float)(u - (double)(j - 1)));
            gi1[i] = clampi(j, 0, sw - 1);     gw1[i] = cubic_w((float)(u - (double)j));
            gi2[i] = clampi(j + 1, 0, sw - 1); gw2[i] = cubic_w((float)(u - (double)(j + 1)));
            gi3[i] = clampi(j + 2, 0, sw - 1); gw3[i] = cubic_w((float)(u - (double)(j + 2)));
        }
    }
    const double v = (y + 0.5) * sy - 0.5;
    const int j = (int)std::floor(v);
    const float* r0 = src + (size_t)clampi(j - 1, 0, sh - 1) * sw;
    const float* r1 = src + (size_t)clampi(j, 0, sh - 1) * sw;
    const float* r2 = src + (size_t)clampi(j + 1, 0, sh - 1) * sw;
    const float* r3 = src + (size_t)clampi(j + 2, 0, sh - 1) * sw;
    const float wv0 = cubic_w((float)(v - (double)(j - 1)));
    const float wv1 = cubic_w((float)(v - (double)j));
    const float wv2 = cubic_w((float)(v - (double)(j + 1)));
    const float wv3 = cubic_w((float)(v - (double)(j + 2)));

    int i = 0;
    const int* p0 = gi0.data(), *p1 = gi1.data(), *p2 = gi2.data(), *p3 = gi3.data();
    const float* q0 = gw0.data(), *q1 = gw1.data(), *q2 = gw2.data(), *q3 = gw3.data();
    const __m256 vw0v = _mm256_set1_ps(wv0), vw1v = _mm256_set1_ps(wv1), vw2v = _mm256_set1_ps(wv2), vw3v = _mm256_set1_ps(wv3);
    for (; i + 8 <= n; i += 8) {
        __m256 acc = _mm256_setzero_ps();
        if (wv0 != 0.f) {
            __m256 h = _mm256_mul_ps(_mm256_i32gather_ps(r0, _mm256_loadu_si256((const __m256i*)(p0 + i)), 4), _mm256_loadu_ps(q0 + i));
            h = _mm256_fmadd_ps(_mm256_i32gather_ps(r0, _mm256_loadu_si256((const __m256i*)(p1 + i)), 4), _mm256_loadu_ps(q1 + i), h);
            h = _mm256_fmadd_ps(_mm256_i32gather_ps(r0, _mm256_loadu_si256((const __m256i*)(p2 + i)), 4), _mm256_loadu_ps(q2 + i), h);
            h = _mm256_fmadd_ps(_mm256_i32gather_ps(r0, _mm256_loadu_si256((const __m256i*)(p3 + i)), 4), _mm256_loadu_ps(q3 + i), h);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(h, vw0v));
        }
        if (wv1 != 0.f) {
            __m256 h = _mm256_mul_ps(_mm256_i32gather_ps(r1, _mm256_loadu_si256((const __m256i*)(p0 + i)), 4), _mm256_loadu_ps(q0 + i));
            h = _mm256_fmadd_ps(_mm256_i32gather_ps(r1, _mm256_loadu_si256((const __m256i*)(p1 + i)), 4), _mm256_loadu_ps(q1 + i), h);
            h = _mm256_fmadd_ps(_mm256_i32gather_ps(r1, _mm256_loadu_si256((const __m256i*)(p2 + i)), 4), _mm256_loadu_ps(q2 + i), h);
            h = _mm256_fmadd_ps(_mm256_i32gather_ps(r1, _mm256_loadu_si256((const __m256i*)(p3 + i)), 4), _mm256_loadu_ps(q3 + i), h);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(h, vw1v));
        }
        if (wv2 != 0.f) {
            __m256 h = _mm256_mul_ps(_mm256_i32gather_ps(r2, _mm256_loadu_si256((const __m256i*)(p0 + i)), 4), _mm256_loadu_ps(q0 + i));
            h = _mm256_fmadd_ps(_mm256_i32gather_ps(r2, _mm256_loadu_si256((const __m256i*)(p1 + i)), 4), _mm256_loadu_ps(q1 + i), h);
            h = _mm256_fmadd_ps(_mm256_i32gather_ps(r2, _mm256_loadu_si256((const __m256i*)(p2 + i)), 4), _mm256_loadu_ps(q2 + i), h);
            h = _mm256_fmadd_ps(_mm256_i32gather_ps(r2, _mm256_loadu_si256((const __m256i*)(p3 + i)), 4), _mm256_loadu_ps(q3 + i), h);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(h, vw2v));
        }
        if (wv3 != 0.f) {
            __m256 h = _mm256_mul_ps(_mm256_i32gather_ps(r3, _mm256_loadu_si256((const __m256i*)(p0 + i)), 4), _mm256_loadu_ps(q0 + i));
            h = _mm256_fmadd_ps(_mm256_i32gather_ps(r3, _mm256_loadu_si256((const __m256i*)(p1 + i)), 4), _mm256_loadu_ps(q1 + i), h);
            h = _mm256_fmadd_ps(_mm256_i32gather_ps(r3, _mm256_loadu_si256((const __m256i*)(p2 + i)), 4), _mm256_loadu_ps(q2 + i), h);
            h = _mm256_fmadd_ps(_mm256_i32gather_ps(r3, _mm256_loadu_si256((const __m256i*)(p3 + i)), 4), _mm256_loadu_ps(q3 + i), h);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(h, vw3v));
        }
        _mm256_storeu_ps(dst + i, acc);
    }
    for (; i < n; ++i) {
        const int x = clampi(xa + i, 0, dw - 1);
        const double u = (x + 0.5) * sx - 0.5;
        const int k = (int)std::floor(u);
        float acc = 0.f;
        for (int kk = -1; kk <= 2; ++kk) {
            const float wv = cubic_w((float)(v - (double)(j + kk)));
            if (wv == 0.f) continue;
            const float* rr = src + (size_t)clampi(j + kk, 0, sh - 1) * sw;
            float a = cubic_w((float)(u - (double)(k - 1))) * rr[clampi(k - 1, 0, sw - 1)];
            a = std::fmaf(cubic_w((float)(u - (double)(k))),     rr[clampi(k, 0, sw - 1)], a);
            a = std::fmaf(cubic_w((float)(u - (double)(k + 1))), rr[clampi(k + 1, 0, sw - 1)], a);
            a = std::fmaf(cubic_w((float)(u - (double)(k + 2))), rr[clampi(k + 2, 0, sw - 1)], a);
            acc += wv * a;
        }
        dst[i] = acc;
    }
}

// 三通道 1× 层流式化：ec1（↑ec8）与 up2c（↑d2，双三次）行内即时计算，d1 GF 直接写 out。
static void stream_channel_1x(const float* g_ref, const uint8_t* X8c, const uint8_t* M1,
                              const float* ec8, int Wb, int Hb, const float* d2,
                              float* out, int W, int H, int r1, int r2) {
    using namespace agf_detail;
    const float inv1 = 1.0f / (float)((2 * r1 + 1) * (2 * r1 + 1));
    const float inv2 = 1.0f / (float)((2 * r2 + 1) * (2 * r2 + 1));
    const __m256 vinv1 = _mm256_set1_ps(inv1), vinv2 = _mm256_set1_ps(inv2);
    const __m256 vzero = _mm256_setzero_ps(), v128 = _mm256_set1_ps(128.f), v255 = _mm256_set1_ps(255.f);

    par_cols(W, [&](int x0, int x1) {
        const int wstrip = x1 - x0;
        thread_local Fws w;
        w.ensure(wstrip, 1, wstrip, r1, r2);
        const int PAD = w.PAD, WP = w.WP, cD = w.cD, h1D = w.h1D, h2D = w.h2D;
        const int maxH1D = w.maxCD, maxH2D = w.maxCD;

        float* scrow = (float*)_aligned_malloc((size_t)WP * 4, 64);
        float* ecrow = (float*)_aligned_malloc((size_t)WP * 4, 64);

        for (int s = 0; s < 4; ++s) std::memset(w.v1 + (size_t)s * WP, 0, sizeof(float) * (size_t)WP);
        for (int s = 0; s < 2; ++s) std::memset(w.v2 + (size_t)s * WP, 0, sizeof(float) * (size_t)WP);

        int slot = 0, h1Slot = 0, h2Slot = 0;
        const int t0 = -(r1 + r2), tEnd = H + (r1 + r2);
        const int xBeg = PAD, xEnd = PAD + align8i(wstrip);
        int hx0 = (PAD - r2 - 8) & ~7; if (hx0 < r1 + 1) hx0 = align8i(r1 + 1);
        int hx1 = (PAD + align8i(wstrip) + r2 + 7) & ~7; if (hx1 + r1 > WP) hx1 = (WP - r1) & ~7;

        for (int t = t0; t < tEnd; ++t) {
            const int ty = clampi(t, 0, H - 1);
            // ---- 引导图 g_ref（中心化，读真实邻列）
            float* dst = w.cRing + (size_t)slot * WP;
            fill_tile(dst, g_ref + (size_t)ty * W, nullptr, nullptr, nullptr, nullptr, W, x0, align8i(wstrip), PAD, WP);
            for (int i = 0; i < WP; i += 8) _mm256_storeu_ps(dst + i, _mm256_sub_ps(_mm256_loadu_ps(dst + i), v128));
            const float* cRow = dst;
            // ---- p = blend(up2c, X8c, M1)（up2c 行内双三次，halo 取真实邻列）
            upsample_row_bilinear(d2, W / 2, H / 2, scrow, W, H, ty, x0 - PAD, x1 + PAD);
            for (int i = 0; i < WP; ++i) {
                const int col = clampi(x0 - PAD + i, 0, W - 1);
                w.prow[i] = (M1[(size_t)ty * W + col] ? (float)X8c[(size_t)ty * W + col] : scrow[i]);
            }
            slot = (slot + 1) % cD;

            const bool warm1 = (t - t0) >= h1D;
            float q1 = 0.f, q2 = 0.f;
            for (int j = -r1; j <= r1; ++j) { const float vv = cRow[hx0 - 1 + j]; q1 += vv; q2 += vv * vv; }
            float* hI  = w.h1 + (size_t)h1Slot * WP;
            float* hI2 = w.h1 + (size_t)(maxH1D + h1Slot) * WP;
            for (int x = hx0; x < hx1; x += 8) pair_step(cRow, cRow, x, r1, warm1, q1, q2, hI, hI2, w.v1, w.v1 + WP);
            float qa = 0.f, qb = 0.f;
            for (int j = -r1; j <= r1; ++j) { const float pv = w.prow[hx0 - 1 + j], vv = cRow[hx0 - 1 + j]; qa += pv; qb += pv * vv; }
            float* hP  = w.h1 + (size_t)(2 * maxH1D + h1Slot) * WP;
            float* hIP = w.h1 + (size_t)(3 * maxH1D + h1Slot) * WP;
            for (int x = hx0; x < hx1; x += 8) pair_step(w.prow, cRow, x, r1, warm1, qa, qb, hP, hIP, w.v1 + 2 * WP, w.v1 + 3 * WP);
            h1Slot = (h1Slot + 1) % h1D;

            const int u = t - r1;
            if (u >= 0) {
                float* ra = w.ra, *rb = w.rb;
                if (u < H) {
                    const float* vI = w.v1, *vI2 = w.v1 + WP, *vP = w.v1 + 2 * WP, *vIP = w.v1 + 3 * WP;
                    // eps = ec1（↑ec8），行 u（halo 取真实邻列）
                    upsample_row_bilinear(ec8, Wb, Hb, ecrow, W, H, u, x0 - PAD, x1 + PAD);
                    for (int x = hx0; x < hx1; x += 8) {
                        const __m256 mI = _mm256_mul_ps(_mm256_load_ps(vI + x), vinv1);
                        __m256 vv = _mm256_mul_ps(_mm256_load_ps(vI2 + x), vinv1);
                        const __m256 var = _mm256_max_ps(_mm256_sub_ps(vv, _mm256_mul_ps(mI, mI)), vzero);
                        const __m256 mp = _mm256_mul_ps(_mm256_load_ps(vP + x), vinv1);
                        const __m256 cv = _mm256_sub_ps(_mm256_mul_ps(_mm256_load_ps(vIP + x), vinv1), _mm256_mul_ps(mI, mp));
                        const __m256 den = _mm256_add_ps(var, _mm256_loadu_ps(ecrow + x));
                        const __m256 av = _mm256_div_ps(cv, den);
                        _mm256_storeu_ps(ra + x, av);
                        _mm256_storeu_ps(rb + x, _mm256_sub_ps(mp, _mm256_mul_ps(av, mI)));
                    }
                    { const float a0 = ra[PAD], b0 = rb[PAD];
                      for (int i = hx0; i < PAD; ++i) { ra[i] = a0; rb[i] = b0; } }
                    { const float a1 = ra[PAD + wstrip - 1], b1 = rb[PAD + wstrip - 1];
                      for (int i = PAD + wstrip; i < WP; ++i) { ra[i] = a1; rb[i] = b1; } }
                }
                const bool warm2 = u >= h2D;
                float qa2 = 0.f, qb2 = 0.f;
                for (int j = -r2; j <= r2; ++j) { qa2 += ra[PAD - 1 + j]; qb2 += rb[PAD - 1 + j]; }
                float* hA = w.h2 + (size_t)h2Slot * WP;
                float* hB = w.h2 + (size_t)(maxH2D + h2Slot) * WP;
                for (int x = xBeg; x < xEnd; x += 8) pair_step_ab(ra, rb, x, r2, warm2, qa2, qb2, hA, hB, w.v2, w.v2 + WP);
                h2Slot = (h2Slot + 1) % h2D;
                const int v = u - r2;
                if (v >= 0 && v < H) {
                    const int extra = (v < r2) ? (r2 - v) : 0;
                    const __m256 ve = _mm256_set1_ps((float)extra);
                    const float* ha0 = w.h2;
                    const float* hb0 = w.h2 + (size_t)maxH2D * WP;
                    const float* va = w.v2, *vb = w.v2 + WP;
                    const float* Iv = w.cRing + (size_t)slot * WP;
                    float* d = out + (size_t)v * W + x0;
                    int x = xBeg;
                    for (; x + 8 <= PAD + wstrip; x += 8) {
                        __m256 sa = _mm256_load_ps(va + x), sb = _mm256_load_ps(vb + x);
                        if (extra) { sa = _mm256_fmadd_ps(ve, _mm256_load_ps(ha0 + x), sa); sb = _mm256_fmadd_ps(ve, _mm256_load_ps(hb0 + x), sb); }
                        __m256 q = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(sa, vinv2), _mm256_load_ps(Iv + x)), _mm256_mul_ps(sb, vinv2));
                        q = _mm256_min_ps(_mm256_max_ps(q, vzero), v255);
                        _mm256_storeu_ps(d + (x - PAD), q);
                    }
                    for (; x < PAD + wstrip; ++x) {
                        float sa = va[x], sb = vb[x];
                        if (extra) { sa += (float)extra * ha0[x]; sb += (float)extra * hb0[x]; }
                        float q = sa * inv2 * Iv[x] + sb * inv2;
                        q = q < 0.f ? 0.f : (q > 255.f ? 255.f : q);
                        d[x - PAD] = q;
                    }
                }
            }
        }
        _aligned_free(scrow);
        _aligned_free(ecrow);
    });
}

// 引导图 1× 层 pass 1：cov(r=5) → t1 → 掩膜 GF(r=5) → 物化 M1。列条带并行（无跨条带读）。
static void stream_guide_m1(uint8_t* const* X8, const float* e8, int Wb, int Hb,
                            const float* b2f, const float* m2u, uint8_t* M1, int W, int H, float thr) {
    using namespace agf_detail;
    const int r = 5, r1 = 5, r2 = 5;
    const float inv1 = 1.0f / (float)((2 * r1 + 1) * (2 * r1 + 1));
    const float inv2 = 1.0f / (float)((2 * r2 + 1) * (2 * r2 + 1));
    const __m256 vinv1 = _mm256_set1_ps(inv1), vinv2 = _mm256_set1_ps(inv2);
    const __m256 vzero = _mm256_setzero_ps(), vthr = _mm256_set1_ps(thr);
    const __m256 v6 = _mm256_set1_ps(6.f), vone = _mm256_set1_ps(1.f);
    const int LAG = 32, LAGM = LAG - 1;   // 覆盖 20 行滞后
    const int TD = 2 * r + 1;             // t1 环深度 11
    const int t0 = -r, tEnd = H + 25;     // M1 最后一行 vv=H-1 在 t=H+24 产出

    par_cols(W, [&](int x0, int x1) {
        const int wstrip = x1 - x0;
        Cws& wC = get_cws(wstrip, r);
        thread_local Fws wM;
        wM.ensure(wstrip, 1, wstrip, r1, r2);
        const int PADC = wC.PAD, WPC = wC.WP, dC = wC.depth;
        const int PAD = wM.PAD, WP = wM.WP, cD = wM.cD, h1D = wM.h1D, h2D = wM.h2D;
        const int maxH1D = wM.maxCD, maxH2D = wM.maxCD;

        float* ring = (float*)_aligned_malloc((size_t)LAG * WP * 4 * 4, 64);
        float* t1Ring = (float*)_aligned_malloc((size_t)TD * WP * 4, 64);
        auto slot4 = [&](int s) { return ring + (size_t)(s & LAGM) * (size_t)(4 * WP); };

        for (int i = 0; i < 3; ++i) std::memset(wC.v[i], 0, sizeof(float) * (size_t)WPC);
        for (int s = 0; s < 4; ++s) std::memset(wM.v1 + (size_t)s * WP, 0, sizeof(float) * (size_t)WP);
        for (int s = 0; s < 2; ++s) std::memset(wM.v2 + (size_t)s * WP, 0, sizeof(float) * (size_t)WP);

        int cSlot = 0, mSlot = 0, mh1Slot = 0, mh2Slot = 0;
        const int xBeg = PAD, xEnd = PAD + align8i(wstrip);
        int hx0 = (PAD - r2 - 8) & ~7; if (hx0 < r1 + 1) hx0 = align8i(r1 + 1);
        int hx1 = (PAD + align8i(wstrip) + r2 + 7) & ~7; if (hx1 + r1 > WP) hx1 = (WP - r1) & ~7;
        const int covLo = hx0 - r1 - 1, covHi = hx1 + r1;   // t1 halo 范围

        for (int t = t0; t < tEnd; ++t) {
            const int ty = clampi(t, 0, H - 1);
            const int s = t & LAGM;
            float* gRow = slot4(s);
            float* uRow = slot4(s) + WP;
            float* eRow = slot4(s) + 2 * WP;
            float* mRow = slot4(s) + 3 * WP;
            for (int i = 0; i < WP; ++i) {
                const int col = clampi(x0 - PAD + i, 0, W - 1);
                const uint8_t* r0 = X8[0] + (size_t)ty * W + col;
                const uint8_t* r1 = X8[1] + (size_t)ty * W + col;
                const uint8_t* r2 = X8[2] + (size_t)ty * W + col;
                gRow[i] = ((float)r0[0] + 2.f * (float)r1[0] + (float)r2[0]) * 0.25f;
            }
            upsample_row_bilinear(b2f, W / 2, H / 2, uRow, W, H, ty, x0 - PAD, x1 + PAD);
            upsample_row_bilinear(e8, Wb, Hb, eRow, W, H, ty, x0 - PAD, x1 + PAD);
            upsample_row_bilinear(m2u, W / 2, H / 2, mRow, W, H, ty, x0 - PAD, x1 + PAD);

            // ===== 阶段 1：cov → t1（halo [covLo,covHi)）=====
            {
                for (int i = 0; i < WP; ++i) wC.pr[i] = gRow[i] - uRow[i];
                for (int i = 0; i < WP; ++i) wC.qr[i] = gRow[i];
                const bool warm = (t - t0) >= dC;
                float q0 = 0.f, q1 = 0.f, q2 = 0.f;
                for (int j = -r; j <= r; ++j) { const float a = wC.pr[covLo - 1 + j], b = wC.qr[covLo - 1 + j]; q0 += a; q1 += b; q2 += a * b; }
                float* h0 = wC.h[0] + (size_t)cSlot * WPC;
                float* h1 = wC.h[1] + (size_t)cSlot * WPC;
                float* h2 = wC.h[2] + (size_t)cSlot * WPC;
                for (int x = covLo; x < covHi; x += 8)
                    cov_step(wC.pr, wC.qr, x, r, warm, q0, q1, q2, h0, h1, h2, wC.v[0], wC.v[1], wC.v[2]);
                cSlot = (cSlot + 1) % dC;
                const int v = t - r;
                if (v >= 0 && v < H) {
                    const float* er = slot4(v) + 2 * WP + PAD;
                    float* d = t1Ring + (size_t)(v % TD) * WP;
                    const float* va = wC.v[0], *vb = wC.v[1], *vc = wC.v[2];
                    int x = covLo;
                    for (; x + 8 <= covHi; x += 8) {
                        const __m256 a = _mm256_load_ps(va + x);
                        const __m256 b = _mm256_load_ps(vb + x);
                        const __m256 cc = _mm256_load_ps(vc + x);
                        __m256 cov = _mm256_mul_ps(_mm256_sub_ps(cc, _mm256_mul_ps(_mm256_mul_ps(a, b), vinv1)), vinv1);
                        const __m256 ev = _mm256_loadu_ps(er + (x - PAD));
                        cov = _mm256_sub_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), cov), _mm256_mul_ps(_mm256_mul_ps(ev, ev), v6));
                        cov = _mm256_min_ps(_mm256_max_ps(cov, vzero), vone);
                        _mm256_storeu_ps(d + x, cov);
                    }
                    for (; x < covHi; ++x) {
                        float cov = (vc[x] - va[x] * vb[x] * inv1) * inv1;
                        const float e = er[x - PAD];
                        cov = std::fabs(cov) - 6.f * e * e;
                        d[x] = cov < 0.f ? 0.f : (cov > 1.f ? 1.f : cov);
                    }
                }
            }

            // ===== 阶段 2：掩膜 GF → M1（行 t-25）=====
            {
                const int u = t - (r + r1 + r);
                if (u >= -(r1 + r2) && u < H + (r1 + r2)) {
                    const float* mRowU = slot4(u) + 3 * WP;
                    const float* t1Row = t1Ring + (size_t)(clampi(u, 0, H - 1) % TD) * WP;
                    float* dst = wM.cRing + (size_t)mSlot * WP;
                    std::memcpy(dst, mRowU, sizeof(float) * (size_t)WP);
                    std::memcpy(wM.prow, t1Row, sizeof(float) * (size_t)WP);
                    const float* cRow = dst;
                    const bool warm1 = (u + (r1 + r2)) >= h1D;
                    float q1 = 0.f, q2 = 0.f;
                    for (int j = -r1; j <= r1; ++j) { const float vv = cRow[hx0 - 1 + j]; q1 += vv; q2 += vv * vv; }
                    float* hI  = wM.h1 + (size_t)mh1Slot * WP;
                    float* hI2 = wM.h1 + (size_t)(maxH1D + mh1Slot) * WP;
                    for (int x = hx0; x < hx1; x += 8) pair_step(cRow, cRow, x, r1, warm1, q1, q2, hI, hI2, wM.v1, wM.v1 + WP);
                    float qa = 0.f, qb = 0.f;
                    for (int j = -r1; j <= r1; ++j) { const float pv = wM.prow[hx0 - 1 + j], vv = cRow[hx0 - 1 + j]; qa += pv; qb += pv * vv; }
                    float* hP  = wM.h1 + (size_t)(2 * maxH1D + mh1Slot) * WP;
                    float* hIP = wM.h1 + (size_t)(3 * maxH1D + mh1Slot) * WP;
                    for (int x = hx0; x < hx1; x += 8) pair_step(wM.prow, cRow, x, r1, warm1, qa, qb, hP, hIP, wM.v1 + 2 * WP, wM.v1 + 3 * WP);
                    mh1Slot = (mh1Slot + 1) % h1D;
                    const int uu = u - r1;
                    if (uu >= 0) {
                        float* ra = wM.ra, *rb = wM.rb;
                        if (uu < H) {
                            const float* vI = wM.v1, *vI2 = wM.v1 + WP, *vP = wM.v1 + 2 * WP, *vIP = wM.v1 + 3 * WP;
                            const float* epRow = slot4(uu) + 2 * WP + PAD;
                            for (int x = hx0; x < hx1; x += 8) {
                                const __m256 mI = _mm256_mul_ps(_mm256_load_ps(vI + x), vinv1);
                                __m256 vv = _mm256_mul_ps(_mm256_load_ps(vI2 + x), vinv1);
                                const __m256 var = _mm256_max_ps(_mm256_sub_ps(vv, _mm256_mul_ps(mI, mI)), vzero);
                                const __m256 mp = _mm256_mul_ps(_mm256_load_ps(vP + x), vinv1);
                                const __m256 cv = _mm256_sub_ps(_mm256_mul_ps(_mm256_load_ps(vIP + x), vinv1), _mm256_mul_ps(mI, mp));
                                const __m256 den = _mm256_add_ps(var, _mm256_loadu_ps(epRow + clampi(x - PAD, 0, wstrip - 1)));
                                const __m256 av = _mm256_div_ps(cv, den);
                                _mm256_storeu_ps(ra + x, av);
                                _mm256_storeu_ps(rb + x, _mm256_sub_ps(mp, _mm256_mul_ps(av, mI)));
                            }
                            { const float a0 = ra[PAD], b0 = rb[PAD];
                              for (int i = hx0; i < PAD; ++i) { ra[i] = a0; rb[i] = b0; } }
                            { const float a1 = ra[PAD + wstrip - 1], b1 = rb[PAD + wstrip - 1];
                              for (int i = PAD + wstrip; i < WP; ++i) { ra[i] = a1; rb[i] = b1; } }
                        }
                        const bool warm2 = uu >= h2D;
                        float qa2 = 0.f, qb2 = 0.f;
                        for (int j = -r2; j <= r2; ++j) { qa2 += ra[PAD - 1 + j]; qb2 += rb[PAD - 1 + j]; }
                        float* hA = wM.h2 + (size_t)mh2Slot * WP;
                        float* hB = wM.h2 + (size_t)(maxH2D + mh2Slot) * WP;
                        for (int x = xBeg; x < xEnd; x += 8) pair_step_ab(ra, rb, x, r2, warm2, qa2, qb2, hA, hB, wM.v2, wM.v2 + WP);
                        mh2Slot = (mh2Slot + 1) % h2D;
                        const int vv = uu - r2;
                        if (vv >= 0 && vv < H) {
                            const float* va = wM.v2, *vb = wM.v2 + WP;
                            const float* Iv = wM.cRing + (size_t)((mSlot + 1) % cD) * WP;
                            uint8_t* mo = M1 + (size_t)vv * W + x0;
                            int x = xBeg;
                            for (; x + 8 <= PAD + wstrip; x += 8) {
                                const __m256 sa = _mm256_load_ps(va + x), sb = _mm256_load_ps(vb + x);
                                const __m256 q = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(sa, vinv2), _mm256_load_ps(Iv + x)), _mm256_mul_ps(sb, vinv2));
                                const __m256i m32 = _mm256_castps_si256(_mm256_cmp_ps(q, vthr, _CMP_GT_OQ));
                                const __m256i m16 = _mm256_packs_epi32(m32, m32);
                                const __m128i m8 = _mm_packs_epi16(_mm256_castsi256_si128(m16), _mm256_castsi256_si128(m16));
                                _mm_storel_epi64((__m128i*)(mo + (x - PAD)), _mm_and_si128(m8, _mm_set1_epi8(1)));
                            }
                            for (; x < PAD + wstrip; ++x) { const float q = va[x] * inv2 * Iv[x] + vb[x] * inv2; mo[x - PAD] = (q > thr) ? 1 : 0; }
                        }
                    }
                    mSlot = (mSlot + 1) % cD;
                }
            }
        }
        _aligned_free(ring);
        _aligned_free(t1Ring);
    });
}

// 引导图 1× 层 pass 2：g_ref GF（读物化 M1 + 重算 g/up1/e1）。列条带并行。
static void stream_guide_gref(uint8_t* const* X8, const float* e8, int Wb, int Hb,
                              const float* b2f, const uint8_t* M1, float* g_ref, int W, int H) {
    using namespace agf_detail;
    const int r1 = 5, r2 = 5;
    const float inv1 = 1.0f / (float)((2 * r1 + 1) * (2 * r1 + 1));
    const float inv2 = 1.0f / (float)((2 * r2 + 1) * (2 * r2 + 1));
    const __m256 vinv1 = _mm256_set1_ps(inv1), vinv2 = _mm256_set1_ps(inv2);
    const __m256 vzero = _mm256_setzero_ps(), v128 = _mm256_set1_ps(128.f);
    const int LAG = 8, LAGM = LAG - 1;    // e1 滞后 r1=5
    const int t0 = -(r1 + r2), tEnd = H + (r1 + r2);

    par_cols(W, [&](int x0, int x1) {
        const int wstrip = x1 - x0;
        thread_local Fws wG;
        wG.ensure(wstrip, 1, wstrip, r1, r2);
        const int PAD = wG.PAD, WP = wG.WP, cD = wG.cD, h1D = wG.h1D, h2D = wG.h2D;
        const int maxH1D = wG.maxCD, maxH2D = wG.maxCD;

        float* ring = (float*)_aligned_malloc((size_t)LAG * WP * 3 * 4, 64);   // [g][up1][e1]
        auto slot3 = [&](int s) { return ring + (size_t)(s & LAGM) * (size_t)(3 * WP); };

        for (int s = 0; s < 4; ++s) std::memset(wG.v1 + (size_t)s * WP, 0, sizeof(float) * (size_t)WP);
        for (int s = 0; s < 2; ++s) std::memset(wG.v2 + (size_t)s * WP, 0, sizeof(float) * (size_t)WP);

        int gSlot = 0, gh1Slot = 0, gh2Slot = 0;
        const int xBeg = PAD, xEnd = PAD + align8i(wstrip);
        int hx0 = (PAD - r2 - 8) & ~7; if (hx0 < r1 + 1) hx0 = align8i(r1 + 1);
        int hx1 = (PAD + align8i(wstrip) + r2 + 7) & ~7; if (hx1 + r1 > WP) hx1 = (WP - r1) & ~7;

        for (int t = t0; t < tEnd; ++t) {
            const int ty = clampi(t, 0, H - 1);
            const int s = t & LAGM;
            float* gRow = slot3(s);
            float* uRow = slot3(s) + WP;
            float* eRow = slot3(s) + 2 * WP;
            for (int i = 0; i < WP; ++i) {
                const int col = clampi(x0 - PAD + i, 0, W - 1);
                const uint8_t* r0 = X8[0] + (size_t)ty * W + col;
                const uint8_t* r1 = X8[1] + (size_t)ty * W + col;
                const uint8_t* r2 = X8[2] + (size_t)ty * W + col;
                gRow[i] = ((float)r0[0] + 2.f * (float)r1[0] + (float)r2[0]) * 0.25f;
            }
            upsample_row_bilinear(b2f, W / 2, H / 2, uRow, W, H, ty, x0 - PAD, x1 + PAD);
            upsample_row_bilinear(e8, Wb, Hb, eRow, W, H, ty, x0 - PAD, x1 + PAD);

            float* dst = wG.cRing + (size_t)gSlot * WP;
            for (int i = 0; i < WP; ++i) {
                const int col = clampi(x0 - PAD + i, 0, W - 1);
                const uint8_t m = M1[(size_t)ty * W + col];
                dst[i] = (m ? gRow[i] : uRow[i]) - 128.f;
            }
            std::memcpy(wG.prow, gRow, sizeof(float) * (size_t)WP);
            const float* cRow = dst;
            gSlot = (gSlot + 1) % cD;

            const bool warm1 = (t - t0) >= h1D;
            float q1 = 0.f, q2 = 0.f;
            for (int j = -r1; j <= r1; ++j) { const float vv = cRow[hx0 - 1 + j]; q1 += vv; q2 += vv * vv; }
            float* hI  = wG.h1 + (size_t)gh1Slot * WP;
            float* hI2 = wG.h1 + (size_t)(maxH1D + gh1Slot) * WP;
            for (int x = hx0; x < hx1; x += 8) pair_step(cRow, cRow, x, r1, warm1, q1, q2, hI, hI2, wG.v1, wG.v1 + WP);
            float qa = 0.f, qb = 0.f;
            for (int j = -r1; j <= r1; ++j) { const float pv = wG.prow[hx0 - 1 + j], vv = cRow[hx0 - 1 + j]; qa += pv; qb += pv * vv; }
            float* hP  = wG.h1 + (size_t)(2 * maxH1D + gh1Slot) * WP;
            float* hIP = wG.h1 + (size_t)(3 * maxH1D + gh1Slot) * WP;
            for (int x = hx0; x < hx1; x += 8) pair_step(wG.prow, cRow, x, r1, warm1, qa, qb, hP, hIP, wG.v1 + 2 * WP, wG.v1 + 3 * WP);
            gh1Slot = (gh1Slot + 1) % h1D;

            const int u = t - r1;
            if (u >= 0) {
                float* ra = wG.ra, *rb = wG.rb;
                if (u < H) {
                    const float* vI = wG.v1, *vI2 = wG.v1 + WP, *vP = wG.v1 + 2 * WP, *vIP = wG.v1 + 3 * WP;
                    const float* epRow = slot3(u) + 2 * WP + PAD;
                    for (int x = hx0; x < hx1; x += 8) {
                        const __m256 mI = _mm256_mul_ps(_mm256_load_ps(vI + x), vinv1);
                        __m256 vv = _mm256_mul_ps(_mm256_load_ps(vI2 + x), vinv1);
                        const __m256 var = _mm256_max_ps(_mm256_sub_ps(vv, _mm256_mul_ps(mI, mI)), vzero);
                        const __m256 mp = _mm256_mul_ps(_mm256_load_ps(vP + x), vinv1);
                        const __m256 cv = _mm256_sub_ps(_mm256_mul_ps(_mm256_load_ps(vIP + x), vinv1), _mm256_mul_ps(mI, mp));
                        const __m256 den = _mm256_add_ps(var, _mm256_loadu_ps(epRow + clampi(x - PAD, 0, wstrip - 1)));
                        const __m256 av = _mm256_div_ps(cv, den);
                        _mm256_storeu_ps(ra + x, av);
                        _mm256_storeu_ps(rb + x, _mm256_sub_ps(mp, _mm256_mul_ps(av, mI)));
                    }
                    { const float a0 = ra[PAD], b0 = rb[PAD];
                      for (int i = hx0; i < PAD; ++i) { ra[i] = a0; rb[i] = b0; } }
                    { const float a1 = ra[PAD + wstrip - 1], b1 = rb[PAD + wstrip - 1];
                      for (int i = PAD + wstrip; i < WP; ++i) { ra[i] = a1; rb[i] = b1; } }
                }
                const bool warm2 = u >= h2D;
                float qa2 = 0.f, qb2 = 0.f;
                for (int j = -r2; j <= r2; ++j) { qa2 += ra[PAD - 1 + j]; qb2 += rb[PAD - 1 + j]; }
                float* hA = wG.h2 + (size_t)gh2Slot * WP;
                float* hB = wG.h2 + (size_t)(maxH2D + gh2Slot) * WP;
                for (int x = xBeg; x < xEnd; x += 8) pair_step_ab(ra, rb, x, r2, warm2, qa2, qb2, hA, hB, wG.v2, wG.v2 + WP);
                gh2Slot = (gh2Slot + 1) % h2D;
                const int v = u - r2;
                if (v >= 0 && v < H) {
                    const int extra = (v < r2) ? (r2 - v) : 0;
                    const __m256 ve = _mm256_set1_ps((float)extra);
                    const float* ha0 = wG.h2;
                    const float* hb0 = wG.h2 + (size_t)maxH2D * WP;
                    const float* va = wG.v2, *vb = wG.v2 + WP;
                    const float* Iv = wG.cRing + (size_t)gSlot * WP;
                    float* d = g_ref + (size_t)v * W + x0;
                    int x = xBeg;
                    for (; x + 8 <= PAD + wstrip; x += 8) {
                        __m256 sa = _mm256_load_ps(va + x), sb = _mm256_load_ps(vb + x);
                        if (extra) { sa = _mm256_fmadd_ps(ve, _mm256_load_ps(ha0 + x), sa); sb = _mm256_fmadd_ps(ve, _mm256_load_ps(hb0 + x), sb); }
                        const __m256 q = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(sa, vinv2), _mm256_load_ps(Iv + x)), _mm256_mul_ps(sb, vinv2));
                        _mm256_storeu_ps(d + (x - PAD), q);
                    }
                    for (; x < PAD + wstrip; ++x) {
                        float sa = va[x], sb = vb[x];
                        if (extra) { sa += (float)extra * ha0[x]; sb += (float)extra * hb0[x]; }
                        d[x - PAD] = sa * inv2 * Iv[x] + sb * inv2;
                    }
                }
            }
        }
        _aligned_free(ring);
    });
}

// =============================================================================
static void denoise_core(const uint8_t* rgb, float* out3, int W, int H, float thr, bool verbose) {
    using namespace agf_detail;
    const size_t N = (size_t)W * H;
    const int Hb = H / 8, Wb = W / 8;
    const size_t NB = (size_t)Hb * Wb;
    static thread_local Pool pool;   // 跨图复用，避免每张图重新 malloc ~1 GB (page fault)
    static thread_local U8Pool u8;
    const int nthr = g_nthreads;    // 由外层（agf_denoise / tiled）预先设定
    auto T0 = std::chrono::steady_clock::now();
    double tNoise = 0, tGuide = 0, tChan = 0;
    double tEps = 0, tGF4 = 0, tGF2 = 0, tGF1 = 0, tBit = 0, tUpG = 0;
    auto tick = [&](double& acc) {
        const auto t1 = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double, std::milli>(t1 - T0).count();
        T0 = t1;
    };

    // ---- 0. 输入（★ X 不物化：p 就是原始 u8，k_c=255/b_c=0）
    uint8_t* X8[3];
    for (int c = 0; c < 3; ++c) {
        X8[c] = u8.get(10 + c, N);
        uint8_t* d = X8[c];
        const uint8_t* s = rgb + c;
        par_n(N, [&](int, size_t a, size_t b) {
            for (size_t i = a; i < b; ++i) d[i] = s[i * 3];
        });
    }
    // ---- 1. 块噪声
    float* V[3];
    double meanV[3];
    for (int c = 0; c < 3; ++c) {
        V[c] = pool.get(110 + c, NB);
        block_noise_u8(X8[c], V[c], W, H);
        double s = 0;
        for (size_t i = 0; i < NB; ++i) { s += V[c][i]; }
        meanV[c] = s / (double)NB;
    }

    tick(tNoise);
    // ---- 2. 引导图
    // 对齐 adapt_guided2：g = sum(X_c * [1,2,1]) / 4，不拉伸（k_a=255, b_a=0）
    float* g = pool.get(120, N);
    par_n(N, [&](int, size_t a, size_t b) {
        for (size_t i = a; i < b; ++i)
            g[i] = ((float)X8[0][i] + 2.f * (float)X8[1][i] + (float)X8[2][i]) * 0.25f;
    });
    // ---- 3. 引导图噪声图
    float* Vg = pool.get(121, NB);
    block_noise(g, Vg, W, H);

    // ---- 4. 金字塔
    float* low2 = pool.get(122, (size_t)(W / 2) * (H / 2));
    float* low4 = pool.get(123, (size_t)(W / 4) * (H / 4));
    float* low8 = pool.get(124, NB);
    downsample2(g, low2, W, H);
    downsample2(low2, low4, W / 2, H / 2);
    downsample2(low4, low8, W / 4, H / 4);

    // ---- 5. eps（引导图）
    float* e8 = pool.get(125, NB);
    par_n(NB, [&](int, size_t a, size_t b) { for (size_t i = a; i < b; ++i) e8[i] = std::sqrt(Vg[i]); });
    float* e4 = pool.get(126, (size_t)(W / 4) * (H / 4));
    float* e2 = pool.get(127, (size_t)(W / 2) * (H / 2));
    uint8_t* e1u = u8.get(20, N);   // eps 1× 改 u8（×16 缩放）
    upsample(e8, Wb, Hb, e4, W / 4, H / 4, false);
    upsample(e8, Wb, Hb, e2, W / 2, H / 2, false);
    upsample_u8(e8, Wb, Hb, e1u, W, H, 16.f);

    uint8_t* M8 = u8.get(0, NB);
    uint8_t* M4 = u8.get(1, (size_t)(W / 4) * (H / 4));
    uint8_t* M2 = u8.get(2, (size_t)(W / 2) * (H / 2));
    uint8_t* M1 = u8.get(3, N);

    // ---- 6. 1/8 层
    float* b8 = pool.get(131, NB);
    float* tmp8 = pool.get(132, NB);
{ const auto b0 = std::chrono::steady_clock::now();
  bitonic(low8, b8, Wb, Hb);
  tBit += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-b0).count(); }
    {
        cov_box(low8, low8, tmp8, Wb, Hb, 3, b8, e8, 1.f, 10.f);
        hard_mask_plane(tmp8, M8, NB, thr);
        GfArgs A;
        A.guide = b8; A.guideAlt = low8; A.guideMask = M8; A.p[0] = low8; A.out[0] = tmp8;
        A.eps[0] = e8; A.NC = 1; A.r1 = A.r2 = 3;
        gf_call(A, Wb, Hb, pool, nthr);
        std::memcpy(b8, tmp8, sizeof(float) * NB);
    }

    // ---- 7. 1/4 层
    float* b4 = pool.get(136, (size_t)(W / 4) * (H / 4));
    {
        const int w4 = W / 4, h4 = H / 4;
        const size_t N4 = (size_t)w4 * h4;
        float* up4 = pool.get(135, N4);
        upsample(b8, Wb, Hb, up4, w4, h4, false);
        float* m8u = pool.get(138, NB);
        par_n(NB, [&](int, size_t a, size_t b) { for (size_t i = a; i < b; ++i) m8u[i] = (float)M8[i]; });
        float* m8up = pool.get(139, N4);
        upsample(m8u, Wb, Hb, m8up, w4, h4, false);
        float* t4 = pool.get(141, N4);
        cov_box(low4, low4, t4, w4, h4, 3, up4, e4, 1.f, 10.f);
        float* mout4 = pool.get(140, N4);
        GfArgs A;
        A.guide = m8up; A.p[0] = t4; A.out[0] = mout4;
        A.eps[0] = e4; A.NC = 1; A.r1 = A.r2 = 5; A.centerGuide = false;
        gf_call(A, w4, h4, pool, nthr);
        hard_mask_plane(mout4, M4, N4, thr);
        GfArgs B;
        B.guide = up4; B.guideAlt = low4; B.guideMask = M4; B.p[0] = low4; B.out[0] = b4;
        B.eps[0] = e4; B.NC = 1; B.r1 = B.r2 = 3;
        gf_call(B, w4, h4, pool, nthr);
    }

    // ---- 8. 1/2 层
    float* b2 = pool.get(150, (size_t)(W / 2) * (H / 2));
    float* b2f = pool.get(151, (size_t)(W / 2) * (H / 2));
    float sig_s2 = 0.f;
    int ar1 = 3, ar2 = 2;
    {
        const int w2 = W / 2, h2 = H / 2;
        const size_t N2 = (size_t)w2 * h2;
        float* up2 = pool.get(152, N2);
        upsample(b4, W / 4, H / 4, up2, w2, h2, false);
        float* m4u = pool.get(153, (size_t)(W / 4) * (H / 4));
        for (size_t i = 0; i < (size_t)(W / 4) * (H / 4); ++i) m4u[i] = (float)M4[i];
        float* m4up = pool.get(154, N2);
        upsample(m4u, W / 4, H / 4, m4up, w2, h2, false);
        float* t2 = pool.get(156, N2);
        cov_box(low2, low2, t2, w2, h2, 5, up2, e2, 1.f, 1.f);
        float* mout2 = pool.get(155, N2);
        GfArgs A;
        A.guide = m4up; A.p[0] = t2; A.out[0] = mout2;
        A.eps[0] = e2; A.NC = 1; A.r1 = A.r2 = 5; A.centerGuide = false;
        gf_call(A, w2, h2, pool, nthr);
        hard_mask_plane(mout2, M2, N2, thr);

        GfArgs B;
        B.guide = up2; B.guideAlt = low2; B.guideMask = M2; B.p[0] = low2; B.out[0] = b2;
        B.eps[0] = e2; B.NC = 1; B.r1 = B.r2 = 3;
        gf_call(B, w2, h2, pool, nthr);

        sig_s2 = 3.f;   // 对齐 adapt_guided2：固定 sigma_s2 = 3（原 var 计算已不需要）
        ar1 = 1 + 2 * (int)std::ceil(sig_s2 / 2.f);
        ar2 = 2 * (int)std::floor(sig_s2 / 2.f);

        GfArgs C;
        C.guide = b2; C.p[0] = low2; C.out[0] = b2f;
        C.eps[0] = e2; C.NC = 1; C.r1 = C.r2 = ar1;
        gf_call(C, w2, h2, pool, nthr);
    }

    // ---- 9. 1x 层（引导图）
    float* g_ref = pool.get(160, N);
    if (g_stream) {
        float* m2u = pool.get(162, (size_t)(W / 2) * (H / 2));
        for (size_t i = 0; i < (size_t)(W / 2) * (H / 2); ++i) m2u[i] = (float)M2[i];
        stream_guide_m1(X8, e8, Wb, Hb, b2f, m2u, M1, W, H, thr);
        stream_guide_gref(X8, e8, Wb, Hb, b2f, M1, g_ref, W, H);
    } else {
        uint8_t* up1u = u8.get(21, N);   // ↑b2f 改 u8
        upsample_u8(b2f, W / 2, H / 2, up1u, W, H);
        float* m2u = pool.get(162, (size_t)(W / 2) * (H / 2));
        for (size_t i = 0; i < (size_t)(W / 2) * (H / 2); ++i) m2u[i] = (float)M2[i];
        uint8_t* m2upu = u8.get(22, N);   // ↑M2 改 u8
        upsample_u8(m2u, W / 2, H / 2, m2upu, W, H);
        float* t1 = pool.get(165, N);
        cov_box(g, g, t1, W, H, 5, nullptr, nullptr, 6.f, 1.f, up1u, e1u, 1.f / 16.f);
        GfArgs A;
        A.guideU8 = m2upu; A.p[0] = t1;
        A.epsU8[0] = e1u; A.epsScale = 1.f / 16.f; A.NC = 1; A.r1 = A.r2 = 5; A.centerGuide = false;
        A.outMask = M1; A.maskThr = thr;
        gf_call(A, W, H, pool, nthr);
        GfArgs B;
        B.guideU8 = up1u; B.guideAlt = g; B.guideMask = M1; B.p[0] = g; B.out[0] = g_ref;
        B.epsU8[0] = e1u; B.epsScale = 1.f / 16.f; B.NC = 1; B.r1 = B.r2 = ar1;
        gf_call(B, W, H, pool, nthr);
    }
    float* g_ref2 = pool.get(167, (size_t)(W / 2) * (H / 2));
    float* g_ref4 = pool.get(168, (size_t)(W / 4) * (H / 4));
    downsample2(g_ref, g_ref2, W, H);
    downsample2(g_ref2, g_ref4, W / 2, H / 2);

    tick(tGuide);
    // ---- 10. 逐通道重建
    for (int c = 0; c < 3; ++c) {
        // ★ 对齐 adapt_guided2：k_a=255, b_a=0 → y 就是 X，不需要 y 平面
        const float kc = 255.f, mnc0 = 0.f;
        
        float* ec8 = pool.get(171, NB);
        par_n(NB, [&](int, size_t a, size_t b) { for (size_t i = a; i < b; ++i) ec8[i] = std::sqrt(V[c][i]); });
        float* ec4 = pool.get(172, (size_t)(W / 4) * (H / 4));
        float* ec2 = pool.get(173, (size_t)(W / 2) * (H / 2));
        uint8_t* ec1u = u8.get(23, N);   // 通道 ε 1× 改 u8（×16 缩放）
        upsample(ec8, Wb, Hb, ec4, W / 4, H / 4, false);
        upsample(ec8, Wb, Hb, ec2, W / 2, H / 2, false);
        upsample_u8(ec8, Wb, Hb, ec1u, W, H, 16.f);
        tick(tEps);
        float* y2 = pool.get(175, (size_t)(W / 2) * (H / 2));
        float* y4 = pool.get(176, (size_t)(W / 4) * (H / 4));
        downsample2_u8f(X8[c], y2, W, H);
        downsample2(y2, y4, W / 2, H / 2);

        float* d4 = pool.get(177, (size_t)(W / 4) * (H / 4));
        {
            const int w4 = W / 4, h4 = H / 4;
            const size_t N4 = (size_t)w4 * h4;
            GfArgs A;
            A.guide = g_ref4; A.p[0] = y4; A.out[0] = d4;
            A.eps[0] = ec4; A.epsScale = 0.25f; A.NC = 1; A.r1 = A.r2 = 5;
            gf_call(A, w4, h4, pool, nthr);
            float* res = pool.get(178, N4);
            float* corr = pool.get(179, N4);
            par_n(N4, [&](int, size_t a, size_t b) { for (size_t i = a; i < b; ++i) res[i] = y4[i] - d4[i]; });
            GfArgs B;
            B.guide = g_ref4; B.p[0] = res; B.out[0] = corr;
            B.eps[0] = ec4; B.epsScale = 1.f; B.NC = 1; B.r1 = B.r2 = 5;
            gf_call(B, w4, h4, pool, nthr);
            par_n(N4, [&](int, size_t a, size_t b) { for (size_t i = a; i < b; ++i) d4[i] += corr[i] / (1.f + ec4[i] * 0.0625f); });
        }
tick(tGF4);
float* d2 = pool.get(180, (size_t)(W / 2) * (H / 2));
        {
            const int w2 = W / 2, h2 = H / 2;
            const size_t N2 = (size_t)w2 * h2;
            float* up4c = pool.get(181, N2);
            upsample(d4, W / 4, H / 4, up4c, w2, h2, true);
            GfArgs A;
            A.guide = g_ref2; A.p[0] = up4c; A.pAlt[0] = y2; A.mask = M2;
            A.out[0] = d2; A.eps[0] = ec2; A.epsScale = 0.5f; A.NC = 1;
            A.r1 = A.r2 = 5 + ar2;
            gf_call(A, w2, h2, pool, nthr);
            float* res = pool.get(183, N2);
            float* corr = pool.get(184, N2);
            par_n(N2, [&](int, size_t a, size_t b) { for (size_t i = a; i < b; ++i) res[i] = y2[i] - d2[i]; });
            GfArgs B;
            B.guide = g_ref2; B.p[0] = res; B.out[0] = corr;
            B.eps[0] = ec2; B.epsScale = 1.f; B.NC = 1; B.r1 = B.r2 = ar2 + 3;
            gf_call(B, w2, h2, pool, nthr);
            par_n(N2, [&](int, size_t a, size_t b) { for (size_t i = a; i < b; ++i) d2[i] += corr[i] / (1.f + ec2[i] * 0.25f); });
        }
        {
            tick(tGF2);
            if (g_stream) {
                stream_channel_1x(g_ref, X8[c], M1, ec8, Wb, Hb, d2,
                                  out3 + (size_t)c * N, W, H, 3 + ar2, 3 + ar2);
            } else {
                uint8_t* up2cu = u8.get(24, N);   // ↑d2 改 u8
                upsample_u8(d2, W / 2, H / 2, up2cu, W, H);
                GfArgs A;
                A.guide = g_ref; A.pU8[0] = up2cu; A.pAltU8[0] = X8[c]; A.mask = M1;
                A.out[0] = out3 + (size_t)c * N;
                A.epsU8[0] = ec1u; A.epsScale = 1.f / 16.f; A.NC = 1;
                A.r1 = A.r2 = 3 + ar2; A.clampOut = true;
                gf_call(A, W, H, pool, nthr);
            }
        }
        tick(tGF1);
    }
    tick(tChan);
    if (verbose) {
        std::printf("  [agf] phase: noise %.0f, guide %.0f, channel %.0f (eps %.0f / gf4 %.0f / gf2 %.0f / gf1 %.0f) | bitonic %.0f up1 %.0f\n",
                    tNoise, tGuide, tEps + tGF4 + tGF2 + tGF1, tEps, tGF4, tGF2, tGF1, tBit, tUpG);
        double gm = 0, gs = 0;
        for (size_t i = 0; i < N; ++i) { gm += g[i]; gs += (double)g[i] * g[i]; }
        gm /= (double)N; gs = std::sqrt(gs / (double)N - gm * gm);
        std::printf("  [agf] meanV = %.3f %.3f %.3f   g: mean %.2f std %.2f\n",
                    meanV[0], meanV[1], meanV[2], gm, gs);
        auto frac = [](const uint8_t* m, size_t n) {
            size_t k = 0; for (size_t i = 0; i < n; ++i) k += m[i] ? 1 : 0;
            return 100.0 * (double)k / (double)n;
        };
        std::printf("  [agf] mask%% = M8 %.1f  M4 %.1f  M2 %.1f  M1 %.1f\n",
                    frac(M8, NB), frac(M4, (size_t)(W / 4) * (H / 4)),
                    frac(M2, (size_t)(W / 2) * (H / 2)), frac(M1, N));
        double em = 0; for (size_t i = 0; i < N; ++i) em += (float)e1u[i] * (1.f / 16.f); em /= (double)N;
        std::printf("  [agf] e1 mean = %.3f\n", em);
        std::printf("  [agf] sigma_s2=%.3f ar1=%d ar2=%d\n", sig_s2, ar1, ar2);
    }
}

// =============================================================================
void agf_denoise(const uint8_t* rgb, float* out3, int W, int H, const AgfParams& prm) {
    g_fused = prm.fused;
    g_nthreads = prm.nthreads > 0 ? prm.nthreads : 1;
    g_gf_tile = prm.tile;
    g_stream = prm.stream;
    denoise_core(rgb, out3, W, H, prm.maskThr, prm.verbose);
}

// =============================================================================
//  分块滑窗：整图切 tile，每 tile 读一次/写一次 interior，中间量留缓存
//  ---------------------------------------------------------------------------
//  正确性前提：HALO 覆盖整条管线（噪声块 → 引导金字塔 → 掩膜链 → g_ref → 三通道）
//  在空间上的最大影响半径；tile 原点和 HALO 都是 8 的倍数，保证 8×8 块 / 金字塔对齐。
// =============================================================================
struct TileCtx {
    std::vector<uint8_t> sub;
    std::vector<float> out;
};

static void denoise_tile(const uint8_t* rgb, int W, int H, float* out3,
                         int tx, int ty, int TW, int TH, int HALO, float thr) {
    const size_t N = (size_t)W * H;
    const int ox0 = tx * TW, oy0 = ty * TH;
    const int ow = std::min(TW, W - ox0);
    const int oh = std::min(TH, H - oy0);
    const int ix0 = std::max(0, ox0 - HALO);
    const int iy0 = std::max(0, oy0 - HALO);
    const int ix1 = std::min(W, ox0 + ow + HALO);
    const int iy1 = std::min(H, oy0 + oh + HALO);
    const int subW = ix1 - ix0, subH = iy1 - iy0;
    // 关键：把 sub-image 补到 32 的倍数（复制边缘），规避原语里对“非 32 倍数宽度”的越界假设。
    const int PW = align32i(subW), PH = align32i(subH);
    const size_t pN = (size_t)PW * PH;

    static thread_local TileCtx ctx;
    if (ctx.sub.size() < pN * 3) ctx.sub.resize(pN * 3);
    if (ctx.out.size() < pN * 3) ctx.out.resize(pN * 3);

    for (int y = 0; y < subH; ++y)
        std::memcpy(ctx.sub.data() + (size_t)y * PW * 3,
                    rgb + ((size_t)(iy0 + y) * W + ix0) * 3, (size_t)subW * 3);
    for (int y = 0; y < subH; ++y) {            // 右侧复制
        uint8_t* row = ctx.sub.data() + (size_t)y * PW * 3;
        for (int x = subW; x < PW; ++x) std::memcpy(row + (size_t)x * 3, row + (size_t)(subW - 1) * 3, 3);
    }
    for (int y = subH; y < PH; ++y)             // 底部复制整行
        std::memcpy(ctx.sub.data() + (size_t)y * PW * 3,
                    ctx.sub.data() + (size_t)(subH - 1) * PW * 3, (size_t)PW * 3);

    denoise_core(ctx.sub.data(), ctx.out.data(), PW, PH, thr, false);

    const int dx = ox0 - ix0, dy = oy0 - iy0;
    for (int c = 0; c < 3; ++c) {
        const float* sp = ctx.out.data() + (size_t)c * pN;
        float* dp = out3 + (size_t)c * N + (size_t)oy0 * W + ox0;
        for (int y = 0; y < oh; ++y)
            std::memcpy(dp + (size_t)y * W, sp + (size_t)(y + dy) * PW + dx,
                        (size_t)ow * sizeof(float));
    }
}

void agf2_denoise_tiled(const uint8_t* rgb, float* out3, int W, int H, const AgfParams& prm) {
    g_fused = prm.fused;
    g_nthreads = 1;                 // 每个 tile 内部单线程（tile 间并行）
    g_gf_tile = 0;
    get_pool(1);                    // 预置 pool 为单线程，避免外层多线程首次并发 ensure

    const int nt = prm.nthreads > 0 ? prm.nthreads : 6;
    const int TW = prm.tileW > 0 ? prm.tileW : 256;
    const int TH = prm.tileH > 0 ? prm.tileH : 256;
    const int HALO = prm.halo >= 0 ? prm.halo : 64;
    const int ntx = (W + TW - 1) / TW;
    const int nty = (H + TH - 1) / TH;
    const int nTiles = ntx * nty;

    std::vector<std::thread> th;
    th.reserve((size_t)(nt - 1));
    for (int t = 1; t < nt; ++t)
        th.emplace_back([&, t]() {
            for (int k = t; k < nTiles; k += nt) {
                const int ty = k / ntx, tx = k - ty * ntx;
                denoise_tile(rgb, W, H, out3, tx, ty, TW, TH, HALO, prm.maskThr);
            }
        });
    for (int k = 0; k < nTiles; k += nt) {
        const int ty = k / ntx, tx = k - ty * ntx;
        denoise_tile(rgb, W, H, out3, tx, ty, TW, TH, HALO, prm.maskThr);
    }
    for (auto& x : th) x.join();
    g_nthreads = prm.nthreads > 0 ? prm.nthreads : 1;   // 恢复到配置线程数，避免污染后续调用（如 --half 的放大）
}

// =============================================================================
//  单元测试钩子：融合内核 vs 平面内核
// =============================================================================
float agf_test_gf(const float* guide, const float* p, const float* eps,
                  float* out_fused, float* out_plane,
                  int W, int H, int r1, int r2, float epsScale, int nthr) {
    Pool pool;
    GfArgs A;
    A.guide = guide; A.p[0] = p; A.eps[0] = eps; A.epsScale = epsScale;
    A.NC = 1; A.r1 = r1; A.r2 = r2; A.centerGuide = false; A.clampOut = false; A.fused = true;
    A.out[0] = out_fused;
    g_fused = true;  gf_call(A, W, H, pool, nthr);
    A.out[0] = out_plane;
    g_fused = false; gf_call(A, W, H, pool, 1);
    g_fused = true;
    float mAll = 0.f, mIn = 0.f;
    int ax = 0, ay = 0;
    const int b = 2 * (r1 + r2) + 1;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const float d = std::fabs(out_fused[(size_t)y * W + x] - out_plane[(size_t)y * W + x]);
            if (d > mAll) { mAll = d; ax = x; ay = y; }
            if (x >= b && x < W - b && y >= b && y < H - b) mIn = std::max(mIn, d);
        }
    std::printf("    [gf] all %.4f (at %d,%d)   interior %.4f\n", mAll, ax, ay, mIn);
    return mAll;
}
