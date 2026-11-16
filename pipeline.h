// =============================================================================
//  pipeline.h -- 对外唯一入口：整条去噪流水线
//  ---------------------------------------------------------------------------
//  agf_denoise_rgb()  就是"整个去噪过程"这一个函数，参数可改，默认值即当前配置。
//  agf_run()          在它外面包一层：读图 -> 去噪 -> 写图 ->（有真值则）算 PSNR。
//  agf_run_auto()     自动判断 input 是"文件"还是"目录"：
//                       文件 -> 单张；目录 -> 批量（noisy 与 gt 按名称排序后逐张对应）
// =============================================================================
#pragma once
#include <cstdint>

#include "agf.h"
#include "config.h"

struct DenoiseResult {
    bool   ok = false;
    int    w = 0, h = 0;
    double ms = 0;              // 纯去噪耗时（不含 PNG 读写）
    bool   hasGt = false;       // 是否真的用上了真值
    double psnrOutGt = 0.0;     // 去噪结果 vs 真值
    double psnrNoisyGt = 0.0;   // 噪声图   vs 真值（基线）
    double psnrOutNoisy = 0.0;  // 去噪结果 vs 噪声图
};

struct BatchResult {
    int    count = 0;           // 成功处理的张数
    int    skipped = 0;         // 跳过（尺寸/读取问题）
    double meanPsnr = 0.0;
    double minPsnr = 0.0;
    double maxPsnr = 0.0;
    double meanMs = 0.0;
    double totalS = 0.0;        // 去噪耗时合计（秒，不含 IO）
    bool   hasGt = false;
};

// ★ 整个去噪过程 = 这一个函数
//    rgb : H*W*3 交织 u8 输入
//    out : H*W*3 交织 u8 输出（可与 rgb 不同）
//    prm : 去噪参数，默认值 = AgfParams 的当前默认（改 prm 即可切换
//          全图 / 分块 / 半分辨率 / 线程数 / mask 阈值 等）
//    要求：W、H 为 128 的倍数（算法按 8×8 块与 2× 金字塔对齐）
void agf_denoise_rgb(const uint8_t* rgb, uint8_t* out, int W, int H,
                     const AgfParams& prm = AgfParams{});

// 单张：读取路径 -> 去噪 -> 写入路径 ->（是否有真值）PSNR
DenoiseResult agf_run(const AgfConfig& cfg);

// 批量：cfg.inputPath 是目录。noisy 与 gt 各自按文件名排序后逐张一一对应；
//       outputPath 视作输出目录（每张写成 <out>\<noisy 文件名>）。
BatchResult agf_run_batch(const AgfConfig& cfg);

// 自动分派（文件 -> agf_run，目录 -> agf_run_batch）。返回进程退出码。
int agf_run_auto(const AgfConfig& cfg);
