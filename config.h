// =============================================================================
//  config.h -- 运行配置：读取路径 / 写入路径 / 是否存在真值图像 / 去噪参数
//  ---------------------------------------------------------------------------
//  这个文件（以及同目录的 run_config.ini）就是整套程序唯一的"入口配置"。
//  想换图只需要改 run_config.ini，不用重新编译。
// =============================================================================
#pragma once
#include <string>

#include "agf.h"

struct AgfConfig {
    // ---- 路径 ----
    std::string inputPath;          // 输入：噪声图（RGB PNG）
    std::string outputPath;         // 输出：去噪结果；留空则不写盘

    // ---- 是否存在真值图像（有就顺便算 PSNR） ----
    bool        hasGroundTruth = false;
    std::string gtPath;             // hasGroundTruth = true 时有效

    // ---- 去噪参数（默认值 = agf.h 里 AgfParams 的当前默认） ----
    AgfParams   params;
};

// 内置默认配置。baseDir 非空时，默认的（相对）路径会相对它解析，
// 一般传 exe 所在目录，这样把整个工程目录搬走也能找到数据。
AgfConfig DefaultConfig(const std::string& baseDir = "");

// 从 ini 文件读取并覆盖 cfg 中的字段。文件不存在返回 false。
// 相对路径（input/output/gt）一律相对 ini 所在目录解析；绝对路径原样保留。
// 支持的 key（大小写不敏感，# 或 ; 开头是注释）：
//   input / output / has_gt / gt
//   thr / nt / tiled / tw / th / halo / half / stream / verbose
bool LoadConfigFromFile(const std::string& iniPath, AgfConfig& cfg, std::string* err = nullptr);
