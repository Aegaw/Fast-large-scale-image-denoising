// =============================================================================
//  config.cpp -- 运行配置的默认值与 ini 解析
// =============================================================================
#include "config.h"

#include <windows.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "image_io.h"

// 若 p 是相对路径，拼到 baseDir 后面；绝对路径原样返回。
// 最后统一用 GetFullPathNameA 规范化（消掉 . / ..，GDI+ 存图对含 .. 的路径不友好）。
static std::string resolve_rel(const std::string& baseDir, const std::string& p) {
    if (p.empty()) return p;
    std::string full = p;
    const bool abs = (p.size() >= 2 && p[1] == ':') || p[0] == '\\' || p[0] == '/';
    if (!abs && !baseDir.empty()) {
        std::string d = baseDir;
        if (!d.empty() && d.back() != '\\' && d.back() != '/') d += '\\';
        full = d + p;
    }
    char buf[MAX_PATH * 2];
    const DWORD n = GetFullPathNameA(full.c_str(), (DWORD)sizeof(buf), buf, nullptr);
    return (n > 0 && n < (DWORD)sizeof(buf)) ? std::string(buf) : full;
}

AgfConfig DefaultConfig(const std::string& baseDir) {
    AgfConfig c;
    // 内置示例（找不到 run_config.ini 时用）；相对路径相对 baseDir 解析。
    // exe 在 agf2\bin，所以数据目录写 ..\noisy_images 之类。
    c.inputPath  = resolve_rel(baseDir, "..\\noisy_images");
    c.outputPath = resolve_rel(baseDir, "..\\out_images");
    c.hasGroundTruth = true;
    c.gtPath = resolve_rel(baseDir, "..\\truth_images");
    // 去噪参数保持 AgfParams 的当前默认（全图模式、6 线程）；
    // run_config.ini 里给的是实测最优（分块 tw384 halo16 nt8）。
    c.params = AgfParams{};
    return c;
}

static std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

static std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static bool to_bool(const std::string& v) {
    const std::string t = lower(trim(v));
    return t == "1" || t == "true" || t == "yes" || t == "on";
}

bool LoadConfigFromFile(const std::string& iniPath, AgfConfig& cfg, std::string* err) {
    std::ifstream f(iniPath);
    if (!f) {
        if (err) *err = "cannot open " + iniPath;
        return false;
    }
    std::string line;
    int lineNo = 0;
    while (std::getline(f, line)) {
        ++lineNo;
        // 去掉注释
        const size_t hash = line.find_first_of("#;");
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            if (err) *err = "line " + std::to_string(lineNo) + ": missing '='";
            return false;
        }
        const std::string key = lower(trim(line.substr(0, eq)));
        const std::string val = trim(line.substr(eq + 1));

        if (key == "input")            cfg.inputPath = val;
        else if (key == "output")      cfg.outputPath = val;
        else if (key == "has_gt")      cfg.hasGroundTruth = to_bool(val);
        else if (key == "gt")          cfg.gtPath = val;
        else if (key == "thr")         cfg.params.maskThr = (float)atof(val.c_str());
        else if (key == "nt")          cfg.params.nthreads = atoi(val.c_str());
        else if (key == "tiled")       cfg.params.tiled = to_bool(val);
        else if (key == "tw")          cfg.params.tileW = atoi(val.c_str());
        else if (key == "th")          cfg.params.tileH = atoi(val.c_str());
        else if (key == "halo")        cfg.params.halo = atoi(val.c_str());
        else if (key == "half")        cfg.params.half = to_bool(val);
        else if (key == "stream")      cfg.params.stream = to_bool(val);
        else if (key == "verbose")     cfg.params.verbose = to_bool(val);
        else if (err) { *err = "line " + std::to_string(lineNo) + ": unknown key '" + key + "'"; return false; }
    }
    // 相对路径一律相对 ini 所在目录解析（绝对路径原样保留），
    // 这样把整个 cpp_agf2 目录搬走也能跑。
    const std::string base = dir_of(iniPath);
    cfg.inputPath = resolve_rel(base, cfg.inputPath);
    cfg.outputPath = resolve_rel(base, cfg.outputPath);
    cfg.gtPath = resolve_rel(base, cfg.gtPath);
    return true;
}
