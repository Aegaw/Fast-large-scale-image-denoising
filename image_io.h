// =============================================================================
//  image_io.h -- PNG 读写（GDI+）与 PSNR
//  ---------------------------------------------------------------------------
//  使用前必须先 GdiplusStartup（main() 里已经做了）。
// =============================================================================
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// RGB 交织、每像素 3 字节
struct Img {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;
};

bool load_png(const std::string& path, Img& im);
bool save_png(const std::string& path, const uint8_t* rgb, int w, int h);

// 8bit 单通道/交织通用的 PSNR（n = 字节数）
double psnr(const uint8_t* a, const uint8_t* b, size_t n);

bool file_exists(const std::string& p);
bool is_directory(const std::string& p);
std::string dir_of(const std::string& p);
std::string base_of(const std::string& p);

// 列出目录下所有 *.png（完整路径），按文件名升序排序
std::vector<std::string> list_png_sorted(const std::string& dir);
