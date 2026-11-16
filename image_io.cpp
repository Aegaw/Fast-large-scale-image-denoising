// =============================================================================
//  image_io.cpp -- PNG 读写（GDI+）与 PSNR
// =============================================================================
#include "image_io.h"

#include <algorithm>
using std::min;      // gdiplus 头里用了未限定的 min/max
using std::max;

#include <windows.h>
#include <gdiplus.h>
#include <cmath>
#include <cstdio>

#pragma comment(lib, "gdiplus.lib")

using namespace Gdiplus;

static std::wstring widen(const std::string& s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w((size_t)n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}

bool load_png(const std::string& path, Img& im) {
    Bitmap bmp(widen(path).c_str());
    if (bmp.GetLastStatus() != Ok) return false;
    const int w = (int)bmp.GetWidth(), h = (int)bmp.GetHeight();
    Bitmap conv(w, h, PixelFormat32bppARGB);
    Graphics g(&conv);
    g.DrawImage(&bmp, 0, 0, w, h);
    Rect r(0, 0, w, h);
    BitmapData bd;
    if (conv.LockBits(&r, ImageLockModeRead, PixelFormat32bppARGB, &bd) != Ok) return false;
    im.w = w; im.h = h;
    im.rgb.resize((size_t)w * h * 3);
    for (int y = 0; y < h; ++y) {
        const uint8_t* s = (const uint8_t*)bd.Scan0 + (size_t)y * bd.Stride;
        uint8_t* d = im.rgb.data() + (size_t)y * w * 3;
        for (int x = 0; x < w; ++x) {
            d[3 * x + 0] = s[4 * x + 2];
            d[3 * x + 1] = s[4 * x + 1];
            d[3 * x + 2] = s[4 * x + 0];
        }
    }
    conv.UnlockBits(&bd);
    return true;
}

static int get_encoder_clsid(const wchar_t* mime, CLSID* clsid) {
    UINT num = 0, size = 0;
    GetImageEncodersSize(&num, &size);
    if (size == 0) return -1;
    std::vector<uint8_t> buf(size);
    ImageCodecInfo* info = (ImageCodecInfo*)buf.data();
    GetImageEncoders(num, size, info);
    for (UINT i = 0; i < num; ++i)
        if (wcscmp(info[i].MimeType, mime) == 0) { *clsid = info[i].Clsid; return (int)i; }
    return -1;
}

bool save_png(const std::string& path, const uint8_t* rgb, int w, int h) {
    Bitmap out(w, h, PixelFormat24bppRGB);
    Rect r(0, 0, w, h);
    BitmapData bd;
    if (out.LockBits(&r, ImageLockModeWrite, PixelFormat24bppRGB, &bd) != Ok) return false;
    for (int y = 0; y < h; ++y) {
        uint8_t* d = (uint8_t*)bd.Scan0 + (size_t)y * bd.Stride;
        const uint8_t* s = rgb + (size_t)y * w * 3;
        for (int x = 0; x < w; ++x) {
            d[3 * x + 0] = s[3 * x + 2];
            d[3 * x + 1] = s[3 * x + 1];
            d[3 * x + 2] = s[3 * x + 0];
        }
    }
    out.UnlockBits(&bd);
    CLSID png;
    if (get_encoder_clsid(L"image/png", &png) < 0) return false;
    return out.Save(widen(path).c_str(), &png, nullptr) == Ok;
}

double psnr(const uint8_t* a, const uint8_t* b, size_t n) {
    double se = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)a[i] - (double)b[i];
        se += d * d;
    }
    const double mse = se / (double)n;
    return mse <= 1e-12 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
}

bool file_exists(const std::string& p) {
    const DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

bool is_directory(const std::string& p) {
    const DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}

std::string dir_of(const std::string& p) {
    const size_t k = p.find_last_of("\\/");
    return k == std::string::npos ? std::string(".") : p.substr(0, k);
}

std::string base_of(const std::string& p) {
    const size_t k = p.find_last_of("\\/");
    return k == std::string::npos ? p : p.substr(k + 1);
}

std::vector<std::string> list_png_sorted(const std::string& dir) {
    std::vector<std::string> out;
    WIN32_FIND_DATAA fd;
    const std::string pat = dir + "\\*.png";
    HANDLE h = FindFirstFileA(pat.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return out;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        out.push_back(dir + "\\" + fd.cFileName);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    std::sort(out.begin(), out.end());   // 按完整路径排序 == 按文件名排序
    return out;
}
