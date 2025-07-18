// sherpa-mlx/csrc/file-utils.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-mlx/csrc/file-utils.h"

#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "sherpa-mlx/csrc/macros.h"

namespace sherpa_mlx {

bool FileExists(const std::string &filename) {
  return std::ifstream(filename).good();
}

void AssertFileExists(const std::string &filename) {
  if (!FileExists(filename)) {
    SHERPA_MLX_LOGE("filename '%s' does not exist", filename.c_str());
    SHERPA_MLX_EXIT(-1);
  }
}

std::vector<char> ReadFile(const std::string &filename) {
  std::ifstream input(filename, std::ios::binary);
  std::vector<char> buffer(std::istreambuf_iterator<char>(input), {});
  return buffer;
}

#if __ANDROID_API__ >= 9
std::vector<char> ReadFile(AAssetManager *mgr, const std::string &filename) {
  AAsset *asset = AAssetManager_open(mgr, filename.c_str(), AASSET_MODE_BUFFER);
  if (!asset) {
    __android_log_print(ANDROID_LOG_FATAL, "sherpa-mlx",
                        "Read binary file: Load '%s' failed", filename.c_str());
    SHERPA_MLX_EXIT(-1);
  }

  auto p = reinterpret_cast<const char *>(AAsset_getBuffer(asset));
  size_t asset_length = AAsset_getLength(asset);

  std::vector<char> buffer(p, p + asset_length);
  AAsset_close(asset);

  return buffer;
}
#endif

#if __OHOS__
std::vector<char> ReadFile(NativeResourceManager *mgr,
                           const std::string &filename) {
  std::unique_ptr<RawFile, decltype(&OH_ResourceManager_CloseRawFile)> fp(
      OH_ResourceManager_OpenRawFile(mgr, filename.c_str()),
      OH_ResourceManager_CloseRawFile);

  if (!fp) {
    std::ostringstream os;
    os << "Read file '" << filename << "' failed.";
    SHERPA_MLX_LOGE("%s", os.str().c_str());
    return {};
  }

  auto len = static_cast<int32_t>(OH_ResourceManager_GetRawFileSize(fp.get()));

  std::vector<char> buffer(len);

  int32_t n = OH_ResourceManager_ReadRawFile(fp.get(), buffer.data(), len);

  if (n != len) {
    std::ostringstream os;
    os << "Read file '" << filename << "' failed. Number of bytes read: " << n
       << ". Expected bytes to read: " << len;
    SHERPA_MLX_LOGE("%s", os.str().c_str());
    return {};
  }

  return buffer;
}
#endif

}  // namespace sherpa_mlx
