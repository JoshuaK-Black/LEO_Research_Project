#pragma once
#include <cstdio>
#include <string>
#include <mutex>
#include <cstdarg>

class VisPublisher {
public:
  explicit VisPublisher(const std::string& path = "/tmp/leo_vis.ndjson")
    : m_path(path) {
    m_fp = std::fopen(m_path.c_str(), "w");
    if (!m_fp) {
      std::perror("open vis file");
    }
    writeHeader();
  }

  ~VisPublisher() {
    if (m_fp) {
      std::fclose(m_fp);
    }
  }

  void WriteLine(const std::string& json) {
    if (!m_fp) {
      return;
    }
    std::lock_guard<std::mutex> lk(m_mu);
    std::fputs(json.c_str(), m_fp);
    std::fputc('\n', m_fp);
    std::fflush(m_fp);
  }

  void EmitSatState(double t, int satId, double latDeg, double lonDeg, double altM) {
    WriteLine(fmt(R"({"type":"sat","t":%.3f,"id":%d,"lat":%.6f,"lon":%.6f,"alt":%.1f})",
                  t, satId, latDeg, lonDeg, altM));
  }

  void EmitIslState(double t, int a, int b, bool up) {
    WriteLine(fmt(R"({"type":"isl","t":%.3f,"a":%d,"b":%d,"up":%s})",
                  t, a, b, up ? "true" : "false"));
  }

  void EmitHop(double t, int srcSat, int dstSat, int nh, bool isSpf) {
    WriteLine(fmt(R"({"type":"hop","t":%.3f,"src":%d,"dst":%d,"nh":%d,"mode":"%s"})",
                  t, srcSat, dstSat, nh, isSpf ? "OSPF" : "OTHER"));
  }

private:
  std::string m_path;
  std::mutex m_mu;
  std::FILE* m_fp{nullptr};

  static std::string fmt(const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    char buf[1024];
    vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
  }

  void writeHeader() {
    WriteLine(R"({"type":"meta","version":1})");
  }
};
