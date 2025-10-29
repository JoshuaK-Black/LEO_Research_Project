// scratch/leo_qrouting_ip.cc — Q-table baseline harness (drop-in)
// Build:  ./ns3 build
// Run:    ./ns3 run "scratch/leo_qrouting_ip --planes=6 --perPlane=11 ..."
// This file matches your OSPF-lite harness’ network style/knobs so results are comparable.

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/application.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/log.h"
#include "ns3/traffic-control-module.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/timestamp-tag.h"
#include "ns3/output-stream-wrapper.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <random>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <csignal>
#include <ctime>
#include <functional>
#include <cmath>
#include <system_error>
#include <map>
#include <limits>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <string>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include "qtable-routing-range.h"  // extended Q-table protocol with range hooks
#include "ns3/opengym-module.h"
#include "leo_rl_gym_env.h"

#include <filesystem>
namespace ns3 {
uint64_t g_rlOverridePkts = 0;
uint64_t g_rlPktUsed = 0;
bool g_logRl = false;
double g_rlStartS = 60.0;
std::atomic<uint64_t> g_qrProbeTxPkts{0};
std::atomic<uint64_t> g_qrPackTxPkts{0};
std::atomic<uint64_t> g_qrFbTxPkts{0};
std::atomic<uint64_t> g_qrProbeTxBytes{0};
std::atomic<uint64_t> g_qrPackTxBytes{0};
std::atomic<uint64_t> g_qrFbTxBytes{0};

void QrCtrlResetCounters()
{
  g_qrProbeTxPkts.store(0, std::memory_order_relaxed);
  g_qrPackTxPkts.store(0, std::memory_order_relaxed);
  g_qrFbTxPkts.store(0, std::memory_order_relaxed);
  g_qrProbeTxBytes.store(0, std::memory_order_relaxed);
  g_qrPackTxBytes.store(0, std::memory_order_relaxed);
  g_qrFbTxBytes.store(0, std::memory_order_relaxed);
}

void QrCtrlPrintProxy(double winS)
{
  const uint64_t ctrlBytes =
      g_qrProbeTxBytes.load(std::memory_order_relaxed) +
      g_qrPackTxBytes.load(std::memory_order_relaxed) +
      g_qrFbTxBytes.load(std::memory_order_relaxed);
  const double ctrlMbps = (winS > 0.0) ? (8.0 * static_cast<double>(ctrlBytes) / winS / 1e6) : 0.0;

  std::cout << "\n==== CONTROL (proxy) ====\n"
            << "QR_ProbePkts: " << g_qrProbeTxPkts.load(std::memory_order_relaxed) << "\n"
            << "QR_PackPkts:  " << g_qrPackTxPkts.load(std::memory_order_relaxed)  << "\n"
            << "QR_FbPkts:    " << g_qrFbTxPkts.load(std::memory_order_relaxed)    << "\n"
            << "QR_CtrlBytesTx: " << ctrlBytes
            << "  QR_CtrlAvgRate(Mbps): " << ctrlMbps << "\n";
}
}
using namespace ns3;

// ---------------- CLI (sim + Q params) ----------------
static uint32_t g_planes=6, g_perPlane=11; static bool g_wrap=true;
static double g_spacingKm=5000, g_simTime=180, g_islRateMbps=40, g_islDelayMs=5;
// Baseline constellation/link dynamics (baked defaults)
static double g_rangeKm=6500, g_checkPeriod=0.10, g_blackoutMs=150;
static double g_rangeStart=0.05;
static uint32_t g_flows=200; static uint16_t g_pktSize=1500; static double g_interPacket=0.02;
static uint32_t g_runSeed=1; // legacy flow-sampler seed (std::mt19937)
// ns-3 RNG controls (new)
static uint32_t g_rngSeed=12345;
static uint32_t g_rngRun=1;
static uint32_t g_numGs=12; static double g_gravityAlpha=2.0;
static bool        g_useFlowMon = false;
static std::string g_exportCsv="";
static std::string g_exportSummary = "";
static bool        g_exportSummaryCsv = false;
static std::string g_tag="QR";
static double g_progressIntervalSec = 5.0;
static uint32_t g_logEveryPkts = 5000;
static double g_logEverySimSec = 5.0;
static bool g_quietApps = true;
static bool g_qrInfoLogs = false;  // control QTableRouting INFO logs
static bool g_enablePcap = false;
static bool g_forceCleanLinks = false;
static bool g_logRebuild = false;
static bool g_logAdj = false;
static bool g_logRange = false;
static bool g_dumpQInit = false;
static bool g_dumpQFinal = false;
static double g_measureStart = 60.0;   // ensure default=60
static bool g_dumpQOnlyGs = false;
static bool g_dstOnlyGs = true;
static bool g_debugFlowEvents = false;
static bool g_forceOverride = false;
static std::string g_pidFilePath;
static std::string g_logFileRequest;
// Optional: force all flows through fixed src/dst (for RL bring-up proof)
static int32_t g_forceSrcNode = -1;  // -1 disables
static int32_t g_forceDstNode = -1;  // -1 disables
// RL env knobs (Step 1 handshake)
static bool     g_rlEnable    = false;
static double   g_rlDelta     = 0.25;   // decision period (s)
static uint32_t g_rlK         = 6;      // max neighbor fanout
static uint16_t g_openGymPort = 5555;   // ns3-gym port
static int32_t  g_rlTrainDst  = -1;     // destination node id to train for (-1 cycle)
static double   g_rlHopPenalty  = 0.0;
static double   g_rlQPenalty    = 0.0;
static double   g_rlDropPenalty = 0.5;
static std::string g_teacherPolicy = "qtable";
// Optional: wait for agent to attach before starting traffic
static bool g_waitAgent = false;

// ISL egress queue depth (packets)
static uint32_t g_queuePkts = 500;  // baseline queue depth
// cmd.AddValue("queuePkts", "ISL egress queue depth (packets)", g_queuePkts);

// Q-table hyperparams
static double g_qAlpha=0.2;
static double g_qGamma=0.9;
static double g_qEpsStart=0.15;
static double g_qEpsFinal=0.02;
static double g_qEpsTau=30.0;
static double g_qSeedHopMs=10.0;
static uint32_t g_qUpdateStride = 10;  // default stride for faster sims (can override via CLI)
static double g_qProbeInterval=0.3; static uint32_t g_qProbeFanout=2;
static double g_qPenaltyDropMs=500; static uint16_t g_ctrlPort=8899;
static std::vector<bool> g_groundDestMask;
static std::string g_schedulerType = "ns3::HeapScheduler";
static double g_satAltKm = 780.0;
static double g_satInclinationDeg = 86.4;
static double g_walkerPhaseF = 2.0;
// SP seeding + refresh + freeze
static bool   g_qSeedFromSpf = true;
static double g_qSpSeedMs    = 3.0;
static double g_qSpfRefresh  = 0.0;
static bool   g_freezeAfterMeasure = false;
// New tunables
static double g_qProbeEwmaW  = 0.20;  // weight of new RTT in EWMA
static double g_qSwitchHyst  = 0.05;  // hysteresis when switching next-hop (fraction)
static double g_qAlphaAfter  = -1.0;  // evaluation alpha (negative disables override)
static double g_qEpsAfter    = -1.0;  // evaluation epsilon (negative disables override)

static std::string
SchedulerAlias(const std::string& type)
{
  if (type.find("Heap") != std::string::npos)
  {
    return "heap";
  }
  if (type.find("Calendar") != std::string::npos)
  {
    return "calendar";
  }
  if (type.find("Map") != std::string::npos)
  {
    return "map";
  }
  return type;
}

// Delay/model constants (mirrors OSPF harness defaults)
static const double kCEff = 3.0e8;   // m/s
static const double kMinD = 1.0;     // ms
static const double kMaxD = 50.0;    // ms
static const double kBucket = 1.0;   // ms quantization
static constexpr double kEarthRadiusM = 6371000.0;
static constexpr double kEarthRotationRate = 7.2921159e-5;
static constexpr double kEarthMu = 3.986004418e14;

struct SatGeom
{
  Vector ecef;
  double latDeg{0.0};
  double lonDeg{0.0};
  double altM{0.0};
};

static SatGeom
ComputeSatGeom(uint32_t satId, double timeSec)
{
  SatGeom out;
  const double altitudeM = g_satAltKm * 1000.0;
  const double r = kEarthRadiusM + altitudeM;
  const double incRad = g_satInclinationDeg * M_PI / 180.0;
  const double ci = std::cos(incRad);
  const double si = std::sin(incRad);
  const uint32_t planes = std::max<uint32_t>(1u, g_planes);
  const uint32_t slotsPerPlane = std::max<uint32_t>(1u, g_perPlane);
  const uint32_t plane = satId / slotsPerPlane;
  const uint32_t slot  = satId % slotsPerPlane;

  const double raan = 2.0 * M_PI * (static_cast<double>(plane) / static_cast<double>(planes));
  const double baseU = 2.0 * M_PI * ((static_cast<double>(slot) +
      (g_walkerPhaseF * static_cast<double>(plane) / static_cast<double>(planes))) /
      static_cast<double>(slotsPerPlane));
  const double meanMotion = std::sqrt(kEarthMu / (r * r * r));
  const double u = baseU + meanMotion * timeSec;

  const double cosU = std::cos(u);
  const double sinU = std::sin(u);
  const double cosRaan = std::cos(raan);
  const double sinRaan = std::sin(raan);

  double xEci = (cosRaan * cosU - sinRaan * sinU * ci) * r;
  double yEci = (sinRaan * cosU + cosRaan * sinU * ci) * r;
  double zEci = (sinU * si) * r;

  const double theta = kEarthRotationRate * timeSec;
  const double cosTheta = std::cos(theta);
  const double sinTheta = std::sin(theta);
  double x = cosTheta * xEci + sinTheta * yEci;
  double y = -sinTheta * xEci + cosTheta * yEci;
  double z = zEci;

  out.ecef = Vector(x, y, z);
  const double rxy = std::sqrt(x * x + y * y);
  const double rnorm = std::sqrt(x * x + y * y + z * z);
  out.latDeg = std::atan2(z, rxy) * 180.0 / M_PI;
  out.lonDeg = std::atan2(y, x) * 180.0 / M_PI;
  if (out.lonDeg > 180.0) out.lonDeg -= 360.0;
  if (out.lonDeg < -180.0) out.lonDeg += 360.0;
  out.altM = rnorm - kEarthRadiusM;
  return out;
}

static inline double
ChordDistance(const Vector& a, const Vector& b)
{
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

static inline bool
HasLineOfSight(const Vector& a, const Vector& b)
{
  const Vector diff(b.x - a.x, b.y - a.y, b.z - a.z);
  const double ab2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
  if (ab2 <= 0.0)
  {
    return true;
  }
  double t = -(a.x * diff.x + a.y * diff.y + a.z * diff.z) / ab2;
  if (t < 0.0) t = 0.0;
  else if (t > 1.0) t = 1.0;
  const Vector closest(a.x + t * diff.x, a.y + t * diff.y, a.z + t * diff.z);
  const double dist2 = closest.x * closest.x + closest.y * closest.y + closest.z * closest.z;
  return dist2 >= kEarthRadiusM * kEarthRadiusM;
}

// --------------- helpers ---------------
static inline double sqr(double x){ return x*x; }
static uint64_t MakeKey(uint32_t u, uint32_t v){ if(u>v) std::swap(u,v); return (uint64_t(u)<<32)|v; }
static double g_progressBeatSec = 10.0;

// Dump current CLI/runtime flags in key=value lines for tiny summary
static std::string DumpFlagsQ() {
  std::ostringstream s;
  // Topology & runtime
  s << "planes=" << g_planes << "\n";
  s << "perPlane=" << g_perPlane << "\n";
  s << "simTime=" << g_simTime << "\n";
  s << "numGs=" << g_numGs << "\n";
  s << "rangeKm=" << g_rangeKm << "\n";
  s << "checkPeriod=" << g_checkPeriod << "\n";
  s << "blackoutMs=" << g_blackoutMs << "\n";
  s << "islRateMbps=" << g_islRateMbps << "\n";
  s << "islDelayMs=" << g_islDelayMs << "\n";
  s << "queuePkts=" << g_queuePkts << "\n";
  s << "satAltitudeKm=" << g_satAltKm << "\n";
  s << "satInclDeg=" << g_satInclinationDeg << "\n";
  s << "walkerPhase=" << g_walkerPhaseF << "\n";

  // Traffic
  s << "flows=" << g_flows << "\n";
  s << "pktSize=" << g_pktSize << "\n";
  s << "interPacket=" << g_interPacket << "\n";

  // Q-learning knobs
  s << "qAlpha=" << g_qAlpha << "\n";
  s << "qGamma=" << g_qGamma << "\n";
  s << "qEpsStart=" << g_qEpsStart << "\n";
  s << "qEpsFinal=" << g_qEpsFinal << "\n";
  s << "qEpsTau=" << g_qEpsTau << "\n";
  s << "qUpdateStride=" << g_qUpdateStride << "\n";
  s << "qProbeInterval=" << g_qProbeInterval << "\n";
  s << "qProbeFanout=" << g_qProbeFanout << "\n";
  s << "qPenaltyDropMs=" << g_qPenaltyDropMs << "\n";
  s << "qSeedFromSpf=" << (g_qSeedFromSpf ? 1 : 0) << "\n";
  s << "freezeAfterMeasure=" << (g_freezeAfterMeasure ? 1 : 0) << "\n";

  // Misc
  s << "measureStart=" << g_measureStart << "\n";
  s << "tag=" << g_tag << "\n";
  return s.str();
}

struct FlowSummary
{
  uint64_t txPackets{0};
  uint64_t rxPackets{0};
  uint64_t rxBytes{0};
  double   firstTx{-1.0};
  double   lastTx{-1.0};
  double   firstRx{-1.0};
  double   lastRx{-1.0};
  double   delaySum{0.0};
  bool     dstIsGs{false};
};

static std::vector<FlowSummary> g_flowSummaries;

static void RecordSinkRx(uint32_t flowId, Ptr<const Packet> pkt, const Address&)
{
  if (flowId >= g_flowSummaries.size())
  {
    return;
  }

  const double now = Simulator::Now().GetSeconds();
  // --- MEASUREMENT WINDOW RX GATING BY ARRIVAL ---
  if (now < g_measureStart)
  {
    return; // ignore pre-measurement packets by arrival time
  }

  auto& stats = g_flowSummaries[flowId];
  // Always count RX for PDR in-window, regardless of tag presence
  stats.rxPackets++;
  stats.rxBytes += pkt->GetSize();
  if (stats.firstRx < 0.0)
  {
    stats.firstRx = now;
  }
  stats.lastRx = now;

  // One-time debug to prove RX accounting is happening post-window
  static int dbg_rx = 0;
  if (dbg_rx < 5)
  {
    TimestampTag tdbg;
    bool hasTag = pkt->PeekPacketTag(tdbg);
    std::cout << "[MEASURE][RX] t=" << now
              << " flow=" << flowId
              << " hasTag=" << (hasTag ? 1 : 0) << "\n";
    ++dbg_rx;
  }

  // Delay: prefer send time from TimestampTag when present
  TimestampTag ts;
  Ptr<Packet> copy = pkt->Copy();
  if (copy->PeekPacketTag(ts))
  {
    const double d = (Simulator::Now() - ts.GetTimestamp()).GetSeconds();
    stats.delaySum += d;
  }
}

namespace
{

class TeeStreambuf : public std::streambuf
{
public:
  TeeStreambuf(std::streambuf* primary, std::streambuf* secondary)
    : m_primary(primary), m_secondary(secondary)
  {
  }

protected:
  int overflow(int ch) override
  {
    if (ch == EOF)
    {
      return !EOF;
    }

    const int p = m_primary ? m_primary->sputc(ch) : ch;
    const int s = m_secondary ? m_secondary->sputc(ch) : ch;
    if (p == EOF || s == EOF)
    {
      return EOF;
    }
    return ch;
  }

  int sync() override
  {
    int status = 0;
    if (m_primary && m_primary->pubsync() != 0)
    {
      status = -1;
    }
    if (m_secondary && m_secondary->pubsync() != 0)
    {
      status = -1;
    }
    return status;
  }

private:
  std::streambuf* m_primary;
  std::streambuf* m_secondary;
};

class LogFileGuard
{
public:
  bool Setup(const std::string& requestedPath, bool allowForce);
  ~LogFileGuard()
  {
    Restore();
  }

  const std::string& Resolved() const
  {
    return m_resolvedPath;
  }

  bool Active() const
  {
    return m_fd >= 0;
  }

private:
  void Restore();
  bool Acquire(const std::string& candidate, bool allowForce);
  std::string BuildTimestampedPath(const std::filesystem::path& base, uint32_t attempt) const;
  void InstallTees();

  int m_fd{-1};
  std::string m_resolvedPath;
  std::ofstream m_stream;
  std::streambuf* m_originalCout{nullptr};
  std::streambuf* m_originalCerr{nullptr};
  std::unique_ptr<TeeStreambuf> m_coutTee;
  std::unique_ptr<TeeStreambuf> m_cerrTee;
  bool m_redirected{false};
};

bool LogFileGuard::Setup(const std::string& requestedPath, bool allowForce)
{
  if (requestedPath.empty())
  {
    return true;
  }

  namespace fs = std::filesystem;

  fs::path base(requestedPath);
  if (base.has_parent_path())
  {
    std::error_code ec;
    fs::create_directories(base.parent_path(), ec);
    if (ec)
    {
      std::cerr << "[ERROR] unable to create log directory for " << requestedPath
                << ": " << ec.message() << "\n";
      return false;
    }
  }

  const fs::path original = base;
  std::string candidate = base.string();
  uint32_t attempt = 0;
  while (true)
  {
    if (Acquire(candidate, allowForce))
    {
      if (m_stream.is_open())
      {
        m_stream.close();
      }
      m_stream.open(candidate, std::ios::out | std::ios::trunc);
      if (!m_stream.is_open())
      {
        std::cerr << "[ERROR] unable to open log file " << candidate << " for writing\n";
        Restore();
        return false;
      }
      m_stream.setf(std::ios::unitbuf);
      m_resolvedPath = candidate;
      InstallTees();
      return true;
    }

    if (allowForce)
    {
      return false;
    }

    candidate = BuildTimestampedPath(original, ++attempt);
  }
}

bool LogFileGuard::Acquire(const std::string& candidate, bool allowForce)
{
  int fd = ::open(candidate.c_str(), O_RDWR | O_CREAT, 0644);
  if (fd < 0)
  {
    std::cerr << "[ERROR] open(" << candidate << ") failed: " << std::strerror(errno) << "\n";
    return false;
  }

  int flags = LOCK_EX | (allowForce ? 0 : LOCK_NB);
  if (flock(fd, flags) != 0)
  {
    if (!allowForce && errno == EWOULDBLOCK)
    {
      ::close(fd);
      return false;
    }

    std::cerr << "[ERROR] flock(" << candidate << ") failed: " << std::strerror(errno) << "\n";
    ::close(fd);
    return false;
  }

  if (ftruncate(fd, 0) != 0)
  {
    std::cerr << "[WARN] ftruncate(" << candidate << ") failed: " << std::strerror(errno) << "\n";
  }

  m_fd = fd;
  return true;
}

std::string LogFileGuard::BuildTimestampedPath(const std::filesystem::path& base, uint32_t attempt) const
{
  namespace fs = std::filesystem;
  auto now = std::chrono::system_clock::now();
  std::time_t tt = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_MSC_VER)
  localtime_s(&tm, &tt);
#else
  std::tm* tmp = std::gmtime(&tt);
  if (tmp)
  {
    tm = *tmp;
  }
#endif

  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);

  std::ostringstream name;
  name << base.stem().string() << '_' << buf;
  if (attempt > 0)
  {
    name << '_' << attempt;
  }
  name << base.extension().string();

  fs::path result = base.parent_path() / name.str();
  return result.string();
}

void LogFileGuard::InstallTees()
{
  if (!m_stream.is_open())
  {
    return;
  }
  m_originalCout = std::cout.rdbuf();
  m_originalCerr = std::cerr.rdbuf();
  m_coutTee = std::make_unique<TeeStreambuf>(m_originalCout, m_stream.rdbuf());
  m_cerrTee = std::make_unique<TeeStreambuf>(m_originalCerr, m_stream.rdbuf());
  std::cout.rdbuf(m_coutTee.get());
  std::cerr.rdbuf(m_cerrTee.get());
  m_redirected = true;
}

void LogFileGuard::Restore()
{
  if (m_redirected)
  {
    std::cout.rdbuf(m_originalCout);
    std::cerr.rdbuf(m_originalCerr);
    m_redirected = false;
  }
  if (m_stream.is_open())
  {
    m_stream.flush();
    m_stream.close();
  }
  if (m_fd >= 0)
  {
    flock(m_fd, LOCK_UN);
    ::close(m_fd);
    m_fd = -1;
  }
}

class PidFileGuard
{
public:
  bool Acquire(const std::string& path, bool allowForce);
  ~PidFileGuard()
  {
    Release();
  }

  const std::string& Path() const
  {
    return m_path;
  }

private:
  void Release();

  int m_fd{-1};
  std::string m_path;
};

bool PidFileGuard::Acquire(const std::string& path, bool allowForce)
{
  if (path.empty())
  {
    return true;
  }

  namespace fs = std::filesystem;
  fs::path p(path);
  if (p.has_parent_path())
  {
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    if (ec)
    {
      std::cerr << "[ERROR] unable to create pid directory for " << path
                << ": " << ec.message() << "\n";
      return false;
    }
  }

  int fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
  if (fd < 0)
  {
    std::cerr << "[ERROR] open(" << path << ") failed: " << std::strerror(errno) << "\n";
    return false;
  }

  int flags = LOCK_EX | (allowForce ? 0 : LOCK_NB);
  if (flock(fd, flags) != 0)
  {
    if (!allowForce && errno == EWOULDBLOCK)
    {
      std::cerr << "[ERROR] pid file " << path << " is locked by another process\n";
    }
    else
    {
      std::cerr << "[ERROR] flock(" << path << ") failed: " << std::strerror(errno) << "\n";
    }
    ::close(fd);
    return false;
  }

  if (lseek(fd, 0, SEEK_SET) == -1)
  {
    std::cerr << "[WARN] lseek(" << path << ") failed: " << std::strerror(errno) << "\n";
  }

  std::string existing(64, '\0');
  ssize_t n = ::read(fd, existing.data(), existing.size());
  if (n > 0)
  {
    existing.resize(static_cast<size_t>(n));
    pid_t previousPid = static_cast<pid_t>(std::strtol(existing.c_str(), nullptr, 10));
    if (previousPid > 0 && previousPid != getpid())
    {
      if (!allowForce && (::kill(previousPid, 0) == 0 || errno == EPERM))
      {
        std::cerr << "[ERROR] pid file " << path << " belongs to live pid " << previousPid
                  << "; use --force=1 to override\n";
        flock(fd, LOCK_UN);
        ::close(fd);
        return false;
      }
    }
  }

  if (ftruncate(fd, 0) != 0)
  {
    std::cerr << "[WARN] ftruncate(" << path << ") failed: " << std::strerror(errno) << "\n";
  }

  if (lseek(fd, 0, SEEK_SET) == -1)
  {
    std::cerr << "[WARN] lseek reset(" << path << ") failed: " << std::strerror(errno) << "\n";
  }

  std::ostringstream oss;
  oss << getpid() << '\n';
  std::string payload = oss.str();
  ssize_t wrote = ::write(fd, payload.data(), payload.size());
  if (wrote < static_cast<ssize_t>(payload.size()))
  {
    std::cerr << "[WARN] short write to pid file " << path << "\n";
  }
  ::fsync(fd);

  m_fd = fd;
  m_path = path;
  return true;
}

void PidFileGuard::Release()
{
  if (m_fd >= 0)
  {
    flock(m_fd, LOCK_UN);
    ::close(m_fd);
    m_fd = -1;
  }
}

struct FlowEventTracker
{
  void Enable(bool on)
  {
    enabled = on;
  }

  void OnScheduled()
  {
    if (!enabled)
    {
      return;
    }
    uint32_t cur = pending.fetch_add(1, std::memory_order_relaxed) + 1;
    uint32_t prev = maxConcurrent.load(std::memory_order_relaxed);
    while (cur > prev &&
           !maxConcurrent.compare_exchange_weak(prev, cur, std::memory_order_relaxed))
    {
    }
  }

  void OnConsumed()
  {
    if (!enabled)
    {
      return;
    }
    uint32_t cur = pending.load(std::memory_order_relaxed);
    while (cur > 0 &&
           !pending.compare_exchange_weak(cur, cur - 1, std::memory_order_relaxed))
    {
    }
  }

  uint32_t Pending() const
  {
    return pending.load(std::memory_order_relaxed);
  }

  uint32_t MaxConcurrent() const
  {
    return maxConcurrent.load(std::memory_order_relaxed);
  }

  bool enabled{false};
  std::atomic<uint32_t> pending{0};
  std::atomic<uint32_t> maxConcurrent{0};
};

struct ProgressState
{
  ns3::Time next{ns3::Seconds(0.0)};
  ns3::Time step{ns3::Seconds(0.0)};
};

class Clock
{
public:
  static ns3::Time Now()
  {
    using namespace std::chrono;
    const auto now = steady_clock::now().time_since_epoch();
    return ns3::NanoSeconds(duration_cast<nanoseconds>(now).count());
  }
};

ProgressState g_progress;
ns3::Time g_progressStartWall = ns3::Seconds(0.0);
EventId g_progressBeatEvent;
LogFileGuard g_logGuard;
PidFileGuard g_pidGuard;
FlowEventTracker g_flowEventTracker;

void EmitProgress(double simSeconds, const char* suffix)
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss << "[PROGRESS] t=" << std::setprecision(1) << simSeconds << " s" << suffix;
  std::cout << oss.str() << '\n';
  std::cout.flush();
}

void ProgressBeat()
{
  const double t = Simulator::Now().GetSeconds();
  EmitProgress(t, "");
  std::cout << "[RL] overrides_applied=" << g_rlOverridePkts << "\n";
  std::cout << "[RL] pkts_forwarded_via_RL=" << g_rlPktUsed << "\n";
  std::cout << "[RL] pkts_forwarded_via_RL=" << g_rlPktUsed << "\n";
  if (g_progressBeatSec > 0.0)
  {
    g_progressBeatEvent = Simulator::Schedule(Seconds(g_progressBeatSec), &ProgressBeat);
  }
}

void StartProgressBeats()
{
  if (g_progressBeatSec <= 0.0)
  {
    return;
  }
  EmitProgress(0.0, "");
  g_progressBeatEvent = Simulator::Schedule(Seconds(g_progressBeatSec), &ProgressBeat);
}

void ProgressTick(double simEndSeconds)
{
  const ns3::Time tNow = ns3::Simulator::Now();
  const double nowSec = tNow.GetSeconds();
  const double pct = (simEndSeconds > 0.0)
                         ? std::min(100.0, 100.0 * nowSec / simEndSeconds)
                         : 0.0;
  const ns3::Time wallElapsed = Clock::Now() - g_progressStartWall;

  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss << "[PROGRESS] t=" << std::setprecision(1) << nowSec << "s ("
      << std::setprecision(1) << pct << "%), wall="
      << std::setprecision(1) << wallElapsed.GetSeconds() << "s";
  NS_LOG_UNCOND(oss.str());

  if (tNow < ns3::Seconds(simEndSeconds))
  {
    g_progress.next = tNow + g_progress.step;
    ns3::Simulator::Schedule(g_progress.step, &ProgressTick, simEndSeconds);
  }
}

void EmitProgressDone()
{
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss << "[PROGRESS] done t=" << std::setprecision(1) << Simulator::Now().GetSeconds() << " s";
  std::cout << oss.str() << '\n';
  std::cout.flush();
}

} // namespace

class FlowSender : public Application
{
public:
  FlowSender() = default;

  void Configure(Ptr<Socket> socket,
                 uint32_t flowId,
                 uint32_t pktSize,
                 double interPacket)
  {
    m_socket = socket;
    m_flowId = flowId;
    m_pktSize = pktSize;
    m_interPacket = interPacket;
  }

  void StartApplication() override
  {
    m_running = true;
    m_eventArmed = false;
    SendImmediate();
  }

  void StopApplication() override
  {
    m_running = false;
    if (m_eventArmed && m_sendEvent.IsPending())
    {
      Simulator::Cancel(m_sendEvent);
      g_flowEventTracker.OnConsumed();
    }
    if (g_debugFlowEvents)
    {
      std::cout << "[FLOWDEBUG] stop flowId=" << m_flowId
                << " eventArmed=" << m_eventArmed << "\n";
    }
    m_eventArmed = false;
    m_sendEvent = EventId();
    if (m_socket)
    {
      m_socket->Close();
    }
  }

  void DoDispose() override
  {
    if (m_eventArmed && m_sendEvent.IsPending())
    {
      Simulator::Cancel(m_sendEvent);
      g_flowEventTracker.OnConsumed();
      m_eventArmed = false;
      m_sendEvent = EventId();
    }
    if (g_debugFlowEvents)
    {
      std::cout << "[FLOWDEBUG] dispose flowId=" << m_flowId
                << " eventArmed=" << m_eventArmed << "\n";
    }
    m_socket = nullptr;
    Application::DoDispose();
  }

private:
  void ScheduleNext()
  {
    if (!m_running || m_eventArmed)
    {
      return;
    }
    m_sendEvent = Simulator::Schedule(Seconds(m_interPacket), &FlowSender::HandleScheduledSend, this);
    m_eventArmed = true;
    g_flowEventTracker.OnScheduled();
  }

  void HandleScheduledSend()
  {
    if (m_eventArmed)
    {
      m_eventArmed = false;
      g_flowEventTracker.OnConsumed();
    }
    m_sendEvent = EventId();
    SendImmediate();
  }

  void SendImmediate()
  {
    if (!m_running || m_socket == nullptr)
    {
      return;
    }

    Ptr<Packet> pkt = Create<Packet>(m_pktSize);
    TimestampTag ts;
    ts.SetTimestamp(Simulator::Now());
    pkt->AddPacketTag(ts);
    m_socket->Send(pkt);

    // Gate TX accounting to measurement window
    auto& stats = g_flowSummaries[m_flowId];
    const double now = Simulator::Now().GetSeconds();
    if (now >= g_measureStart)
    {
      stats.txPackets++;
      if (stats.firstTx < 0.0)
      {
        stats.firstTx = now;
      }
      stats.lastTx = now;
    }

    ScheduleNext();
  }

  Ptr<Socket> m_socket;
  uint32_t m_flowId{0};
  uint32_t m_pktSize{0};
  double m_interPacket{1.0};
  bool m_running{false};
  EventId m_sendEvent;
  bool m_eventArmed{false};
};

static void DumpQTables(const std::vector< Ptr<QTableRouting> >& qr,
                        const std::string& path)
{
  try
  {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  }
  catch (...)
  {
  }

  std::ostringstream oss;
  const std::vector<bool>* mask = (g_dumpQOnlyGs && !g_groundDestMask.empty())
                                     ? &g_groundDestMask
                                     : nullptr;
  for (uint32_t n = 0; n < qr.size(); ++n)
  {
    oss << "=== Node " << n << " ===\n";
    qr[n]->PrintQTable(oss, mask);
  }

  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (out.is_open())
  {
    out << oss.str();
  }
}

struct LinkRef {
  uint32_t a,b;
  Ipv4InterfaceContainer ifs;
  Ptr<RateErrorModel> aErr,bErr;
  Ptr<PointToPointChannel> ch;
  Time lastDelay;
};

// simple GS “gravity” set (same as OSPF harness)
struct Gs { std::string name; double x; double y; double pop; };
static std::vector<Gs> DefaultGs(double W, double H){
  return {
    {"NA-West",0.15*W,0.70*H,60},{"NA-East",0.35*W,0.70*H,90},{"LATAM-N",0.35*W,0.45*H,80},
    {"LATAM-S",0.40*W,0.25*H,60},{"EU-West",0.55*W,0.70*H,100},{"EU-East",0.65*W,0.70*H,90},
    {"AF-North",0.60*W,0.50*H,120},{"AF-South",0.62*W,0.25*H,70},{"ME",0.72*W,0.55*H,80},
    {"IN",0.80*W,0.55*H,140},{"SEA",0.88*W,0.45*H,180},{"EA",0.90*W,0.65*H,200},
  };
}
struct PairSampler {
  struct Item { uint32_t i,j; double cdf; };
  std::vector<Item> items; double total{0};
  static double Dist2D(const Gs& a, const Gs& b){ return std::sqrt(sqr(a.x-b.x)+sqr(a.y-b.y))+1.0; }
  PairSampler(const std::vector<Gs>& gs, double alpha, uint32_t useK){
    uint32_t K=std::min<uint32_t>(useK, gs.size());
    for(uint32_t i=0;i<K;++i) for(uint32_t j=0;j<K;++j) if(i!=j){
      double w = (gs[i].pop*gs[j].pop)/std::pow(Dist2D(gs[i],gs[j]), alpha);
      if (w <= 0) continue;
      total += w; items.push_back({i, j, total});
    }
    for(auto &it:items) it.cdf/=total;
  }
  std::pair<uint32_t,uint32_t> Sample(std::mt19937 &rng) const {
    std::uniform_real_distribution<double> U(0.0,1.0); double u=U(rng);
    size_t lo=0,hi=items.size(); while(lo+1<hi){ size_t m=(lo+hi)/2; (u<=items[m].cdf)? hi=m:lo=m; }
    return {items[lo].i, items[lo].j};
  }
};

// Range manager drives link up/down and delay; notifies QTableRouting
class RangeManager final : public Object {
public:
  static TypeId GetTypeId(void);
  RangeManager(const NodeContainer& nodes,
               std::vector<LinkRef>& links,
               double rangeMeters,
               double period,
               bool forceClean,
               double startDelay)
    : m_nodes(nodes), m_links(links), m_range(rangeMeters), m_period(period),
      m_forceCleanLinks(forceClean), m_startDelay(std::max(0.0, startDelay)) {}

  void Start()
  {
    if (m_forceCleanLinks)
    {
      for (auto &L : m_links)
      {
        L.aErr->SetRate(0.0);
        L.bErr->SetRate(0.0);
        QTableRouting::SetLinkUp(L.a, L.b, true);
      }
    }
    Simulator::Schedule(Seconds(m_startDelay), &RangeManager::Tick, this);
  }

  // Allow the harness to pass all QTableRouting instances for rebuild-on-change
  void SetQrArray(const std::vector< Ptr<QTableRouting> >& qr) { m_qr = &qr; }

  // Gym helpers
  uint32_t GetNumNodes() const { return m_nodes.GetN(); }
  std::vector<uint32_t> GetUpNeighbors(uint32_t nodeId) const {
    std::vector<uint32_t> ups;
    for (const auto &L : m_links) {
      if (L.a == nodeId) {
        if (QTableRouting::IsLinkUp(L.a, L.b)) ups.push_back(L.b);
      } else if (L.b == nodeId) {
        if (QTableRouting::IsLinkUp(L.a, L.b)) ups.push_back(L.a);
      }
    }
    std::sort(ups.begin(), ups.end());
    ups.erase(std::unique(ups.begin(), ups.end()), ups.end());
    return ups;
  }

private:
  static double Bucket(double v, double step) { return step > 0 ? step * std::round(v / step) : v; }

  void Tick() {
    const double now = Simulator::Now().GetSeconds();
    const double blackout = g_blackoutMs / 1000.0;
    bool topologyChanged = false;

    for (auto &L : m_links) {
      SatGeom geomA = ComputeSatGeom(L.a, now);
      SatGeom geomB = ComputeSatGeom(L.b, now);
      const double chordMeters = ChordDistance(geomA.ecef, geomB.ecef);
      const bool losOk = HasLineOfSight(geomA.ecef, geomB.ecef);
      const bool inRange = losOk && (chordMeters <= m_range);
      const uint64_t key = MakeKey(L.a, L.b);

      const bool isUp  = (L.aErr->GetRate() == 0.0);
      const bool mayUp = (m_nextUp.find(key) == m_nextUp.end() || now >= m_nextUp[key]);

      if (!inRange) {
        m_nextUp[key] = now + blackout;
        if (isUp) {
          L.aErr->SetRate(1.0); L.bErr->SetRate(1.0);
          QTableRouting::SetLinkUp(L.a, L.b, false);
          topologyChanged = true;
        }
      } else {
        const bool shouldBeUp = mayUp;

        if (shouldBeUp) {
          if (!isUp) {
            L.aErr->SetRate(0.0); L.bErr->SetRate(0.0);
            if (g_logRange)
            {
              std::cout << "[RM] link UP " << L.a << "<->" << L.b
                        << " at t=" << Simulator::Now().GetSeconds()
                        << "\n";
            }
            topologyChanged = true;
          }
          QTableRouting::SetLinkUp(L.a, L.b, true);
        } else if (isUp) {
          L.aErr->SetRate(1.0); L.bErr->SetRate(1.0);
          if (g_logRange)
          {
            std::cout << "[RM] link DOWN " << L.a << "<->" << L.b
                      << " at t=" << Simulator::Now().GetSeconds() << "\n";
          }
          QTableRouting::SetLinkUp(L.a, L.b, false);
          topologyChanged = true;
        }

        if (shouldBeUp && L.aErr->GetRate() == 0.0) {
          double propMs = std::max(kMinD, std::min(kMaxD, 1000.0 * chordMeters / kCEff));
          Time desired = MilliSeconds(Bucket(propMs, kBucket));
          if (desired != L.lastDelay) { L.ch->SetAttribute("Delay", TimeValue(desired)); L.lastDelay = desired; }
        }
      }
    }

    // Coalesce rebuilds into one event per tick (reduces transient holes)
    if (topologyChanged && m_qr) {
      if (!m_rebuildEv.IsPending()) {
        m_rebuildEv = Simulator::Schedule(Seconds(0.05), [this](){
          for (auto &rt : *m_qr) rt->KickRebuild();
        });
      }
    }

    Simulator::Schedule(Seconds(m_period), &RangeManager::Tick, this);
  }


 

  


  NodeContainer m_nodes;
  std::vector<LinkRef>& m_links;
  double m_range, m_period;
  std::unordered_map<uint64_t, double> m_nextUp;
  EventId m_rebuildEv;  // coalesced rebuild timer

  // Optional pointer to all QR instances (for rebuild on link flips)
  const std::vector< Ptr<QTableRouting> >* m_qr{nullptr};
  bool   m_forceCleanLinks{false};
  double m_startDelay{0.3};
};

NS_OBJECT_ENSURE_REGISTERED(RangeManager);

TypeId RangeManager::GetTypeId(void)
{
  static TypeId tid = TypeId("RangeManager")
    .SetParent<Object>()
    .SetGroupName("scratch");
  return tid;
}

#include "leo_rl_gym_env_impl.tcc"


int main(int argc, char* argv[]){
  try
  {
    LogComponentEnableAll(LOG_PREFIX_TIME);
    LogComponentEnableAll(LOG_PREFIX_FUNC);
    LogComponentEnableAll(LOG_PREFIX_NODE);
    CommandLine cmd;
    bool zeroRewardBeforeMeasure = false;
  // sim/plumbing flags (mirror OSPF-lite harness)
  cmd.AddValue("planes", "Number of orbital planes.", g_planes);
  cmd.AddValue("perPlane", "Satellites per plane.", g_perPlane);
  cmd.AddValue("wrap", "Torus neighbor wrap.", g_wrap);
  cmd.AddValue("spacingKm", "Grid spacing (km).", g_spacingKm);
  cmd.AddValue("simTime", "Simulation time (s).", g_simTime);
  cmd.AddValue("islRateMbps", "ISL rate (Mbps).", g_islRateMbps);
  cmd.AddValue("islDelayMs", "Seeded ISL delay (ms).", g_islDelayMs);
  cmd.AddValue("rangeKm", "ISL range threshold (km).", g_rangeKm);
  cmd.AddValue("checkPeriod", "Range check period (s).", g_checkPeriod);
  cmd.AddValue("blackoutMs", "Blackout after link re-enters range (ms).", g_blackoutMs);
  cmd.AddValue("rangeStart", "Initial range-manager tick delay (s).", g_rangeStart);
  cmd.AddValue("flows","Number of concurrent UDP flows.", g_flows);
  cmd.AddValue("pktSize","UDP payload size (bytes).", g_pktSize);
  cmd.AddValue("interPacket","UDP inter-packet interval (s).", g_interPacket);
  cmd.AddValue("runSeed","Flow sampler seed (std::mt19937).", g_runSeed);
  // Expose ns-3 RNG controls (match OSPF fix)
  cmd.AddValue("RngSeed","ns-3 master RNG seed.", g_rngSeed);
  cmd.AddValue("RngRun","ns-3 RNG run index.", g_rngRun);
  cmd.AddValue("numGs","How many GS regions from the default list.", g_numGs);
  cmd.AddValue("gravityAlpha","Gravity model alpha.", g_gravityAlpha);
  cmd.AddValue("useFlowMon","Install FlowMonitor instrumentation (default: false).", g_useFlowMon);
  cmd.AddValue("exportCsv","Write per-flow CSV to this path (empty to disable).", g_exportCsv);
  cmd.AddValue("exportSummary", "Write a tiny summary file (flags + DATA block).", g_exportSummary);
  cmd.AddValue("exportSummaryCsv","Append aggregate metrics to results/qrouting_summary.csv.", g_exportSummaryCsv);
  cmd.AddValue("forceSrcNode", "If >=0, force all flows' source node id.", g_forceSrcNode);
  cmd.AddValue("forceDstNode", "If >=0, force all flows' destination node id.", g_forceDstNode);
  cmd.AddValue("tag","Freeform tag echoed in logs.", g_tag);
  cmd.AddValue("progressIntervalSec","Heartbeat interval in sim seconds.", g_progressIntervalSec);
  cmd.AddValue("progressBeatSec","Emit [PROGRESS] heartbeat every this many sim-seconds.", g_progressBeatSec);
  cmd.AddValue("logEveryPkts","Emit Q-routing info logs at most once per this many packets.", g_logEveryPkts);
  cmd.AddValue("logEverySimSec","Emit Q-routing info logs at most once per this many sim-seconds.", g_logEverySimSec);
  cmd.AddValue("quietApps","Suppress verbose per-flow/per-packet app logs.", g_quietApps);
  cmd.AddValue("qrInfoLogs","Enable QTableRouting INFO logs (0/1).", g_qrInfoLogs);
  cmd.AddValue("enablePcap","Enable pcap capture on all ISLs.", g_enablePcap);
  cmd.AddValue("forceCleanLinks","Force ISL error rates to zero at install (diagnostic).", g_forceCleanLinks);
  cmd.AddValue("logRebuild","Enable verbose routing table rebuild logs.", g_logRebuild);
  cmd.AddValue("logAdj","Enable adjacency wiring logs.", g_logAdj);
  cmd.AddValue("logRange","Enable RangeManager link state logs.", g_logRange);
  cmd.AddValue("dumpQInit","Write Q-table snapshot at start (0/1).", g_dumpQInit);
  cmd.AddValue("dumpQFinal","Write Q-table snapshot at end (0/1).", g_dumpQFinal);
  cmd.AddValue("dumpQOnlyGs","Limit Q-table dumps to GS-anchor destinations (0/1).", g_dumpQOnlyGs);
  cmd.AddValue("dstOnlyGs","Track Q-table destinations only for GS anchors (0/1).", g_dstOnlyGs);
  cmd.AddValue("measureStart","Reset flow metrics at this sim time (s).", g_measureStart);
  cmd.AddValue("debugFlowEvents","Track FlowSender scheduled events (0/1).", g_debugFlowEvents);
  cmd.AddValue("force","Override pid/log guard rails (0/1).", g_forceOverride);
  cmd.AddValue("pidFile","PID guard file path (empty to disable).", g_pidFilePath);
  cmd.AddValue("logFile","Tee stdout/stderr into this log file (empty to disable).", g_logFileRequest);
  cmd.AddValue("queuePkts", "ISL egress queue depth (packets)", g_queuePkts);
  cmd.AddValue("satAltitudeKm", "Satellite orbital altitude (km).", g_satAltKm);
  cmd.AddValue("satInclDeg", "Satellite inclination (degrees).", g_satInclinationDeg);
  cmd.AddValue("walkerPhase", "Walker constellation phase F.", g_walkerPhaseF);
  // RL env flags
  cmd.AddValue("openGymPort", "ns3-gym port", g_openGymPort);
  cmd.AddValue("rlEnable", "Enable RL env", g_rlEnable);
  cmd.AddValue("waitAgent", "Wait for agent to attach before starting traffic", g_waitAgent);
  cmd.AddValue("rlDelta", "Decision period (s)", g_rlDelta);
  cmd.AddValue("rlK", "Max neighbor fanout", g_rlK);
  cmd.AddValue("rlTrainDst", "Destination node id to train for (-1 cycle)", g_rlTrainDst);
  cmd.AddValue("rlHopPenalty", "Hop penalty weight", g_rlHopPenalty);
  cmd.AddValue("rlQPenalty", "Q penalty weight", g_rlQPenalty);
  cmd.AddValue("rlDropPenalty", "Drop penalty weight", g_rlDropPenalty);
  cmd.AddValue("rlStartS", "Sim time (s) after which RL overrides may apply.", g_rlStartS);
  cmd.AddValue("logRl", "Verbose RL picker logs (0/1).", g_logRl);
  cmd.AddValue("teacher", "Teacher policy for RL (qtable|ospf|none)", g_teacherPolicy);
  cmd.AddValue("zeroRewardBeforeMeasure", "Zero RL reward before measureStart (default: false)", zeroRewardBeforeMeasure);
  // Q-table flags (now recognized)
  cmd.AddValue("qAlpha","Q-learning alpha.", g_qAlpha);
  cmd.AddValue("qGamma","Q-learning gamma.", g_qGamma);
  cmd.AddValue("qSeedHopMs","Per-hop prior delay used when seeding Q-tables (ms).", g_qSeedHopMs);
  cmd.AddValue("qUpdateStride",
               "Only update Q every Nth forwarded packet (>=1). Default: 1 (update every packet).",
               g_qUpdateStride);
  cmd.AddValue("qEps","Initial epsilon for epsilon-greedy (alias of qEpsStart).", g_qEpsStart);
  cmd.AddValue("qEpsStart","Initial epsilon for epsilon-greedy decay.", g_qEpsStart);
  cmd.AddValue("qEpsFinal","Final epsilon floor after decay.", g_qEpsFinal);
  cmd.AddValue("qEpsTau","Epsilon decay time constant (s).", g_qEpsTau);
  cmd.AddValue("qProbeInterval","Neighbor probe interval (s).", g_qProbeInterval);
  cmd.AddValue("qProbeFanout","Max neighbors to probe per tick.", g_qProbeFanout);
  cmd.AddValue("qPenaltyDropMs","Penalty window after drop (ms).", g_qPenaltyDropMs);
  cmd.AddValue("ctrlPort","UDP port for control/probes.", g_ctrlPort);
  // New Q-routing stability knobs
  cmd.AddValue("qProbeEwmaW","Weight of new RTT sample when updating neighbor cost EWMA (0.01..0.50).", g_qProbeEwmaW);
  cmd.AddValue("qSwitchHyst","Only switch next-hop if best candidate improves by this fraction (0..0.25).", g_qSwitchHyst);
  cmd.AddValue("qAlphaAfter","Alpha to use after measureStart (negative to disable).", g_qAlphaAfter);
  cmd.AddValue("qEpsAfter","Epsilon to use after measureStart (negative to disable).", g_qEpsAfter);
  // SP seeding + refresh + freeze
  cmd.AddValue("qSeedFromSpf", "Seed Q from shortest paths at init", g_qSeedFromSpf);
  cmd.AddValue("qSpSeedMs",    "ms per hop when seeding from SP",   g_qSpSeedMs);
  cmd.AddValue("qSpfRefresh",  "Rebuild SP tables every S seconds (0=never)", g_qSpfRefresh);
  cmd.AddValue("freezeAfterMeasure", "Set alpha=0 and eps=0 at measureStart", g_freezeAfterMeasure);
  cmd.AddValue("measureStart", "Seconds when measurement begins (used by freeze)", g_measureStart);

  cmd.Parse(argc, argv);

  g_rangeKm = std::max(0.0, g_rangeKm);
  g_satAltKm = std::max(0.0, g_satAltKm);
  g_satInclinationDeg = std::clamp(g_satInclinationDeg, 0.0, 180.0);

  g_zeroRewardBeforeMeasure = zeroRewardBeforeMeasure;

  g_qEpsStart = std::clamp(g_qEpsStart, 0.0, 1.0);
  g_qEpsFinal = std::clamp(g_qEpsFinal, 0.0, 1.0);
  if (g_qEpsFinal > g_qEpsStart)
  {
    g_qEpsFinal = g_qEpsStart;
  }
  g_qEpsTau = std::max(1e-6, g_qEpsTau);
  g_qSeedHopMs = std::max(0.1, g_qSeedHopMs);
  g_qUpdateStride = std::max<uint32_t>(1u, g_qUpdateStride);
  g_qProbeEwmaW = std::clamp(g_qProbeEwmaW, 0.01, 0.50);
  g_qSwitchHyst = std::clamp(g_qSwitchHyst, 0.0, 0.25);
  if (g_qAlphaAfter >= 0.0) g_qAlphaAfter = std::clamp(g_qAlphaAfter, 0.0, 1.0);
  if (g_qEpsAfter   >= 0.0) g_qEpsAfter   = std::clamp(g_qEpsAfter,   0.0, 1.0);

  g_flowEventTracker.Enable(g_debugFlowEvents);

  if (!g_pidGuard.Acquire(g_pidFilePath, g_forceOverride))
  {
    return 1;
  }

  if (!g_logGuard.Setup(g_logFileRequest, g_forceOverride))
  {
    return 1;
  }

  QTableRouting::SetRebuildLogging(g_logRebuild);
  g_flowSummaries.clear();
  g_flowSummaries.resize(g_flows);

  if (!g_quietApps)
  {
    LogComponentEnable("PacketSink", LOG_LEVEL_INFO);
  }
  if (g_qrInfoLogs)
  {
    LogComponentEnable("QTableRouting", LOG_LEVEL_INFO);
  }

  g_progressIntervalSec = std::max(0.0, g_progressIntervalSec);
  g_progressBeatSec = std::max(0.0, g_progressBeatSec);
  g_logEverySimSec = std::max(0.0, g_logEverySimSec);
  QTableRouting::SetLogPolicy(g_logEveryPkts, Seconds(g_logEverySimSec));

  ObjectFactory schedulerFactory;
  schedulerFactory.SetTypeId(g_schedulerType);
  Simulator::SetScheduler(schedulerFactory);

  if (!g_pidFilePath.empty())
  {
    std::cout << "[PID] locked " << g_pidFilePath << "\n";
  }
  if (!g_logFileRequest.empty() && g_logGuard.Active())
  {
    std::cout << "[LOG] using " << g_logGuard.Resolved() << "\n";
  }

  std::cout << "[SCHED] active=" << SchedulerAlias(g_schedulerType)
            << " type=" << g_schedulerType << "\n";
  std::cout << "[RUN] pid=" << static_cast<long>(getpid())
            << " runSeed=" << g_runSeed
            << " rngSeed=" << g_rngSeed
            << " rngRun=" << g_rngRun
            << " flows=" << g_flows
            << " progressBeatSec=" << g_progressBeatSec
            << " measureStart=" << g_measureStart
            << " dumpQInit=" << g_dumpQInit
            << " dumpQFinal=" << g_dumpQFinal
            << " dumpQOnlyGs=" << g_dumpQOnlyGs
            << " dstOnlyGs=" << g_dstOnlyGs
            << " debugFlowEvents=" << g_debugFlowEvents
            << " tag=" << g_tag
            << "\n";

  std::cout << "[QR-CONFIG] alpha=" << g_qAlpha
            << " gamma=" << g_qGamma
            << " epsStart=" << g_qEpsStart
            << " epsFinal=" << g_qEpsFinal
            << " epsTau=" << g_qEpsTau
            << " seedHopMs=" << g_qSeedHopMs
            << " spSeed=" << g_qSeedFromSpf
            << " spSeedMs=" << g_qSpSeedMs
            << " spfRefreshS=" << g_qSpfRefresh
            << " freezeAfterMeasure=" << g_freezeAfterMeasure
            << " probeInterval=" << g_qProbeInterval
            << " probeFanout=" << g_qProbeFanout
            << " penaltyDropMs=" << g_qPenaltyDropMs
            << " ctrlPort=" << g_ctrlPort
            << " tag=" << g_tag
            << "\n";

  {
    std::ostringstream qs;
    qs << g_queuePkts << "p";
    Config::SetDefault("ns3::DropTailQueue<Packet>::MaxSize", StringValue(qs.str()));
  }

  // ---------- topology ----------
  const uint32_t N = g_planes * g_perPlane;
  static std::vector<Ipv4Address> loopbacks;
  loopbacks.resize(N);
  // Apply ns-3 RNG seed/run from CLI (do this BEFORE creating models/streams)
  RngSeedManager::SetSeed(g_rngSeed); RngSeedManager::SetRun(g_rngRun);
  NodeContainer sats; sats.Create(N);

  double spacing = g_spacingKm*1000.0; // field dims
  auto idx=[&](uint32_t r,uint32_t c){return r*g_perPlane+c;};

  MobilityHelper mob; Ptr<ListPositionAllocator> pos = CreateObject<ListPositionAllocator>();
  for(uint32_t r=0;r<g_planes;++r) for(uint32_t c=0;c<g_perPlane;++c) pos->Add(Vector(c*spacing, r*spacing, 0));
  mob.SetPositionAllocator(pos);
  mob.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
  mob.Install(sats);
  for(uint32_t r=0;r<g_planes;++r) for(uint32_t c=0;c<g_perPlane;++c){
    auto m = sats.Get(idx(r,c))->GetObject<ConstantVelocityMobilityModel>();
    double vx = (c%2==0) ? 200.0 : -200.0; double vy = (r%2==0) ? -200.0 : 200.0; m->SetVelocity(Vector(vx,vy,0));
  }

  InternetStackHelper internet; internet.Install(sats);

  // QTableRouting per node
  std::vector< Ptr<QTableRouting> > qr(N);
  for (uint32_t n = 0; n < N; ++n){
    Ptr<Ipv4> ipv4 = sats.Get(n)->GetObject<Ipv4>();
    qr[n] = CreateObject<QTableRouting>();
    ipv4->SetRoutingProtocol(qr[n]);
    qr[n]->Configure(g_qAlpha, g_qGamma, g_qEpsStart,
                     Seconds(g_qProbeInterval), g_qProbeFanout,
                     MilliSeconds(g_qPenaltyDropMs), g_ctrlPort, nullptr);
    qr[n]->SetSeedHopMs(g_qSeedHopMs);
    qr[n]->SetEpsSchedule(g_qEpsStart, g_qEpsFinal, g_qEpsTau);
    qr[n]->SetUpdateStride(g_qUpdateStride);
    // SP seeding + refresh + freeze
    qr[n]->EnableSeedFromSpf(g_qSeedFromSpf);
    qr[n]->SetSpSeedMs(g_qSpSeedMs);
    qr[n]->SetSpfRefresh(g_qSpfRefresh);
    qr[n]->SetMeasureStart(g_measureStart);
    qr[n]->EnableFreezeAfterMeasure(g_freezeAfterMeasure);
  }

  // Global stability knobs
  QTableRouting::SetProbeEwmaW(g_qProbeEwmaW);
  QTableRouting::SetSwitchHyst(g_qSwitchHyst);
  QTableRouting::SetLogPolicy(g_logEveryPkts, Seconds(g_logEverySimSec));
  // Honor update stride globally via existing per-instance setting above

  // Reset RL packet usage counter at start of run
  g_rlPktUsed = 0;

  // Schedule evaluation regime at measureStart, if provided and not freezing
  if (!g_freezeAfterMeasure && g_qAlphaAfter >= 0.0 && g_qEpsAfter >= 0.0)
  {
    Simulator::Schedule(Seconds(g_measureStart), [](){
      QTableRouting::SetAlphaEps(g_qAlphaAfter, g_qEpsAfter);
    });
  }

  // grid edges (4-neighbor torus)
  std::vector<std::pair<uint32_t,uint32_t>> edges;
  for(uint32_t r=0;r<g_planes;++r){
    for(uint32_t c=0;c<g_perPlane;++c){
      uint32_t u=idx(r,c);
      if(c+1<g_perPlane || g_wrap){ uint32_t v=idx(r,(c+1)%g_perPlane); if(u<v) edges.push_back({u,v}); }
      if(r+1<g_planes || g_wrap){ uint32_t v=idx((r+1)%g_planes,c);     if(u<v) edges.push_back({u,v}); }
    }
  }

  PointToPointHelper p2p;
  p2p.SetDeviceAttribute("DataRate", StringValue(std::to_string((uint32_t)g_islRateMbps)+"Mbps"));
  p2p.SetChannelAttribute("Delay", StringValue(std::to_string(g_islDelayMs)+"ms"));
  Ipv4AddressHelper ip; ip.SetBase("10.0.0.0","255.255.255.252");

  std::vector<LinkRef> links; links.reserve(edges.size());
  for (auto [a,b] : edges) {
    NetDeviceContainer devs = p2p.Install(NodeContainer(sats.Get(a), sats.Get(b)));

    Ptr<RateErrorModel> ea = CreateObject<RateErrorModel>(), eb = CreateObject<RateErrorModel>();
    ea->SetUnit(RateErrorModel::ERROR_UNIT_PACKET);
    eb->SetUnit(RateErrorModel::ERROR_UNIT_PACKET);
    const double initialRate = g_forceCleanLinks ? 0.0 : 1.0;
    if (g_forceCleanLinks)
    {
      if (g_logRange)
      {
        std::cout << "[DIAG] forcing clean link " << a << "<->" << b << "\n";
      }
      QTableRouting::SetLinkUp(a, b, true);
    }
    ea->SetRate(initialRate);
    eb->SetRate(initialRate);
    devs.Get(0)->SetAttribute("ReceiveErrorModel", PointerValue(ea));
    devs.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(eb));

    Ipv4InterfaceContainer ifs = ip.Assign(devs); ip.NewNetwork();
    Ptr<PointToPointChannel> ch = DynamicCast<PointToPointChannel>(devs.Get(0)->GetChannel());
    Time seededDelay = MilliSeconds(g_islDelayMs); ch->SetAttribute("Delay", TimeValue(seededDelay));

    // Wire adjacencies into QTableRouting
    Ptr<Ipv4> iA = sats.Get(a)->GetObject<Ipv4>();
    Ptr<Ipv4> iB = sats.Get(b)->GetObject<Ipv4>();
    int32_t ifA = iA->GetInterfaceForDevice(devs.Get(0));
    int32_t ifB = iB->GetInterfaceForDevice(devs.Get(1));
    if (ifA < 0 || ifB < 0)
    {
      NS_LOG_WARN("Skipping adjacency " << a << "<->" << b
                   << " because interface lookup failed (ifA=" << ifA
                   << ", ifB=" << ifB << ")");
      std::cout << "[WARN] bad iface index for " << a << "<->" << b
                << " (ifA=" << ifA << ", ifB=" << ifB << ")\n";
    }
    else
    {
      Ipv4Address aAddr = ifs.GetAddress(0);
      Ipv4Address bAddr = ifs.GetAddress(1);
      qr[a]->AddAdjacency(b, aAddr, bAddr, static_cast<uint32_t>(ifA));
      qr[b]->AddAdjacency(a, bAddr, aAddr, static_cast<uint32_t>(ifB));
      if (g_logAdj)
      {
        std::cout << "[ADJ] " << a << "<->" << b
                  << " ifA=" << ifA << " ifB=" << ifB
                  << " addrA=" << aAddr << " addrB=" << bAddr
                  << "\n";
      }
    }


    links.push_back({a,b,ifs,ea,eb,ch,seededDelay});
  }
  if (g_enablePcap)
  {
    p2p.EnablePcapAll("qr", true);
  }

  auto dumpRoutingTables = [&qr, N](const std::string& label) {
    if (!g_logRebuild)
    {
      return;
    }
    std::cout << "[ROUTES] " << label << "\n";
    Ptr<OutputStreamWrapper> ow = Create<OutputStreamWrapper>(&std::cout);
    for (uint32_t n = 0; n < N; ++n)
    {
      qr[n]->PrintRoutingTable(ow, Time::S);
    }
  };

  // Add per-node /32 loopback-ish addresses and register them with routing
  for (uint32_t n = 0; n < N; ++n) {
    Ptr<Ipv4> ip4 = sats.Get(n)->GetObject<Ipv4>();
    uint32_t iface = std::min(1u, std::max(0u, ip4->GetNInterfaces() - 1));
    Ipv4InterfaceAddress rid(
      Ipv4Address(("192.168." + std::to_string(n/256) + "." + std::to_string(n%256)).c_str()),
      Ipv4Mask("255.255.255.255"));
    ip4->AddAddress(iface, rid);
    ip4->SetUp(iface);

    uint32_t nAddrs = ip4->GetNAddresses(iface);
    uint32_t addrIdx = (nAddrs > 0) ? (nAddrs - 1) : 0;
    Ipv4Address my32 = ip4->GetAddress(iface, addrIdx).GetLocal();
    loopbacks[n] = my32;
    qr[n]->RegisterNodeAddress(n, my32);
    if (g_logAdj)
    {
      std::cout << "[LOOPBACK] n=" << n << " ip=" << my32 << "\n";
    }
  }

  // Build forwarding tables once (AFTER node /32s are registered)
  for (uint32_t n = 0; n < N; ++n) { qr[n]->KickRebuild(); }
  dumpRoutingTables("initial rebuild");

  const double fieldWidth = spacing * (g_perPlane - 1);
  const double fieldHeight = spacing * (g_planes - 1);
  auto gsList = DefaultGs(fieldWidth, fieldHeight);
  g_numGs = std::max<uint32_t>(1u, std::min<uint32_t>(g_numGs, static_cast<uint32_t>(gsList.size())));
  gsList.resize(g_numGs);

  auto nearestSat = [&](double x, double y) -> uint32_t {
    double best = std::numeric_limits<double>::infinity();
    uint32_t bestIdx = 0u;
    for (uint32_t n = 0; n < N; ++n)
    {
      auto m = sats.Get(n)->GetObject<MobilityModel>();
      Vector pos = m->GetPosition();
      double dx = pos.x - x;
      double dy = pos.y - y;
      double d = std::sqrt(dx * dx + dy * dy);
      if (d < best)
      {
        best = d;
        bestIdx = n;
      }
    }
    return bestIdx;
  };

  g_groundDestMask.assign(N, false);
  for (const auto& gs : gsList)
  {
    uint32_t anchor = nearestSat(gs.x, gs.y);
    if (anchor < N)
    {
      g_groundDestMask[anchor] = true;
    }
  }

  std::vector<bool> routerMask;
  if (g_dstOnlyGs)
  {
    routerMask = g_groundDestMask;
  }
  for (auto& inst : qr)
  {
    inst->SetDestinationMask(routerMask);
  }


 
if (g_dumpQInit)
{
  DumpQTables(qr, "results/qrouting_qtable_init.txt");
}


  // Range manager: link on/off + delay updates + blackout
  Ptr<RangeManager> rm = CreateObject<RangeManager>(sats, links, g_rangeKm*1000.0,
                                                   g_checkPeriod, g_forceCleanLinks,
                                                   g_rangeStart);
  rm->SetQrArray(qr);

  rm->Start();
  // Rebuild once more shortly after first RM tick
  Simulator::Schedule(Seconds(std::max(0.0, g_rangeStart + 0.30)), [=](){
  for (uint32_t n = 0; n < N; ++n) { qr[n]->KickRebuild(); }
  dumpRoutingTables("post RM rebuild");
});

  QTableRouting::SetRlEnabled(g_rlEnable);
  if (!g_rlEnable)
  {
    QTableRouting::InstallRlPicker(nullptr);
  }

  // ----- Optional: wire minimal ns3-gym env (handshake) -----
  if (g_rlEnable && !qr.empty())
  {
    std::cout << "[RL][BUILD] QR-OG-server READY\n";
    Ptr<OpenGymInterface> og = OpenGymInterface::Get(g_openGymPort);

    std::cout << "[DBG] (1) Creating env\n";
    Ptr<LeoRoutingGymEnv> env = CreateObject<LeoRoutingGymEnv>(
        Seconds(g_rlDelta),
        g_rlK,
        qr,
        rm,
        g_rlTrainDst,
        g_rlHopPenalty,
        g_rlQPenalty,
        g_rlDropPenalty,
        static_cast<double>(g_queuePkts),
        g_measureStart,
        g_simTime,
        &g_groundDestMask);

    std::cout << "[DBG] (2) Attaching interface\n";
    env->SetOpenGymInterface(og);
    env->SetTeacherPolicy(g_teacherPolicy);
    if (g_forceSrcNode >= 0)
    {
      env->SetFocusNode(g_forceSrcNode);
    }

    std::cout << "[DBG] (3) Calling og->Init()\n";
    // In this build, Init() sets up and binds the REP socket; no Start() API.
    og->Init();
    std::cout << "[RL] gym_start port=" << g_openGymPort << "\n";

    std::cout << "[DBG] (4) Starting env\n";
    env->Start();
    // Optionally block starting traffic until agent handshake is observed
    if (g_waitAgent) {
      std::cout << "[ENV] waiting for agent attach...\n";
      uint32_t tries = 0;
      while (!LeoRoutingGymEnv::AgentAttached() && tries < 400) { // ~40s @100ms
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        ++tries;
      }
      if (!LeoRoutingGymEnv::AgentAttached()) {
        std::cout << "[ENV][WARN] agent did not attach within timeout; continuing.\n";
      } else {
        std::cout << "[ENV] agent attached.\n";
      }
    }
  }


  // ------------- gravity traffic (match OSPF exactly) -------------
  // Flow placement / any stdlib randomness keeps using runSeed
  std::mt19937 rng(g_runSeed);
  PairSampler sampler(gsList, g_gravityAlpha, g_numGs);

  std::cout << "[TRAFFIC] flows=" << g_flows
            << " pktSize=" << g_pktSize
            << " interPacket=" << g_interPacket
            << " pps=" << (1.0 / g_interPacket)
            << "\n";

  uint16_t basePort = 9000;
  double startT = 3.0;
  uint32_t flowsMade = 0;
  for (; flowsMade < g_flows; ++flowsMade) {
    auto [gi, gj] = sampler.Sample(rng);
    uint32_t sIdx = nearestSat(gsList[gi].x, gsList[gi].y);
    uint32_t dIdx = nearestSat(gsList[gj].x, gsList[gj].y);
    // Optional forcing for bring-up: override with fixed node ids
    if (g_forceSrcNode >= 0 && static_cast<uint32_t>(g_forceSrcNode) < N) {
      sIdx = static_cast<uint32_t>(g_forceSrcNode);
    }
    if (g_forceDstNode >= 0 && static_cast<uint32_t>(g_forceDstNode) < N) {
      dIdx = static_cast<uint32_t>(g_forceDstNode);
    }
    if (sIdx == dIdx) {
      dIdx = (dIdx + 1) % N;
    }

    uint16_t dstPort = basePort + flowsMade;
    Ipv4Address dst = loopbacks[dIdx];
    // Mark whether this flow's destination is a GS-anchored node
    bool dstIsGs = (dIdx < g_groundDestMask.size()) ? g_groundDestMask[dIdx] : false;
    if (flowsMade < g_flowSummaries.size())
    {
      g_flowSummaries[flowsMade].dstIsGs = dstIsGs;
    }
    // Print the first few flows regardless of quietApps to verify GS marking
    if (!g_quietApps || flowsMade < 5)
    {
      std::cout << "[FLOWSETUP] f=" << flowsMade
                << " srcId=" << sIdx
                << " dstId=" << dIdx
                << " dstIsGs=" << (dstIsGs ? 1 : 0)
                << " dst=" << dst
                << " port=" << dstPort
                << " pps=" << (1.0 / g_interPacket)
                << "\n";
    }

    PacketSinkHelper sink("ns3::UdpSocketFactory",
                          InetSocketAddress(Ipv4Address::GetAny(), dstPort));
    auto sApps = sink.Install(sats.Get(dIdx));
    sApps.Start(Seconds(1.0));
    sApps.Stop(Seconds(g_simTime + 1.0));
    Ptr<PacketSink> sinkPtr = (sApps.GetN() > 0)
                                ? DynamicCast<PacketSink>(sApps.Get(0))
                                : nullptr;

    const uint32_t flowId = flowsMade;
    if (sinkPtr)
    {
      sinkPtr->TraceConnectWithoutContext("Rx", MakeBoundCallback(&RecordSinkRx, flowId));
    }

    Ptr<Node> srcNode = sats.Get(sIdx);
    Ptr<Socket> sock = Socket::CreateSocket(srcNode, UdpSocketFactory::GetTypeId());
    sock->Bind();
    sock->Connect(InetSocketAddress(dst, dstPort));

    Ptr<FlowSender> sender = CreateObject<FlowSender>();
    sender->Configure(sock, flowId, g_pktSize, g_interPacket);
    srcNode->AddApplication(sender);

    double flowStart = startT + 0.05 * flowsMade;
    sender->SetStartTime(Seconds(flowStart));
    sender->SetStopTime(Seconds(g_simTime));
  }
  if (!g_quietApps)
  {
    std::cout << "[FLOWSETUP] requested=" << g_flows
              << " created=" << flowsMade
              << "\n";
  }


  // ------------- run + metrics -------------
  static FlowMonitorHelper fmHelper;
  Ptr<FlowMonitor> mon;
  if (g_useFlowMon)
  {
    fmHelper.SetMonitorAttribute("DelayBinWidth", DoubleValue(0.005));
    fmHelper.SetMonitorAttribute("JitterBinWidth", DoubleValue(0.005));
    fmHelper.SetMonitorAttribute("PacketSizeBinWidth", DoubleValue(256.0));
    mon = fmHelper.InstallAll();
  }

  if (g_simTime > 0.0 && g_progressIntervalSec > 0.0) {
    g_progress.step = Seconds(g_progressIntervalSec);
    g_progress.next = g_progress.step;
    g_progressStartWall = Clock::Now();
    Simulator::Schedule(g_progress.next, &ProgressTick, g_simTime);
  }

  if (g_simTime > 0.0)
  {
    StartProgressBeats();
  }

  if (g_measureStart > 0.0 && g_measureStart < g_simTime)
  {
    // Only log the measurement start; accounting is timestamp-gated now
    Simulator::Schedule(Seconds(g_measureStart), []() {
      QrCtrlResetCounters();
      std::ostringstream oss;
      oss.setf(std::ios::fixed);
      oss << "[MEASURE] start at t=" << std::setprecision(3)
          << Simulator::Now().GetSeconds() << " s";
      std::cout << oss.str() << '\n';
    });
    // Snapshot a few seconds into the window to confirm counters advance
    if (g_measureStart + 5.0 < g_simTime)
    {
      Simulator::Schedule(Seconds(g_measureStart + 5.0), [](){
        uint64_t rxSum = 0, txSum = 0;
        for (const auto& fs : g_flowSummaries)
        {
          rxSum += fs.rxPackets;
          txSum += fs.txPackets;
        }
        std::cout << "[MEASURE][SNAP] +5s rxSum=" << rxSum << " txSum=" << txSum << "\n";
      });
    }
  }
  else
  {
    QrCtrlResetCounters();
  }

  Simulator::Stop(Seconds(g_simTime));
  Simulator::Run();

  if (g_simTime > 0.0)
  {
    EmitProgressDone();
  }

  if (g_useFlowMon && mon)
  {
    mon->CheckForLostPackets();
  }

if (g_dumpQFinal)
{
  DumpQTables(qr, "results/qrouting_qtable_final_" + g_tag + ".txt");
}


  uint64_t totalTx = 0;
  uint64_t totalRx = 0;
  double totalDelaySec = 0.0;
  double sumThrMbps = 0.0;
  uint32_t countedFlows = 0;

  auto computeDuration = [](const FlowSummary& st) {
    double rxSpan = (st.firstRx >= 0.0 && st.lastRx > st.firstRx) ? (st.lastRx - st.firstRx) : 0.0;
    double txSpan = (st.firstTx >= 0.0 && st.lastTx > st.firstTx) ? (st.lastTx - st.firstTx) : 0.0;
    double dur = (rxSpan > 0.0) ? rxSpan : txSpan;
    if (dur <= 0.0 && st.firstTx >= 0.0)
    {
      dur = std::max(0.0, g_simTime - st.firstTx);
    }
    return dur;
  };

  for (const auto& st : g_flowSummaries)
  {
    if (g_dstOnlyGs && !st.dstIsGs)
    {
      continue; // filter flows not anchored at GS when requested
    }
    totalTx += st.txPackets;
    totalRx += st.rxPackets;
    totalDelaySec += st.delaySum;

    if (st.rxPackets > 0)
    {
      double dur = computeDuration(st);
      if (dur > 0.0 && st.rxBytes > 0)
      {
        sumThrMbps += (static_cast<double>(st.rxBytes) * 8.0) / (dur * 1e6);
      }
      ++countedFlows;
    }
  }

  const uint64_t lost = (totalTx >= totalRx) ? (totalTx - totalRx) : 0;
  const double pdr = (totalTx > 0) ? static_cast<double>(totalRx) / static_cast<double>(totalTx) : 0.0;
  const double avgDelaySec = (totalRx > 0) ? (totalDelaySec / static_cast<double>(totalRx)) : 0.0;
  const double avgThrMbpsPerFlow = (countedFlows > 0)
                                     ? (sumThrMbps / static_cast<double>(countedFlows))
                                     : 0.0;

  // ---- Aggregate QTable control overhead across all nodes ----
  uint64_t ctrlPktsTx = 0, ctrlPktsRx = 0;
  uint64_t ctrlPayloadTx = 0, ctrlPayloadRx = 0;
  uint64_t ctrlWireTx = 0, ctrlWireRx = 0;
  uint64_t probeTx = 0, probeRx = 0, probePayTx = 0, probePayRx = 0;
  uint64_t probeWireTx = 0, probeWireRx = 0;
  uint64_t packTx = 0, packRx = 0, packPayTx = 0, packPayRx = 0;
  uint64_t packWireTx = 0, packWireRx = 0;
  uint64_t fbTx = 0, fbRx = 0, fbPayTx = 0, fbPayRx = 0;
  uint64_t fbWireTx = 0, fbWireRx = 0;

  for (const auto& r : qr)
  {
    if (!r)
    {
      continue;
    }
    ctrlPktsTx    += r->GetCtrlPktsTx();
    ctrlPktsRx    += r->GetCtrlPktsRx();
    ctrlPayloadTx += r->GetCtrlPayloadTx();
    ctrlPayloadRx += r->GetCtrlPayloadRx();
    ctrlWireTx    += r->GetCtrlWireTx();
    ctrlWireRx    += r->GetCtrlWireRx();

    probeTx      += r->GetProbePktsTx();
    probeRx      += r->GetProbePktsRx();
    probePayTx   += r->GetProbePayloadTx();
    probePayRx   += r->GetProbePayloadRx();
    probeWireTx  += r->GetProbeWireTx();
    probeWireRx  += r->GetProbeWireRx();

    packTx       += r->GetProbeAckPktsTx();
    packRx       += r->GetProbeAckPktsRx();
    packPayTx    += r->GetProbeAckPayloadTx();
    packPayRx    += r->GetProbeAckPayloadRx();
    packWireTx   += r->GetProbeAckWireTx();
    packWireRx   += r->GetProbeAckWireRx();

    fbTx         += r->GetFeedbackPktsTx();
    fbRx         += r->GetFeedbackPktsRx();
    fbPayTx      += r->GetFeedbackPayloadTx();
    fbPayRx      += r->GetFeedbackPayloadRx();
    fbWireTx     += r->GetFeedbackWireTx();
    fbWireRx     += r->GetFeedbackWireRx();
  }

  std::cout << "[CTRL][QR] tx_pkts=" << ctrlPktsTx
            << " tx_payload_B=" << ctrlPayloadTx
            << " tx_wire_B=" << ctrlWireTx
            << " rx_pkts=" << ctrlPktsRx
            << " rx_payload_B=" << ctrlPayloadRx
            << " rx_wire_B=" << ctrlWireRx
            << " | probe_tx_pkts=" << probeTx
            << " pack_tx_pkts=" << packTx
            << " fb_tx_pkts=" << fbTx
            << '\n';

  const double ctrlWindowSec = std::max(0.0, g_simTime - g_measureStart);
  QrCtrlPrintProxy(ctrlWindowSec);

  // Optional CSV: extend existing export logic if/when a summary row is added.

  if (g_debugFlowEvents)
  {
    std::cout << "[FLOWDEBUG] maxScheduled=" << g_flowEventTracker.MaxConcurrent()
              << " pending=" << g_flowEventTracker.Pending()
              << " flows=" << g_flows << "\n";
  }

  if (!g_exportCsv.empty())
  {
    try
    {
      std::filesystem::path p(g_exportCsv);
      if (p.has_parent_path())
      {
        std::filesystem::create_directories(p.parent_path());
      }
    }
    catch (...) { /* ignore */ }

    std::ofstream out(g_exportCsv, std::ios::out | std::ios::trunc);
    if (out)
    {
      out << "tag,flowId,txPackets,rxPackets,delayMeanMs,throughputKbps\n";
      out.setf(std::ios::fixed, std::ios::floatfield);
      for (uint32_t fid = 0; fid < g_flowSummaries.size(); ++fid)
      {
        const auto& st = g_flowSummaries[fid];
        double delayMeanMs = (st.rxPackets > 0)
                               ? (st.delaySum / static_cast<double>(st.rxPackets) * 1000.0)
                               : 0.0;
        double dur = computeDuration(st);
        double thrKbps = (dur > 0.0 && st.rxBytes > 0)
                           ? (static_cast<double>(st.rxBytes) * 8.0) / (dur * 1000.0)
                           : 0.0;

        out << g_tag << ',' << fid << ',' << st.txPackets << ',' << st.rxPackets << ','
            << std::setprecision(3) << delayMeanMs << ','
            << std::setprecision(3) << thrKbps << '\n';
      }
    }
  }

  const uint32_t nodes = N;
  const uint32_t isls  = static_cast<uint32_t>(edges.size());

  // print-once guard
  static bool kPrinted = false;
  if (!kPrinted) {
    kPrinted = true;

    std::ostringstream dataBlock;
    dataBlock << std::fixed << std::setprecision(6);
    dataBlock << "\n" << g_planes << "x" << g_perPlane << " " << g_tag << "\n";
    dataBlock << "==== DATA METRICS ====\n";
    dataBlock << "Nodes: " << nodes
              << "  ISLs: " << isls
              << "  Range(km): " << g_rangeKm
              << "  Probe(s): " << g_qProbeInterval
              << "  Flows: " << g_flows << "\n";
    dataBlock << "TxPkts: " << totalTx
              << "  RxPkts: " << totalRx
              << "  Lost: "  << lost
              << "  PDR: "   << std::setprecision(5) << pdr
              << "  AvgDelay(s): " << std::setprecision(7) << avgDelaySec
              << "  AvgThroughput(Mbps/flow): " << std::setprecision(5) << avgThrMbpsPerFlow
              << "\n";
    dataBlock << "[RL] overrides_applied=" << g_rlOverridePkts << "\n";
    dataBlock << "[RL] pkts_forwarded_via_RL=" << g_rlPktUsed << "\n";
    std::cout << dataBlock.str();
    // Suppress machine-readable CSV line previously printed for harness
    
    if (g_exportSummaryCsv)
    {
      try
      {
        const char* sumPath = "results/qrouting_summary.csv";
        std::filesystem::create_directories(std::filesystem::path(sumPath).parent_path());
        bool writeHeader = !std::filesystem::exists(sumPath) || std::filesystem::file_size(sumPath) == 0;
        std::ofstream s(sumPath, std::ios::out | std::ios::app);
        if (writeHeader)
        {
          s << "tag,planes,perPlane,flows,rangeKm,isls,tx,rx,lost,pdr,avgDelaySec,avgMbpsPerFlow\n";
        }
        s << g_tag << "," << g_planes << "," << g_perPlane << "," << g_flows << ","
          << g_rangeKm << "," << isls << ","
          << totalTx << "," << totalRx << "," << lost << ","
          << std::setprecision(6) << pdr << ","
          << std::setprecision(7) << avgDelaySec << ","
          << std::setprecision(6) << avgThrMbpsPerFlow << "\n";
        s.close();
      }
      catch (...) { /* ignore */ }
    }

    // tiny summary file if requested
    if (!g_exportSummary.empty()) {
      try {
        std::filesystem::path p(g_exportSummary);
        if (p.has_parent_path()) {
          std::filesystem::create_directories(p.parent_path());
        }
        std::ofstream out(g_exportSummary, std::ios::out | std::ios::trunc);
        if (out.is_open()) {
          out << "# FLAGS\n" << DumpFlagsQ() << "\n";
          out << dataBlock.str();
        }
      } catch (...) {
        std::cerr << "[WARN] failed to write exportSummary to " << g_exportSummary << "\n";
      }
    }
  }

  // Suppress per-node [DROP] summaries

    Simulator::Destroy();
    return 0;
  }
  catch (const std::exception& ex)
  {
    std::cerr << "[FATAL] exception: " << ex.what() << "\n";
  }
  catch (...)
  {
    std::cerr << "[FATAL] unknown exception\n";
  }

  Simulator::Destroy();
  return 1;
}
// Optional: force all flows to use a fixed src/dst node id (for RL bring-up proof)
