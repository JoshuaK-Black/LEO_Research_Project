// scratch/leo_iridium_ospf.cc (packetized-control variant)
//
// LEO OSPF-lite baseline (Iridium 6x11) with distance-based delays and gravity traffic
// + Minimal packetized control plane (Hello + LSU + LSACK) and event-driven SPF.
//
// Build:  ./ns3 build
// Run:    ./ns3 run "scratch/leo_iridium_ospf --planes=6 --perPlane=11 --simTime=300 --flows=200 --interPacket=0.015 --exportCsv=results.csv"
//
// Notes:
// - This file is a drop-in replacement for your existing scratch/leo_iridium_ospf.cc.
// - Implements Steps 1–3 requested: (1) UDP control-plane application with Hello/LSU/Ack,
//   (2) LSAs originated from RangeManager link up/down events, (3) event-driven SPF with delay/hold.
// - It keeps a periodic SPF watchdog (every spfInterval seconds) as a safety net.
// - Control-plane is intentionally lightweight (not a full OSPF implementation): reliable LSU via simple ACK & retry.

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/random-variable-stream.h"
#include "ns3/packet-sink.h"
#include "ns3/traffic-control-module.h"
#include "ns3/timestamp-tag.h"
#include "ns3/udp-socket-factory.h"
#include "vis_publisher.h"
#include "ns3/simulator.h"
#include "ns3/node-list.h"
#include "ns3/mobility-model.h"

using namespace ns3;

static VisPublisher* g_vis = nullptr;

// ------------------------------ CLI knobs -----------------------------------
static double g_blackoutMs = 200.0;
static uint32_t g_planes = 6;
static uint32_t g_perPlane = 11;
static bool g_wrap = true;
static double g_spacingKm = 5000;
static double g_simTime = 180.0;
static double g_islRateMbps = 40.0;
static double g_islDelayMs = 5.0;
static double g_rangeKm = 6500.0;
static double g_checkPeriod = 0.10;
static double g_spfInterval = 2.0;  // watchdog interval (still kept)
static double g_spfJitterMax = 0.2;
static double g_minSpfHold = 1.0;  // legacy periodic hold
static uint32_t g_flows = 200;
static uint16_t g_pktSize = 1500;
static double g_interPacket = 0.015;
static uint32_t g_runSeed = 1;         // flow-sampler seed (back-compat)
static uint32_t g_rngSeed = 1;         // ns-3 master RNG seed (new)
static uint32_t g_rngRun = 1;          // ns-3 RNG run number (new)

// Gravity model
static uint32_t g_numGs = 12;
static double g_gravityAlpha = 2.0;
static std::string g_exportCsv = "results.csv";
// --- small summary output ---
static std::string g_exportSummary = "";
// Flow selection for metrics (match QR behavior)
static bool g_dstOnlyGs = true; // count only GS-anchored destinations by default
static std::vector<bool> g_groundDestMask;
static double g_satAltKm = 780.0;
static double g_satInclinationDeg = 86.4;
static double g_walkerPhaseF = 2.0;
static bool g_emitSatStates = false;

static constexpr double kEarthRadiusM = 6371000.0;
static constexpr double kEarthRotationRate = 7.2921159e-5;     // rad/s
static constexpr double kEarthMu = 3.986004418e14;             // m^3/s^2

struct SatGeom
{
  ns3::Vector ecef;
  double      latDeg{0.0};
  double      lonDeg{0.0};
  double      altM{0.0};
};

static SatGeom ComputeSatGeom(uint32_t satId, double timeSec)
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

  out.ecef = ns3::Vector(x, y, z);
  const double rxy = std::sqrt(x * x + y * y);
  const double rnorm = std::sqrt(x * x + y * y + z * z);
  out.latDeg = std::atan2(z, rxy) * 180.0 / M_PI;
  out.lonDeg = std::atan2(y, x) * 180.0 / M_PI;
  if (out.lonDeg > 180.0) out.lonDeg -= 360.0;
  if (out.lonDeg < -180.0) out.lonDeg += 360.0;
  out.altM = rnorm - kEarthRadiusM;
  return out;
}

static inline double ChordDistance(const ns3::Vector& a, const ns3::Vector& b)
{
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

static inline bool HasLineOfSight(const ns3::Vector& a, const ns3::Vector& b)
{
  const ns3::Vector diff(b.x - a.x, b.y - a.y, b.z - a.z);
  const double ab2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
  if (ab2 <= 0.0)
  {
    return true;
  }
  double t = -(a.x * diff.x + a.y * diff.y + a.z * diff.z) / ab2;
  if (t < 0.0) t = 0.0;
  else if (t > 1.0) t = 1.0;
  const ns3::Vector closest(a.x + t * diff.x, a.y + t * diff.y, a.z + t * diff.z);
  const double dist2 = closest.x * closest.x + closest.y * closest.y + closest.z * closest.z;
  return dist2 >= kEarthRadiusM * kEarthRadiusM;
}

static void EmitSatStates()
{
  if (!g_vis)
  {
    return;
  }
  const double t = ns3::Simulator::Now().GetSeconds();
  const uint32_t count = ns3::NodeList::GetNNodes();
  for (uint32_t i = 0; i < count; ++i)
  {
    SatGeom geom = ComputeSatGeom(i, t);
    g_vis->EmitSatState(t, static_cast<int>(i), geom.latDeg, geom.lonDeg, geom.altM);
  }
  ns3::Simulator::Schedule(ns3::Seconds(0.5), &EmitSatStates);
}

// Distance-based delay model
static double g_cEff = 3.0e8;  // m/s
static double g_minDelayMs = 1.0;
static double g_maxDelayMs = 50.0;
static double g_delayBucketMs = 1.0;

// Control-plane (new)
static uint16_t g_ctrlPort = 8899;      // UDP port for control-plane
static double g_helloInterval = 10.0;   // s (default 10s)
static double g_deadInterval = 40.0;    // s (default 40s)
static double g_helloJitterFrac = 0.2;  // +/- 20% jitter on hello timer
static uint32_t g_helloMissK = 3;       // declare DOWN after K missed hellos
static double g_spfDelay = 0.25;        // s (after LSDB change)
static double g_spfHold = 1.0;          // s (min gap between SPFs)
static double g_minLsArrival = 1.0;     // s (drop same-LSA if arriving too fast)
static double g_lsaMinInterval = 1.0;   // s (min origination interval per LSA)
static double g_lsaBackoffMax = 5.0;    // s (cap for LSA origination backoff)
static uint32_t g_lsaFanout = 2;        // max neighbor LSAs per send tick (seed fanout)
static double g_lsaRetransmit = 0.5;    // s (basic retry timer)

// OSPF realism / overhead knobs
static uint32_t g_helloBytes = 48;
static uint32_t g_lsaBytes = 64;
static double g_lsaRefreshSec = 30.0;
static double g_neighborScanSec = 0.5;
static bool g_fullFlood = false;
static bool g_ackRetransMix = true;
static double g_ackDelayBase = 0.03;    // s (ack base delay)
static double g_ackDelayJitter = 0.02;  // s (ack jitter)
static double g_lsaMaxAge = 3600.0;     // s (purge stale LSA records)

// Control-plane shaper
static double g_ctrlRateKbps = 128.0;   // kbps budget
static uint32_t g_ctrlBucket = 4096;    // bytes

// SPF backoff controls
static double g_spfBackoffInit = 0.25;  // s
static double g_spfBackoffMax = 5.0;    // s
static double g_spfBackoffDecay = 0.5;  // s per successful SPF

// Route install delay (+jitter)
static double g_fibInstallBase = 0.02;  // s
static double g_fibInstallJitter = 0.02; // s

// Harness compatibility knobs (logging + measurement)
static double g_measureStart = 60.0;  // was 0.0
static double g_progressBeatSec = 0.0;
static bool g_quietApps = true;
static bool g_enablePcap = false;
static uint32_t g_logEveryPkts = 0;
static double g_logEverySimSec = 0.0;
static std::string g_pidFile;
static std::string g_logFile;
static std::string g_tag;

// ISL egress queue depth (packets)
static uint32_t g_queuePkts = 500;  // baseline

static void ProgressBeat();

// ------------------------------ Helpers -------------------------------------
static inline double sqr(double x) { return x * x; }

// Dump current CLI/runtime flags in key=value lines for tiny summary
static std::string DumpFlagsOspfLite() {
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
  s << "emitSatStates=" << (g_emitSatStates ? 1 : 0) << "\n";

  // Traffic
  s << "flows=" << g_flows << "\n";
  s << "pktSize=" << g_pktSize << "\n";
  s << "interPacket=" << g_interPacket << "\n";

  // OSPF control
  s << "helloInterval=" << g_helloInterval << "\n";
  s << "deadInterval=" << g_deadInterval << "\n";
  s << "helloJitter=" << g_helloJitterFrac << "\n";
  s << "helloMissK=" << g_helloMissK << "\n";
  s << "lsaMinInterval=" << g_lsaMinInterval << "\n";
  s << "minLsArrival=" << g_minLsArrival << "\n";
  s << "lsaBackoffMax=" << g_lsaBackoffMax << "\n";
  s << "lsaFanout=" << g_lsaFanout << "\n";
  s << "lsaRetransmit=" << g_lsaRetransmit << "\n";
  s << "lsaRefresh=" << g_lsaRefreshSec << "\n";
  s << "lsaMaxAge=" << g_lsaMaxAge << "\n";
  s << "fullFlood=" << (g_fullFlood ? 1 : 0) << "\n";
  s << "ackRetrans=" << (g_ackRetransMix ? 1 : 0) << "\n";
  s << "ackDelayBase=" << g_ackDelayBase << "\n";
  s << "ackDelayJitter=" << g_ackDelayJitter << "\n";

  // SPF & FIB
  s << "spfDelay=" << g_spfDelay << "\n";
  s << "spfHold=" << g_spfHold << "\n";
  s << "spfBackoffInit=" << g_spfBackoffInit << "\n";
  s << "spfBackoffMax=" << g_spfBackoffMax << "\n";
  s << "spfBackoffDecay=" << g_spfBackoffDecay << "\n";
  s << "fibInstallBase=" << g_fibInstallBase << "\n";
  s << "fibInstallJitter=" << g_fibInstallJitter << "\n";

  // Control-plane shaper
  s << "ctrlRateKbps=" << g_ctrlRateKbps << "\n";
  s << "ctrlBucket=" << g_ctrlBucket << "\n";

  // Misc
  s << "measureStart=" << g_measureStart << "\n";
  s << "tag=" << g_tag << "\n";
  return s.str();
}

struct LinkRef {
  uint32_t a, b;
  Ipv4InterfaceContainer ifs;
  Ptr<RateErrorModel> aErr, bErr;
  Ptr<PointToPointChannel> ch;
  Time lastDelay;
};

static uint64_t MakeKey(uint32_t u, uint32_t v) {
  if (u > v) std::swap(u, v);
  return (static_cast<uint64_t>(u) << 32) | v;
}

static void ProgressBeat() {
  double now = Simulator::Now().GetSeconds();
  auto oldFlags = std::cout.flags();
  auto oldPrec = std::cout.precision();
  std::cout << "[PROGRESS] t=" << std::fixed << std::setprecision(3) << now;
  if (!g_tag.empty()) {
    std::cout << " tag=" << g_tag;
  }
  std::cout << "\n";
  std::cout.flags(oldFlags);
  std::cout.precision(oldPrec);
  if (g_progressBeatSec > 0.0 && now + g_progressBeatSec <= g_simTime + 1e-6) {
    Simulator::Schedule(Seconds(g_progressBeatSec), &ProgressBeat);
  }
}

struct FlowSummary {
  uint64_t txPackets{0};
  uint64_t rxPackets{0};
  uint64_t rxBytes{0};
  double firstTx{-1.0};
  double lastTx{-1.0};
  double firstRx{-1.0};
  double lastRx{-1.0};
  double delaySum{0.0};
  bool   dstIsGs{false};
};

static std::vector<FlowSummary> g_flowSummaries;

static void ResetFlowSummaries() {
  for (auto& st : g_flowSummaries) {
    st.txPackets = 0;
    st.rxPackets = 0;
    st.rxBytes = 0;
    st.firstTx = -1.0;
    st.lastTx = -1.0;
    st.firstRx = -1.0;
    st.lastRx = -1.0;
    st.delaySum = 0.0;
  }
}

static void RecordSinkRx(uint32_t flowId, Ptr<const Packet> pkt, const Address&) {
  if (flowId >= g_flowSummaries.size()) {
    return;
  }
  const double now = Simulator::Now().GetSeconds();
  if (now < g_measureStart) {
    return; // arrival-time gating for measurement window
  }
  auto& stats = g_flowSummaries[flowId];
  stats.rxPackets++;
  stats.rxBytes += pkt->GetSize();
  if (stats.firstRx < 0.0) {
    stats.firstRx = now;
  }
  stats.lastRx = now;

  Ptr<Packet> copy = pkt->Copy();
  TimestampTag ts;
  if (copy->PeekPacketTag(ts)) {
    Time delta = Simulator::Now() - ts.GetTimestamp();
    stats.delaySum += delta.GetSeconds();
  }
}

struct FlowEventTracker {
  void Enable(bool on) { enabled = on; }

  void OnScheduled() {
    if (!enabled) {
      return;
    }
    uint32_t cur = pending.fetch_add(1, std::memory_order_relaxed) + 1;
    uint32_t prev = maxConcurrent.load(std::memory_order_relaxed);
    while (cur > prev && !maxConcurrent.compare_exchange_weak(prev, cur, std::memory_order_relaxed)) {
    }
  }

  void OnConsumed() {
    if (!enabled) {
      return;
    }
    uint32_t cur = pending.load(std::memory_order_relaxed);
    while (cur > 0 && !pending.compare_exchange_weak(cur, cur - 1, std::memory_order_relaxed)) {
    }
  }

  uint32_t Pending() const { return pending.load(std::memory_order_relaxed); }

  uint32_t MaxConcurrent() const { return maxConcurrent.load(std::memory_order_relaxed); }

  bool enabled{false};
  std::atomic<uint32_t> pending{0};
  std::atomic<uint32_t> maxConcurrent{0};
};

static FlowEventTracker g_flowEventTracker;
static bool g_debugFlowEvents = false;

class FlowSender : public Application {
 public:
  FlowSender() = default;

  void Configure(Ptr<Socket> socket, uint32_t flowId, uint32_t pktSize, double interPacket) {
    m_socket = socket;
    m_flowId = flowId;
    m_pktSize = pktSize;
    m_interPacket = interPacket;
  }

  void StartApplication() override {
    m_running = true;
    m_eventArmed = false;
    SendImmediate();
  }

  void StopApplication() override {
    m_running = false;
    if (m_eventArmed && m_sendEvent.IsPending()) {
      Simulator::Cancel(m_sendEvent);
      g_flowEventTracker.OnConsumed();
    }
    if (g_debugFlowEvents) {
      std::cout << "[FLOWDEBUG] stop flowId=" << m_flowId << " eventArmed=" << m_eventArmed << "\n";
    }
    m_eventArmed = false;
    m_sendEvent = EventId();
    if (m_socket) {
      m_socket->Close();
    }
  }

  void DoDispose() override {
    if (m_eventArmed && m_sendEvent.IsPending()) {
      Simulator::Cancel(m_sendEvent);
      g_flowEventTracker.OnConsumed();
      m_eventArmed = false;
      m_sendEvent = EventId();
    }
    if (g_debugFlowEvents) {
      std::cout << "[FLOWDEBUG] dispose flowId=" << m_flowId << " eventArmed=" << m_eventArmed << "\n";
    }
    m_socket = nullptr;
    Application::DoDispose();
  }

 private:
  void ScheduleNext() {
    if (!m_running || m_eventArmed) {
      return;
    }
    m_sendEvent = Simulator::Schedule(Seconds(m_interPacket), &FlowSender::HandleScheduledSend, this);
    m_eventArmed = true;
    g_flowEventTracker.OnScheduled();
  }

  void HandleScheduledSend() {
    if (m_eventArmed) {
      m_eventArmed = false;
      g_flowEventTracker.OnConsumed();
    }
    m_sendEvent = EventId();
    SendImmediate();
  }

  void SendImmediate() {
    if (!m_running || m_socket == nullptr) {
      return;
    }

    Ptr<Packet> pkt = Create<Packet>(m_pktSize);
    TimestampTag ts;
    ts.SetTimestamp(Simulator::Now());
    pkt->AddPacketTag(ts);
    m_socket->Send(pkt);

    auto& stats = g_flowSummaries[m_flowId];
    const double now = Simulator::Now().GetSeconds();
    if (now >= g_measureStart) {
      stats.txPackets++;
      if (stats.firstTx < 0.0) {
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

static void ResetMeasureCounters() {
  // Log-only marker; counters are gated by time now
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss << "[MEASURE] start at t=" << std::setprecision(3) << Simulator::Now().GetSeconds() << " s";
  std::cout << oss.str() << '\n';
}


// ===================== OSPF-lite (routing core) =============================
// OspfLite: holds adjacency state and drives SPF recomputation.
class OspfLite final : public Object {
 public:
  OspfLite(NodeContainer nodes,
           const std::vector<LinkRef>& links,
           double spfInterval,
           double spfJitterMax,
           double minSpfHold)
      : m_nodes(nodes),
        m_links(links),
        m_spfInterval(spfInterval),
        m_spfJitterMax(spfJitterMax),
        m_minSpfHold(minSpfHold),
        m_dirty(true),
        m_spfRuns(0),
        m_linkUpEvents(0),
        m_linkDownEvents(0),
        m_routeChanges(0),
        m_lastSpf(Seconds(0.0)),
        m_spfPending(false),
        m_spfBackoff(g_spfBackoffInit) {}

  void Start() { ScheduleSpf(Seconds(1.0)); }
  void PreparePrevMaps(uint32_t N) { m_prevNextHop.assign(N, {}); }
  void SetLoopbacks(const std::vector<Ipv4Address>& addrs) {
    m_loopbacks = addrs;
    m_dirty = true;
  }

  void SetLinkUp(uint32_t a, uint32_t b, bool up) {
    auto key = MakeKey(a, b);
    bool prev = m_linkUp[key];
    if (prev != up) {
      m_linkUp[key] = up;
      if (up)
        ++m_linkUpEvents;
      else
        ++m_linkDownEvents;
      m_dirty = true;
    }
  }

  // Event-driven SPF request (new): schedule SPF with delay/hold + adaptive backoff
  void RequestSpf() {
    Time now = Simulator::Now();
    if (m_spfPending) return;  // already scheduled due to an earlier change

    double since = (now - m_lastSpf).GetSeconds();
    double hold = std::max(0.0, g_spfHold - since);
    double wait = hold + g_spfDelay + m_spfBackoff;
    m_spfPending = true;
    Simulator::Schedule(Seconds(wait), &OspfLite::RunSpf, this);
    m_spfBackoff = std::min(m_spfBackoff * 2.0, g_spfBackoffMax);
  }

  uint64_t GetSpfRuns() const { return m_spfRuns; }
  uint64_t GetLinkUpEvents() const { return m_linkUpEvents; }
  uint64_t GetLinkDownEvents() const { return m_linkDownEvents; }
  uint64_t GetRouteChanges() const { return m_routeChanges; }

  // Expose adjacency helper for the control plane
  const std::unordered_map<uint64_t, bool>& GetLinkBitmap() const { return m_linkUp; }

 private:
  struct NhInfo {
    Ipv4Address nh;
    uint32_t oif;
  };
  using Graph = std::vector<std::vector<std::pair<uint32_t, double>>>;

  Graph BuildGraph() {
    Graph g(m_nodes.GetN());
    for (const auto& L : m_links) {
      auto key = MakeKey(L.a, L.b);
      if (m_linkUp[key]) {
        g[L.a].push_back({L.b, 1.0});
        g[L.b].push_back({L.a, 1.0});
      }
    }
    return g;
  }

  std::vector<int> Dijkstra(const Graph& g, uint32_t src) {
    const double INF = 1e18;
    std::vector<double> dist(g.size(), INF);
    std::vector<int> par(g.size(), -1);
    using P = std::pair<double, uint32_t>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
    dist[src] = 0;
    pq.push({0, src});
    while (!pq.empty()) {
      auto [d, u] = pq.top();
      pq.pop();
      if (d > dist[u]) continue;
      for (auto [v, w] : g[u]) {
        if (dist[v] > d + w) {
          dist[v] = d + w;
          par[v] = (int)u;
          pq.push({dist[v], v});
        }
      }
    }
    return par;
  }

  void InstallRoutesFromTree(uint32_t src,
                             const std::vector<int>& parent,
                             const std::vector<Ipv4Address>& loopbacks,
                             const std::vector<std::map<uint32_t, NhInfo>>& nhInfo) {
    Ptr<Ipv4> ip4 = m_nodes.Get(src)->GetObject<Ipv4>();
    Ipv4StaticRoutingHelper rh;
    Ptr<Ipv4StaticRouting> rt = rh.GetStaticRouting(ip4);
    for (int i = int(rt->GetNRoutes()) - 1; i >= 0; --i) rt->RemoveRoute(uint32_t(i));

    auto& prev = m_prevNextHop[src];
    std::map<uint32_t, uint32_t> current;

    for (uint32_t dst = 0; dst < m_nodes.GetN(); ++dst) {
      if (dst == src) continue;
      int u = dst, p = parent[dst];
      if (p < 0) continue;
      while (p >= 0 && (uint32_t)p != src) {
        u = p;
        p = parent[u];
      }
      uint32_t nh = (p < 0) ? (uint32_t)u : (uint32_t)u;

      auto it = nhInfo[src].find(nh);
      if (it == nhInfo[src].end()) continue;
      const NhInfo& info = it->second;

      rt->AddHostRouteTo(loopbacks[dst], info.nh, info.oif);
      current[dst] = nh;
      if (g_vis) {
        static uint64_t hopSeq = 0;
        if ((hopSeq++ % 50) == 0) {
          double t = Simulator::Now().GetSeconds();
          g_vis->EmitHop(t,
                         static_cast<int>(src),
                         static_cast<int>(dst),
                         static_cast<int>(nh),
                         true);
        }
      }
    }

    for (auto& kv : current) {
      auto it = prev.find(kv.first);
      if (it == prev.end() || it->second != kv.second) ++m_routeChanges;
    }
    prev.swap(current);
  }

  // RunSpf: recompute routes when topology changes or watchdog fires.
  void RunSpf() {
    m_spfPending = false;
    Time now = Simulator::Now();
    if (!m_dirty && (now - m_lastSpf).GetSeconds() < m_minSpfHold) {
      ScheduleSpf(Seconds(m_spfInterval));
      return;
    }
    m_dirty = false;
    m_lastSpf = now;
    ++m_spfRuns;
    // relax backoff slightly after a successful SPF
    m_spfBackoff = std::max(g_spfBackoffInit, m_spfBackoff - g_spfBackoffDecay);

    auto g = BuildGraph();

    std::vector<Ipv4Address> loop = m_loopbacks;
    if (loop.size() != m_nodes.GetN()) {
      loop.assign(m_nodes.GetN(), Ipv4Address("0.0.0.0"));
    }
    std::vector<std::map<uint32_t, NhInfo>> nhInfo(m_nodes.GetN());

    for (const auto& L : m_links) {
      Ptr<Ipv4> a4 = m_nodes.Get(L.a)->GetObject<Ipv4>();
      Ptr<Ipv4> b4 = m_nodes.Get(L.b)->GetObject<Ipv4>();

      if (loop[L.a] == Ipv4Address("0.0.0.0")) {
        loop[L.a] = a4->GetAddress(std::min(1u, a4->GetNInterfaces() - 1), 0).GetLocal();
      }
      if (loop[L.b] == Ipv4Address("0.0.0.0")) {
        loop[L.b] = b4->GetAddress(std::min(1u, b4->GetNInterfaces() - 1), 0).GetLocal();
      }

      Ipv4Address aAddr = L.ifs.GetAddress(0);
      Ipv4Address bAddr = L.ifs.GetAddress(1);

      uint32_t aOif = a4->GetInterfaceForAddress(aAddr);
      uint32_t bOif = b4->GetInterfaceForAddress(bAddr);

      // Guard: if lookup failed, skip this neighbor to avoid invalid OIF
      if (aOif == uint32_t(-1) || bOif == uint32_t(-1)) {
        nhInfo[L.a].erase(L.b);
        nhInfo[L.b].erase(L.a);
        continue;
      }

      if (m_linkUp.count(MakeKey(L.a, L.b)) && m_linkUp.at(MakeKey(L.a, L.b))) {
        nhInfo[L.a][L.b] = NhInfo{bAddr, aOif};
        nhInfo[L.b][L.a] = NhInfo{aAddr, bOif};
      } else {
        nhInfo[L.a].erase(L.b);
        nhInfo[L.b].erase(L.a);
      }
    }

    for (uint32_t s = 0; s < m_nodes.GetN(); ++s) {
      auto tree = Dijkstra(g, s);
      // emulate RIB-to-FIB programming time with small delay + jitter
      Ptr<UniformRandomVariable> U = CreateObject<UniformRandomVariable>();
      double jitter = U->GetValue(0.0, g_fibInstallJitter);
      double wait = std::max(0.0, g_fibInstallBase + jitter);
      Simulator::Schedule(Seconds(wait), &OspfLite::InstallRoutesFromTree, this, s, tree, loop, nhInfo);
    }

    ScheduleSpf(Seconds(m_spfInterval));  // watchdog keeps running
  }

  void ScheduleSpf(Time base) {
    double j = 0.0;
    if (m_spfJitterMax > 0) {
      Ptr<UniformRandomVariable> rv = CreateObject<UniformRandomVariable>();
      j = rv->GetValue(0.0, m_spfJitterMax);
    }
    Simulator::Schedule(base + Seconds(j), &OspfLite::RunSpf, this);
  }

  NodeContainer m_nodes;
  std::vector<LinkRef> m_links;
  double m_spfInterval, m_spfJitterMax, m_minSpfHold;
  std::unordered_map<uint64_t, bool> m_linkUp;
  bool m_dirty;
  uint64_t m_spfRuns, m_linkUpEvents, m_linkDownEvents, m_routeChanges;
  Time m_lastSpf;
  bool m_spfPending;
  double m_spfBackoff;  // adaptive backoff under churn
  std::vector<Ipv4Address> m_loopbacks;
  std::vector<std::map<uint32_t, uint32_t>> m_prevNextHop;
};

// =================== Minimal Control-Plane Agent ============================
// OspfLiteAgent: lightweight Hello/LSA agent per satellite.
class OspfLiteAgent : public Application {
 public:
  enum PktType : uint8_t { HELLO = 1,
                           LSU = 2,
                           LSACK = 3 };

  struct LsaEdge {
    uint32_t u, v;
    bool up;
  };
  struct LsaKey {
    uint32_t u, v;
    bool operator==(const LsaKey& o) const { return (u == o.u && v == o.v) || (u == o.v && v == o.u); }
  };
  struct LsaKeyHash {
    size_t operator()(LsaKey const& k) const {
      uint64_t x = MakeKey(k.u, k.v);
      return std::hash<uint64_t>()(x);
    }
  };
  struct LsaRec {
    uint32_t seq;
    double lastUpdate;
    bool up;
  };

  OspfLiteAgent() {}

  void Configure(Ptr<OspfLite> core,
                 NodeContainer nodes,
                 const std::vector<LinkRef>& links,
                 uint32_t selfId,
                 uint16_t port) {
    m_core = core;
    m_nodes = nodes;
    m_links = links;
    m_self = selfId;
    m_port = port;

    // Build neighbor address map from LinkRef
    for (const auto& L : links) {
      if (L.a == m_self) {
        m_neighbors[L.b] = L.ifs.GetAddress(1);  // a->b uses b's addr
      } else if (L.b == m_self) {
        m_neighbors[L.a] = L.ifs.GetAddress(0);  // b->a uses a's addr
      }
    }
  }

  void StartApplication() override {
    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
    m_sock = Socket::CreateSocket(GetNode(), tid);
    InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), m_port);
    m_sock->Bind(local);
    m_sock->SetRecvCallback(MakeCallback(&OspfLiteAgent::Recv, this));

    // Periodic Hello timer
    m_helloEv = Simulator::Schedule(Seconds(0.5), &OspfLiteAgent::SendHello, this);
    // Retransmit scan
    m_retxEv = Simulator::Schedule(Seconds(g_lsaRetransmit), &OspfLiteAgent::RetxTick, this);
    if (g_lsaRefreshSec > 0) {
      m_refreshEv = Simulator::Schedule(Seconds(g_lsaRefreshSec), &OspfLiteAgent::RefreshTick, this);
    }
    if (g_neighborScanSec > 0) {
      m_deadEv = Simulator::Schedule(Seconds(g_neighborScanSec), &OspfLiteAgent::NeighborScan, this);
    }
    // Age scan for MaxAge purge
    m_ageEv = Simulator::Schedule(Seconds(5.0), &OspfLiteAgent::AgeTick, this);
  }

  void StopApplication() override {
    if (m_sock) {
      m_sock->SetRecvCallback(MakeNullCallback<void, Ptr<Socket>>());
      m_sock->Close();
      m_sock = nullptr;
    }
    if (m_helloEv.IsPending()) m_helloEv.Cancel();
    if (m_retxEv.IsPending()) m_retxEv.Cancel();
    if (m_refreshEv.IsPending()) m_refreshEv.Cancel();
    if (m_deadEv.IsPending()) m_deadEv.Cancel();
    if (m_ageEv.IsPending()) m_ageEv.Cancel();
  }

  // NotifyLinkChange: originate LSAs when an adjacent link toggles.
  void NotifyLinkChange(uint32_t a, uint32_t b, bool up) {
    // Originates LSA only if this node is one endpoint
    if (a != m_self && b != m_self) return;
    LsaKey key{std::min(a, b), std::max(a, b)};
    double now = Simulator::Now().GetSeconds();
    auto& rec = m_lsdb[key];
    // per-LSA exponential backoff to throttle flaps
    BackoffState& bo = m_backoff[key];
    if (bo.cur <= 0.0) bo.cur = std::max(0.5, g_lsaMinInterval);
    double since = now - rec.lastUpdate;
    if (rec.seq == 0 || rec.up != up || since >= g_lsaMinInterval) {
      // schedule a (re-)origin after backoff; coalesce multiple triggers
      if (bo.ev.IsPending()) {
        Simulator::Cancel(bo.ev);
      }
      double wait = bo.cur;
      bo.max = g_lsaBackoffMax;
      bo.cur = std::min(bo.cur * 2.0, bo.max);
      bo.ev = Simulator::Schedule(Seconds(wait), [this, key, up]() {
        auto& rec2 = m_lsdb[key];
        rec2.seq += 1;
        rec2.lastUpdate = Simulator::Now().GetSeconds();
        rec2.up = up;
        if (g_fullFlood) {
          FloodLsuAll(key.u, key.v, up);
        } else {
          FloodLsu(key, rec2.seq, up);
        }
        m_core->RequestSpf();
      });
    }
  }

  uint64_t GetCtrlBytesTx() const { return m_ctrlBytesTx; }

 private:
  enum NbrState { DOWN = 0,
                  INIT = 1,
                  TWOWAY = 2,
                  FULL = 3 };

  struct BackoffState {
    double cur{0.0};
    double max{g_lsaBackoffMax};
    EventId ev;  // pending origination event for this LSA key
  };

  // SendHello: periodically announce liveness to neighbors.
  void SendHello() {
    for (auto& kv : m_neighbors) {
      uint32_t nei = kv.first;
      SendCtrlPkt(HELLO, nei, 0, 0, m_self, nei, g_helloBytes);
    }
    // jitter next hello by +/- g_helloJitterFrac
    Ptr<UniformRandomVariable> U = CreateObject<UniformRandomVariable>();
    double frac = U->GetValue(-g_helloJitterFrac, g_helloJitterFrac);
    double next = std::max(0.2, g_helloInterval * (1.0 + frac));
    m_helloEv = Simulator::Schedule(Seconds(next), &OspfLiteAgent::SendHello, this);
  }

  // RetxTick: resend pending LSUs to neighbors awaiting acknowledgments.
  void RetxTick() {
    double now = Simulator::Now().GetSeconds();
    for (auto& kv : m_outbox) {
      auto& state = kv.second;
      if (state.pending.empty()) continue;
      if (now - state.lastSend < g_lsaRetransmit) continue;

      if (g_ackRetransMix) {
        uint32_t sent = 0;
        for (auto nei : state.pending) {
          SendCtrlPkt(LSU, nei, state.seq, state.up ? 1 : 0, state.u, state.v, g_lsaBytes);
          ++sent;
          if (!g_fullFlood && sent >= g_lsaFanout) break;
        }
      } else {
        uint32_t budget = g_lsaFanout;
        for (auto nit = state.pending.begin(); nit != state.pending.end() && budget > 0;) {
          SendCtrlPkt(LSU, *nit, state.seq, state.up ? 1 : 0, state.u, state.v, g_lsaBytes);
          nit = state.pending.erase(nit);
          budget--;
        }
      }
      state.lastSend = now;
    }
    m_retxEv = Simulator::Schedule(Seconds(g_lsaRetransmit), &OspfLiteAgent::RetxTick, this);
  }

  // FloodLsu: seed a fresh LSU across up neighbors and track pending acks.
  void FloodLsu(LsaKey key, uint32_t seq, bool up) {
    // Create outbox entry
    OutState& st = m_outbox[key];
    st.seq = seq;
    st.up = up;
    st.u = key.u;
    st.v = key.v;
    st.lastSend = 0.0;
    // Pending = all neighbors currently up
    st.pending.clear();
    auto upMap = m_core->GetLinkBitmap();
    for (auto& kv : m_neighbors) {
      uint32_t nei = kv.first;
      if (upMap.find(MakeKey(m_self, nei)) != upMap.end() && upMap.at(MakeKey(m_self, nei))) {
        st.pending.insert(nei);
      }
    }
    // Immediate send to a few neighbors, rest will go via retransmit pacing
    uint32_t budget = g_lsaFanout;
    for (auto it = st.pending.begin(); it != st.pending.end() && budget > 0;) {
      SendCtrlPkt(LSU, *it, seq, up ? 1 : 0, key.u, key.v, g_lsaBytes);
      if (g_ackRetransMix) {
        ++it;
      } else {
        it = st.pending.erase(it);
      }
      budget--;
    }
    st.lastSend = Simulator::Now().GetSeconds();
    if (!g_ackRetransMix) {
      // For legacy mode we only track yet-to-send neighbors.
      for (auto it = st.pending.begin(); it != st.pending.end();) {
        if (upMap.find(MakeKey(m_self, *it)) == upMap.end() || !upMap.at(MakeKey(m_self, *it))) {
          it = st.pending.erase(it);
        } else {
          ++it;
        }
      }
    }
  }

  void FloodLsuAll(uint32_t u, uint32_t v, bool up) {
    double now = Simulator::Now().GetSeconds();
    LsaKey key{std::min(u, v), std::max(u, v)};
    uint32_t seq = ++m_seqCounter;

    auto& rec = m_lsdb[key];
    rec.seq = seq;
    rec.lastUpdate = now;
    rec.up = up;

    OutState& st = m_outbox[key];
    st.seq = seq;
    st.up = up;
    st.u = key.u;
    st.v = key.v;
    st.lastSend = 0.0;
    st.pending.clear();

    for (auto& kv : m_neighbors) {
      uint32_t nei = kv.first;
      SendCtrlPkt(LSU, nei, seq, up ? 1 : 0, key.u, key.v, g_lsaBytes);
      if (g_ackRetransMix) {
        st.pending.insert(nei);
      }
    }
    if (!g_ackRetransMix) {
      st.pending.clear();
    }
    st.lastSend = now;
  }

  void RefreshTick() {
    auto upMap = m_core->GetLinkBitmap();
    for (auto& kv : m_neighbors) {
      uint32_t nei = kv.first;
      uint64_t key = MakeKey(m_self, nei);
      bool linkUp = (upMap.find(key) != upMap.end()) ? upMap.at(key) : false;
      FloodLsuAll(m_self, nei, linkUp);
    }
    if (g_lsaRefreshSec > 0) {
      m_refreshEv = Simulator::Schedule(Seconds(g_lsaRefreshSec), &OspfLiteAgent::RefreshTick, this);
    }
  }

  void AgeTick() {
    if (g_lsaMaxAge <= 0) return;
    double now = Simulator::Now().GetSeconds();
    bool purged = false;
    for (auto it = m_lsdb.begin(); it != m_lsdb.end();) {
      if (now - it->second.lastUpdate > g_lsaMaxAge) {
        it = m_lsdb.erase(it);
        purged = true;
      } else {
        ++it;
      }
    }
    if (purged) {
      m_core->RequestSpf();
    }
    m_ageEv = Simulator::Schedule(Seconds(5.0), &OspfLiteAgent::AgeTick, this);
  }

  void NeighborScan() {
    double now = Simulator::Now().GetSeconds();
    for (auto& kv : m_neighbors) {
      uint32_t nei = kv.first;
      double last = m_lastHello[nei];
      double missThreshold = g_helloInterval * std::max(1u, g_helloMissK);
      if (now - last > missThreshold) {
        if (m_nbrState[nei] != DOWN) {
          m_nbrState[nei] = DOWN;
          if (g_fullFlood) {
            FloodLsuAll(m_self, nei, false);
          } else {
            LsaKey key{std::min(m_self, nei), std::max(m_self, nei)};
            auto& rec = m_lsdb[key];
            rec.seq += 1;
            rec.lastUpdate = now;
            rec.up = false;
            FloodLsu(key, rec.seq, false);
          }
          m_core->RequestSpf();
        }
      } else {
        if (m_nbrState[nei] < TWOWAY) {
          m_nbrState[nei] = TWOWAY;
        }
      }
    }
    if (g_neighborScanSec > 0) {
      m_deadEv = Simulator::Schedule(Seconds(g_neighborScanSec), &OspfLiteAgent::NeighborScan, this);
    }
  }

  // SendCtrlPkt: encode and transmit a control packet toward a neighbor.
  void SendCtrlPkt(PktType t,
                   uint32_t nei,
                   uint32_t seq,
                   uint32_t upFlag,
                   uint32_t u,
                   uint32_t v,
                   uint32_t targetBytes) {
    if (!m_sock) return;  // socket not created yet; skip send

    auto it = m_neighbors.find(nei);
    if (it == m_neighbors.end()) return;
    Ipv4Address dst = it->second;

    // [type(1)][src(2)][seq(4)][u(2)][v(2)][up(1)] = 12 bytes
    uint8_t hdr[12] = {0};
    hdr[0] = static_cast<uint8_t>(t);
    hdr[1] = (m_self >> 8) & 0xFF;
    hdr[2] = m_self & 0xFF;
    hdr[3] = (seq >> 24) & 0xFF;
    hdr[4] = (seq >> 16) & 0xFF;
    hdr[5] = (seq >> 8) & 0xFF;
    hdr[6] = seq & 0xFF;
    hdr[7] = (u >> 8) & 0xFF;
    hdr[8] = u & 0xFF;
    hdr[9] = (v >> 8) & 0xFF;
    hdr[10] = v & 0xFF;
    hdr[11] = static_cast<uint8_t>(upFlag & 0x1);

    uint32_t padLen = (targetBytes > sizeof(hdr)) ? (targetBytes - sizeof(hdr)) : 0u;
    std::vector<uint8_t> buf(sizeof(hdr) + padLen, 0);
    std::memcpy(buf.data(), hdr, sizeof(hdr));

    uint32_t len = std::max<uint32_t>((uint32_t)sizeof(hdr), targetBytes);
    // Token bucket check; if not enough credit, retry shortly
    if (!Admit(len)) {
      Simulator::Schedule(MilliSeconds(10), &OspfLiteAgent::SendCtrlPkt, this, t, nei, seq, upFlag, u, v, targetBytes);
      return;
    }
    Ptr<Packet> p = Create<Packet>(buf.data(), buf.size());
    m_ctrlBytesTx += buf.size();
    m_sock->SendTo(p, 0, InetSocketAddress(dst, m_port));
  }
  // Recv: handle Hello/LSU/LSACK packets from neighbors.
  void Recv(Ptr<Socket> s) {
    Address from;
    Ptr<Packet> p = s->RecvFrom(from);
    if (!p) return;

    uint8_t data[16];
    uint32_t n = p->CopyData(data, sizeof(data));
    if (n < 8) return;  // need at least type+src+seq
    PktType t = static_cast<PktType>(data[0]);
    uint32_t src = (uint32_t(data[1]) << 8) | uint32_t(data[2]);
    uint32_t seq = (uint32_t(data[3]) << 24) | (uint32_t(data[4]) << 16) | (uint32_t(data[5]) << 8) | uint32_t(data[6]);
    uint32_t u = 0, v = 0;
    bool up = false;
    if (n >= 12) {
      u = (uint32_t(data[7]) << 8) | uint32_t(data[8]);
      v = (uint32_t(data[9]) << 8) | uint32_t(data[10]);
      up = (n >= 13) ? (data[11] & 1) : false;
    }

    if (t == HELLO) {
      double now = Simulator::Now().GetSeconds();
      m_lastHello[src] = now;
      if (m_nbrState[src] < INIT) {
        m_nbrState[src] = INIT;
      }
      return;
    }

    if (t == LSU) {
      // Update LSDB if newer and pass MinLSArrival filter
      LsaKey key{std::min(u, v), std::max(u, v)};
      auto& rec = m_lsdb[key];
      double now = Simulator::Now().GetSeconds();
      bool newer = (seq > rec.seq);
      bool tooSoon = (now - rec.lastUpdate < g_minLsArrival);
      if (newer && !tooSoon) {
        rec.seq = seq;
        rec.lastUpdate = now;
        rec.up = up;
        // Reflect to routing core and request SPF
        m_core->RequestSpf();
        OutState& st = m_outbox[key];
        st.seq = seq;
        st.up = up;
        st.u = key.u;
        st.v = key.v;
        st.pending.clear();

        for (auto& kv : m_neighbors) {
          uint32_t nei = kv.first;
          if (nei == src) continue;
          SendCtrlPkt(LSU, nei, seq, up ? 1 : 0, key.u, key.v, g_lsaBytes);
          if (g_ackRetransMix) {
            st.pending.insert(nei);
          }
        }
        if (!g_ackRetransMix) {
          st.pending.clear();
        }
        st.lastSend = now;
      }
      // Delayed LSACK back to src with small jitter (coalesces naturally)
      Ptr<UniformRandomVariable> U = CreateObject<UniformRandomVariable>();
      double delay = g_ackDelayBase + U->GetValue(0.0, g_ackDelayJitter);
      Simulator::Schedule(Seconds(delay), &OspfLiteAgent::SendCtrlPkt, this, LSACK, src, seq, 0, key.u, key.v, g_lsaBytes);
      return;
    }

    if (t == LSACK) {
      // Clear pending ack for this (u,v,seq)
      LsaKey key{std::min(u, v), std::max(u, v)};
      auto it = m_outbox.find(key);
      if (it != m_outbox.end() && it->second.seq == seq) {
        it->second.pending.erase(src);
      }
      return;
    }
  }

  struct OutState {
    uint32_t seq{0};
    bool up{false};
    uint32_t u{0}, v{0};
    double lastSend{0.0};
    std::set<uint32_t> pending;
  };

  Ptr<OspfLite> m_core;
  NodeContainer m_nodes;
  std::vector<LinkRef> m_links;
  uint32_t m_self{0};
  uint16_t m_port{8899};
  Ptr<Socket> m_sock;
  std::map<uint32_t, Ipv4Address> m_neighbors;  // neighborId -> addr
  EventId m_helloEv, m_retxEv, m_refreshEv, m_deadEv, m_ageEv;

  std::unordered_map<LsaKey, LsaRec, LsaKeyHash> m_lsdb;
  std::map<uint32_t, double> m_lastHello;  // src->last time
  std::map<uint32_t, NbrState> m_nbrState;
  std::unordered_map<LsaKey, OutState, LsaKeyHash> m_outbox;
  std::unordered_map<LsaKey, BackoffState, LsaKeyHash> m_backoff;
  uint32_t m_seqCounter{0};
  uint64_t m_ctrlBytesTx{0};

  // Control-plane shaper state
  double m_cpRateBps{g_ctrlRateKbps * 1000.0};
  double m_cpBucket{double(g_ctrlBucket) / 2.0};
  double m_cpBucketMax{double(g_ctrlBucket)};
  double m_cpLastRefill{0.0};

  bool Admit(uint32_t bytes) {
    double now = Simulator::Now().GetSeconds();
    double add = m_cpRateBps * (now - m_cpLastRefill) / 8.0;
    m_cpBucket = std::min(m_cpBucketMax, m_cpBucket + add);
    m_cpLastRefill = now;
    if (m_cpBucket >= bytes) {
      m_cpBucket -= bytes;
      return true;
    }
    return false;
  }
};

// ---------------------------- Range manager ---------------------------------
// RangeManager: toggles ISLs based on distance and blackout timers.
class RangeManager final : public Object {
 public:
  RangeManager(const NodeContainer& nodes,
               std::vector<LinkRef>& links,
               Ptr<OspfLite> ospf,
               std::vector<Ptr<OspfLiteAgent>> agents,
               double rangeMeters,
               double period)
      : m_nodes(nodes), m_links(links), m_ospf(ospf), m_agents(agents), m_range(rangeMeters), m_period(period) {}

  void Start() { Simulator::Schedule(Seconds(0.3), &RangeManager::Tick, this); }

 private:
  NodeContainer m_nodes;
  std::vector<LinkRef>& m_links;
  Ptr<OspfLite> m_ospf;
  std::vector<Ptr<OspfLiteAgent>> m_agents;  // one per node
  double m_range, m_period;
  std::unordered_map<uint64_t, double> m_nextUp;
  static double Bucket(double val, double step) { return step > 0 ? step * std::round(val / step) : val; }

  // Tick: evaluate link ranges, toggle error models, and adjust delay.
  void Tick() {
    const double now = Simulator::Now().GetSeconds();
    const double blackout = g_blackoutMs / 1000.0;

    for (auto& L : m_links) {
      SatGeom geomA = ComputeSatGeom(L.a, now);
      SatGeom geomB = ComputeSatGeom(L.b, now);
      const double chordMeters = ChordDistance(geomA.ecef, geomB.ecef);
      const bool losOk = HasLineOfSight(geomA.ecef, geomB.ecef);
      const bool inRange = losOk && (chordMeters <= m_range);
      const uint64_t key = MakeKey(L.a, L.b);

      const bool isUp = (L.aErr->GetRate() == 0.0);
      const bool mayUp = (m_nextUp.find(key) == m_nextUp.end() || now >= m_nextUp[key]);

      if (!inRange) {
        m_nextUp[key] = now + blackout;
        if (isUp) {
          L.aErr->SetRate(1.0);
          L.bErr->SetRate(1.0);
          m_ospf->SetLinkUp(L.a, L.b, false);
          if (g_vis) {
            g_vis->EmitIslState(now, static_cast<int>(L.a), static_cast<int>(L.b), false);
          }
          // Debounced, single-end origination (lower id) after small delay
          uint32_t lo = std::min(L.a, L.b);
          Ptr<UniformRandomVariable> U = CreateObject<UniformRandomVariable>();
          double d = U->GetValue(0.2, 0.8);
          Simulator::Schedule(Seconds(d), &OspfLiteAgent::NotifyLinkChange, m_agents[lo], L.a, L.b, false);
        }
      } else {
        const bool shouldBeUp = mayUp;
        if (shouldBeUp && !isUp) {
          L.aErr->SetRate(0.0);
          L.bErr->SetRate(0.0);
          m_ospf->SetLinkUp(L.a, L.b, true);
          if (g_vis) {
            g_vis->EmitIslState(now, static_cast<int>(L.a), static_cast<int>(L.b), true);
          }
          uint32_t lo = std::min(L.a, L.b);
          Ptr<UniformRandomVariable> U = CreateObject<UniformRandomVariable>();
          double d = U->GetValue(0.2, 0.8);
          Simulator::Schedule(Seconds(d), &OspfLiteAgent::NotifyLinkChange, m_agents[lo], L.a, L.b, true);
        } else if (!shouldBeUp && isUp) {
          L.aErr->SetRate(1.0);
          L.bErr->SetRate(1.0);
          m_ospf->SetLinkUp(L.a, L.b, false);
          if (g_vis) {
            g_vis->EmitIslState(now, static_cast<int>(L.a), static_cast<int>(L.b), false);
          }
          uint32_t lo = std::min(L.a, L.b);
          Ptr<UniformRandomVariable> U = CreateObject<UniformRandomVariable>();
          double d = U->GetValue(0.2, 0.8);
          Simulator::Schedule(Seconds(d), &OspfLiteAgent::NotifyLinkChange, m_agents[lo], L.a, L.b, false);
        }
        if (shouldBeUp && L.aErr->GetRate() == 0.0) {
          double propMs = 1000.0 * chordMeters / g_cEff;
          propMs = std::max(g_minDelayMs, std::min(g_maxDelayMs, propMs));
          double bucketed = Bucket(propMs, g_delayBucketMs);
          Time desired = MilliSeconds(bucketed);
          if (desired != L.lastDelay) {
            L.ch->SetAttribute("Delay", TimeValue(desired));
            L.lastDelay = desired;
          }
        }
      }
    }
    Simulator::Schedule(Seconds(m_period), &RangeManager::Tick, this);
  }
};

// ---------------------------- Gravity model ---------------------------------
struct Gs {
  std::string name;
  double x;
  double y;
  double pop;
};
static std::vector<Gs> DefaultGs(double fieldWidth, double fieldHeight) {
  return {
      {"NA-West", 0.15 * fieldWidth, 0.70 * fieldHeight, 60},
      {"NA-East", 0.35 * fieldWidth, 0.70 * fieldHeight, 90},
      {"LATAM-N", 0.35 * fieldWidth, 0.45 * fieldHeight, 80},
      {"LATAM-S", 0.40 * fieldWidth, 0.25 * fieldHeight, 60},
      {"EU-West", 0.55 * fieldWidth, 0.70 * fieldHeight, 100},
      {"EU-East", 0.65 * fieldWidth, 0.70 * fieldHeight, 90},
      {"AF-North", 0.60 * fieldWidth, 0.50 * fieldHeight, 120},
      {"AF-South", 0.62 * fieldWidth, 0.25 * fieldHeight, 70},
      {"ME", 0.72 * fieldWidth, 0.55 * fieldHeight, 80},
      {"IN", 0.80 * fieldWidth, 0.55 * fieldHeight, 140},
      {"SEA", 0.88 * fieldWidth, 0.45 * fieldHeight, 180},
      {"EA", 0.90 * fieldWidth, 0.65 * fieldHeight, 200},
  };
}

struct PairSampler {
  struct Item {
    uint32_t i, j;
    double cdf;
  };
  std::vector<Item> items;
  double total{0};
  static double Dist2D(const Gs& a, const Gs& b) { return std::sqrt(sqr(a.x - b.x) + sqr(a.y - b.y)) + 1.0; }
  PairSampler(const std::vector<Gs>& gs, double alpha, uint32_t useFirstK) {
    uint32_t K = std::min<uint32_t>(useFirstK, gs.size());
    for (uint32_t i = 0; i < K; ++i) {
      for (uint32_t j = 0; j < K; ++j) {
        if (i == j) continue;
        double w = (gs[i].pop * gs[j].pop) / std::pow(Dist2D(gs[i], gs[j]), alpha);
        if (w <= 0) continue;
        total += w;
        items.push_back({i, j, total});
      }
    }
    for (auto& it : items) it.cdf /= total;
  }
  std::pair<uint32_t, uint32_t> Sample(std::mt19937& rng) const {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double u = U(rng);
    size_t lo = 0, hi = items.size();
    while (lo + 1 < hi) {
      size_t mid = (lo + hi) / 2;
      if (u <= items[mid].cdf)
        hi = mid;
      else
        lo = mid;
    }
    return {items[lo].i, items[lo].j};
  }
};

// ------------------------------ Main ----------------------------------------
int main(int argc, char* argv[]) {
  CommandLine cmd;
  cmd.AddValue("planes", "Number of orbital planes.", g_planes);
  cmd.AddValue("perPlane", "Satellites per plane.", g_perPlane);
  cmd.AddValue("wrap", "Torus neighbor wrap.", g_wrap);
  cmd.AddValue("spacingKm", "Grid spacing (km).", g_spacingKm);
  cmd.AddValue("simTime", "Simulation time (s).", g_simTime);
  cmd.AddValue("islRateMbps", "ISL rate (Mbps).", g_islRateMbps);
  cmd.AddValue("islDelayMs", "ISL propagation delay (ms).", g_islDelayMs);
  cmd.AddValue("rangeKm", "ISL range threshold (km).", g_rangeKm);
  cmd.AddValue("checkPeriod", "Range check period (s).", g_checkPeriod);
  cmd.AddValue("spfInterval", "SPF watchdog period (s).", g_spfInterval);
  cmd.AddValue("spfJitterMax", "SPF jitter max (s).", g_spfJitterMax);
  cmd.AddValue("minSpfHold", "Legacy min time between SPF runs (s).", g_minSpfHold);
  cmd.AddValue("flows", "Number of concurrent UDP flows.", g_flows);
  cmd.AddValue("pktSize", "UDP payload size (bytes).", g_pktSize);
  cmd.AddValue("interPacket", "UDP inter-packet interval (s).", g_interPacket);
  // Seeds: accept both a flow sampler seed, and the standard ns-3 RNG knobs
  cmd.AddValue("runSeed", "Flow sampler seed (std::mt19937).", g_runSeed);
  cmd.AddValue("RngSeed", "ns-3 master RNG seed.", g_rngSeed);
  cmd.AddValue("RngRun", "ns-3 RNG run number.", g_rngRun);
  cmd.AddValue("numGs", "How many GS regions from the default list.", g_numGs);
  cmd.AddValue("dstOnlyGs", "Track and report only flows whose destination is a GS-anchored node (0/1).", g_dstOnlyGs);
  cmd.AddValue("gravityAlpha", "Gravity model alpha.", g_gravityAlpha);
  cmd.AddValue("exportCsv", "Write metrics CSV to this path.", g_exportCsv);
  cmd.AddValue("exportSummary", "Write a tiny summary file (flags + DATA and CONTROL blocks).", g_exportSummary);
  cmd.AddValue("blackoutMs", "Blackout after link re-enters range (ms).", g_blackoutMs);
  cmd.AddValue("cEff", "Effective propagation speed (m/s).", g_cEff);
  cmd.AddValue("minDelayMs", "Minimum per-link delay (ms).", g_minDelayMs);
  cmd.AddValue("maxDelayMs", "Maximum per-link delay (ms).", g_maxDelayMs);
  cmd.AddValue("delayBucketMs", "Bucket step for delay updates (ms).", g_delayBucketMs);
  // New control-plane knobs
  cmd.AddValue("ctrlPort", "UDP port for control plane.", g_ctrlPort);
  cmd.AddValue("helloInterval", "Hello period (s).", g_helloInterval);
  cmd.AddValue("deadInterval", "Dead interval (s).", g_deadInterval);
  cmd.AddValue("helloJitter", "Hello jitter fraction (+/-).", g_helloJitterFrac);
  cmd.AddValue("helloMissK", "Neighbor down after K missed hellos.", g_helloMissK);
  cmd.AddValue("spfDelay", "Event-driven SPF delay (s).", g_spfDelay);
  cmd.AddValue("spfHold", "Event-driven SPF hold (s).", g_spfHold);
  cmd.AddValue("minLsArrival", "Min LS arrival accept interval (s).", g_minLsArrival);
  cmd.AddValue("lsaMinInterval", "Min interval between re-originations (s).", g_lsaMinInterval);
  cmd.AddValue("lsaBackoffMax", "Max LSA origination backoff (s).", g_lsaBackoffMax);
  cmd.AddValue("lsaFanout", "Neighbors per LSU tick.", g_lsaFanout);
  cmd.AddValue("lsaRetransmit", "LSU retransmit period (s).", g_lsaRetransmit);
  cmd.AddValue("helloBytes", "Approx Hello payload bytes.", g_helloBytes);
  cmd.AddValue("lsaBytes", "Approx LSA payload bytes.", g_lsaBytes);
  cmd.AddValue("lsaRefresh", "Periodic LS refresh (s).", g_lsaRefreshSec);
  cmd.AddValue("lsaMaxAge", "Max LSA age before purged (s).", g_lsaMaxAge);
  cmd.AddValue("nbrScan", "Neighbor scan period (s).", g_neighborScanSec);
  cmd.AddValue("fullFlood", "Flood LSAs to all neighbors.", g_fullFlood);
  cmd.AddValue("ackRetrans", "Keep per-neighbor retransmit lists.", g_ackRetransMix);
  cmd.AddValue("ackDelayBase", "LSACK delay base (s).", g_ackDelayBase);
  cmd.AddValue("ackDelayJitter", "LSACK delay jitter (s).", g_ackDelayJitter);
  cmd.AddValue("ctrlRateKbps", "Control-plane rate shaping (kbps).", g_ctrlRateKbps);
  cmd.AddValue("ctrlBucket", "Control-plane token bucket (bytes).", g_ctrlBucket);
  cmd.AddValue("spfBackoffInit", "Initial SPF backoff (s).", g_spfBackoffInit);
  cmd.AddValue("spfBackoffMax", "Max SPF backoff (s).", g_spfBackoffMax);
  cmd.AddValue("spfBackoffDecay", "SPF backoff decay per run (s).", g_spfBackoffDecay);
  cmd.AddValue("fibInstallBase", "Base route install delay (s).", g_fibInstallBase);
  cmd.AddValue("fibInstallJitter", "Extra route install jitter (s).", g_fibInstallJitter);
  cmd.AddValue("measureStart", "Ignore flow stats before this time (s).", g_measureStart);
  cmd.AddValue("progressBeatSec", "Emit progress log every N seconds.", g_progressBeatSec);
  cmd.AddValue("quietApps", "Suppress app setup logging.", g_quietApps);
  cmd.AddValue("enablePcap", "Enable per-link PCAP capture.", g_enablePcap);
  cmd.AddValue("logEveryPkts", "(compat) unused packet logging interval.", g_logEveryPkts);
  cmd.AddValue("logEverySimSec", "(compat) unused sim logging interval (s).", g_logEverySimSec);
  cmd.AddValue("queuePkts", "ISL egress queue depth (packets)", g_queuePkts);
  cmd.AddValue("satAltitudeKm", "Satellite orbital altitude (km).", g_satAltKm);
  cmd.AddValue("satInclDeg", "Satellite inclination (degrees).", g_satInclinationDeg);
  cmd.AddValue("walkerPhase", "Walker constellation phase F.", g_walkerPhaseF);
  cmd.AddValue("emitSatStates", "Emit satellite lat/lon/alt lines to NDJSON.", g_emitSatStates);
  cmd.AddValue("pidFile", "Write running PID to this path.", g_pidFile);
  cmd.AddValue("logFile", "(compat) requested log file path.", g_logFile);
  cmd.AddValue("tag", "Run tag for logging.", g_tag);
  cmd.AddValue("debugFlowEvents", "Trace FlowSender self-reschedule bookkeeping.", g_debugFlowEvents);
  cmd.Parse(argc, argv);

  g_rangeKm = std::max(0.0, g_rangeKm);
  g_satAltKm = std::max(0.0, g_satAltKm);
  g_satInclinationDeg = std::clamp(g_satInclinationDeg, 0.0, 180.0);


  g_flowSummaries.clear();
  g_flowSummaries.resize(g_flows);
  ResetFlowSummaries();
  g_flowEventTracker.Enable(g_debugFlowEvents);

  if (!g_pidFile.empty()) {
    std::ofstream pf(g_pidFile);
    if (pf.is_open()) {
      pf << static_cast<long>(getpid()) << "\n";
    }
  }
  if (!g_quietApps) {
    if (!g_logFile.empty()) {
      std::cout << "[LOG] using " << g_logFile << "\n";
    }
    std::cout << "[RUN] pid=" << static_cast<long>(getpid())
              << " seed=" << g_runSeed
              << " rngSeed=" << g_rngSeed
              << " rngRun=" << g_rngRun
              << " flows=" << g_flows
              << " progressBeatSec=" << g_progressBeatSec
              << " measureStart=" << g_measureStart;
    if (!g_tag.empty()) {
      std::cout << " tag=" << g_tag;
    }
    std::cout << "\n";
    std::cout << "[DATA] FlowSender=1 measureStart=" << g_measureStart
              << " dataCounters=data-only ctrlCounters=CtrlBytesTx" << "\n";
  }

  {
    std::ostringstream qs;
    qs << g_queuePkts << "p";
    Config::SetDefault("ns3::DropTailQueue<Packet>::MaxSize", StringValue(qs.str()));
  }

  const uint32_t N = g_planes * g_perPlane;
  // Honor standard ns-3 RNG controls; do not override with hard-coded seed
  RngSeedManager::SetSeed(g_rngSeed);
  RngSeedManager::SetRun(g_rngRun);

  // --- Create satellites and configure mobility ---
  NodeContainer sats;
  sats.Create(N);

  double spacing = g_spacingKm * 1000.0;
  double W = spacing * (g_perPlane - 1);
  double H = spacing * (g_planes - 1);

  MobilityHelper mob;
  Ptr<ListPositionAllocator> pos = CreateObject<ListPositionAllocator>();
  auto idx = [&](uint32_t r, uint32_t c) { return r * g_perPlane + c; };
  for (uint32_t r = 0; r < g_planes; ++r)
    for (uint32_t c = 0; c < g_perPlane; ++c) pos->Add(Vector(c * spacing, r * spacing, 0));
  mob.SetPositionAllocator(pos);
  mob.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
  mob.Install(sats);
  for (uint32_t r = 0; r < g_planes; ++r) {
    for (uint32_t c = 0; c < g_perPlane; ++c) {
      auto m = sats.Get(idx(r, c))->GetObject<ConstantVelocityMobilityModel>();
      double vx = (c % 2 == 0) ? 200.0 : -200.0;
      double vy = (r % 2 == 0) ? -200.0 : 200.0;
      m->SetVelocity(Vector(vx, vy, 0));
    }
  }

  // --- Install Internet stack ---
  InternetStackHelper internet;
  internet.Install(sats);

  std::vector<std::pair<uint32_t, uint32_t>> edges;
  for (uint32_t r = 0; r < g_planes; ++r) {
    for (uint32_t c = 0; c < g_perPlane; ++c) {
      uint32_t u = idx(r, c);
      if (c + 1 < g_perPlane || g_wrap) {
        uint32_t v = idx(r, (c + 1) % g_perPlane);
        if (u < v) edges.push_back({u, v});
      }
      if (r + 1 < g_planes || g_wrap) {
        uint32_t v = idx((r + 1) % g_planes, c);
        if (u < v) edges.push_back({u, v});
      }
    }
  }

  PointToPointHelper p2p;
  std::ostringstream dr;
  dr << (uint32_t)g_islRateMbps << "Mbps";
  std::ostringstream de;
  de << g_islDelayMs << "ms";
  p2p.SetDeviceAttribute("DataRate", StringValue(dr.str()));
  p2p.SetChannelAttribute("Delay", StringValue(de.str()));
  Ipv4AddressHelper ip;
  ip.SetBase("10.0.0.0", "255.255.255.252");

  std::vector<LinkRef> links;
  links.reserve(edges.size());
  for (auto [a, b] : edges) {
    NetDeviceContainer devs = p2p.Install(NodeContainer(sats.Get(a), sats.Get(b)));
    Ptr<RateErrorModel> ea = CreateObject<RateErrorModel>();
    Ptr<RateErrorModel> eb = CreateObject<RateErrorModel>();
    ea->SetUnit(RateErrorModel::ERROR_UNIT_PACKET);
    ea->SetRate(1.0);
    eb->SetUnit(RateErrorModel::ERROR_UNIT_PACKET);
    eb->SetRate(1.0);
    devs.Get(0)->SetAttribute("ReceiveErrorModel", PointerValue(ea));
    devs.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(eb));
    Ipv4InterfaceContainer ifs = ip.Assign(devs);
    ip.NewNetwork();
    Ptr<PointToPointChannel> ch = DynamicCast<PointToPointChannel>(devs.Get(0)->GetChannel());
    Time seededDelay = MilliSeconds(g_islDelayMs);
    ch->SetAttribute("Delay", TimeValue(seededDelay));
    links.push_back({a, b, ifs, ea, eb, ch, seededDelay});
    if (g_enablePcap) {
      std::ostringstream pcapPrefix;
      pcapPrefix << "ospf_lite_" << a << "_" << b;
      p2p.EnablePcap(pcapPrefix.str(), devs, false);
    }
  }

  std::vector<Ipv4Address> loopbacks(N, Ipv4Address("0.0.0.0"));
  for (uint32_t n = 0; n < N; ++n) {
    Ptr<Ipv4> ip4 = sats.Get(n)->GetObject<Ipv4>();

    std::ostringstream lb;
    lb << "192.168." << (n / 256) << '.' << (n % 256);
    Ipv4Address loAddr(lb.str().c_str());

    // Attach stable /32 alias to loopback interface (index 0)
    Ipv4InterfaceAddress loAlias(loAddr, Ipv4Mask("255.255.255.255"));
    ip4->AddAddress(0, loAlias);
    ip4->SetUp(0);
    loopbacks[n] = loAddr;
  }

  // --- Set up OSPF-lite core ---
  Ptr<OspfLite> ospf = CreateObject<OspfLite>(sats, links, g_spfInterval, g_spfJitterMax, g_minSpfHold);
  ospf->PreparePrevMaps(N);
  ospf->SetLoopbacks(loopbacks);

  // --- Install per-node OSPF-lite agents ---
  std::vector<Ptr<OspfLiteAgent>> agents(N);
  for (uint32_t n = 0; n < N; ++n) {
    agents[n] = CreateObject<OspfLiteAgent>();
    agents[n]->Configure(ospf, sats, links, n, g_ctrlPort);
    sats.Get(n)->AddApplication(agents[n]);
    agents[n]->SetStartTime(Seconds(0.05));
    agents[n]->SetStopTime(Seconds(g_simTime));
  }

  // --- Start range manager and routing core ---
  Ptr<RangeManager> rm = CreateObject<RangeManager>(sats, links, ospf, agents, g_rangeKm * 1000.0, g_checkPeriod);
  rm->Start();
  ospf->Start();

  if (!g_vis) {
    g_vis = new VisPublisher("/tmp/leo_vis.ndjson");
  }
  if (g_emitSatStates)
  {
    Simulator::Schedule(Seconds(0.2), &EmitSatStates);
  }
  Simulator::Schedule(Seconds(0.2), []() {
    if (g_vis)
    {
      g_vis->WriteLine(R"({"type":"layout","mode":"logical","planes":6,"perPlane":11,"alt":780000,"incDeg":86.4,"phase":2})");
    }
  });

  // --- Prepare gravity model traffic anchors ---
  auto gsList = DefaultGs(W, H);
  g_numGs = std::max<uint32_t>(1, std::min<uint32_t>(g_numGs, gsList.size()));
  gsList.resize(g_numGs);

  std::mt19937 rng(g_runSeed);
  PairSampler sampler(gsList, g_gravityAlpha, g_numGs);

  auto nearestSat = [&](double x, double y) -> uint32_t {
    double best = 1e99;
    uint32_t bestIdx = 0;
    for (uint32_t n = 0; n < N; ++n) {
      auto m = sats.Get(n)->GetObject<MobilityModel>();
      Vector P = m->GetPosition();
      double d = std::sqrt(sqr(P.x - x) + sqr(P.y - y));
      if (d < best) {
        best = d;
        bestIdx = n;
      }
    }
    return bestIdx;
  };

  // Build GS destination mask (anchor satellites) for dstOnlyGs / marking
  g_groundDestMask.assign(N, false);
  for (const auto& gs : gsList)
  {
    uint32_t anchor = nearestSat(gs.x, gs.y);
    if (anchor < N)
    {
      g_groundDestMask[anchor] = true;
    }
  }

  std::cout << "[TRAFFIC] flows=" << g_flows
            << " pktSize=" << g_pktSize
            << " interPacket=" << g_interPacket
            << " pps=" << (g_interPacket > 0.0 ? (1.0 / g_interPacket) : 0.0)
            << "\n";

  uint16_t basePort = 9000;
  double startT = 3.0;
  uint32_t flowsMade = 0;
  for (; flowsMade < g_flows; ++flowsMade) {
    auto [gi, gj] = sampler.Sample(rng);
    uint32_t sIdx = nearestSat(gsList[gi].x, gsList[gi].y);
    uint32_t dIdx = nearestSat(gsList[gj].x, gsList[gj].y);
    if (sIdx == dIdx) {
      dIdx = (dIdx + 1) % N;
    }

    uint16_t dstPort = basePort + flowsMade;
    Ipv4Address dst = loopbacks[dIdx];

    if (!g_quietApps) {
      std::cout << "[FLOWSETUP] f=" << flowsMade
                << " srcId=" << sIdx
                << " dstId=" << dIdx
                << " dst=" << dst
                << " port=" << dstPort
                << " pps=" << (g_interPacket > 0.0 ? (1.0 / g_interPacket) : 0.0)
                << "\n";
    }

    PacketSinkHelper sink("ns3::UdpSocketFactory",
                          InetSocketAddress(Ipv4Address::GetAny(), dstPort));
    auto sApps = sink.Install(sats.Get(dIdx));
    sApps.Start(Seconds(1.0));
    sApps.Stop(Seconds(g_simTime + 1.0));
    Ptr<PacketSink> sinkPtr = (sApps.GetN() > 0) ? DynamicCast<PacketSink>(sApps.Get(0)) : nullptr;

    const uint32_t flowId = flowsMade;
    if (sinkPtr) {
      sinkPtr->TraceConnectWithoutContext("Rx", MakeBoundCallback(&RecordSinkRx, flowId));
    }

    // Mark if this flow's destination is GS-anchored for downstream aggregation
    if (flowId < g_flowSummaries.size())
    {
      g_flowSummaries[flowId].dstIsGs = (dIdx < g_groundDestMask.size()) ? g_groundDestMask[dIdx] : false;
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

  if (!g_quietApps) {
    std::cout << "[FLOWSETUP] requested=" << g_flows
              << " created=" << flowsMade
              << "\n";
  }

  if (g_progressBeatSec > 0.0 && g_progressBeatSec < g_simTime) {
    Simulator::Schedule(Seconds(g_progressBeatSec), &ProgressBeat);
  }

  if (g_measureStart > 0.0 && g_measureStart < g_simTime) {
    Simulator::Schedule(Seconds(g_measureStart), &ResetMeasureCounters);
  }

  Simulator::Stop(Seconds(g_simTime));
  Simulator::Run();

  uint64_t totalCtrlBytes = 0;
  for (const auto& agent : agents) {
    if (agent) {
      totalCtrlBytes += agent->GetCtrlBytesTx();
    }
  }
  double ctrlAvgMbps = (g_simTime > 0) ? (8.0 * totalCtrlBytes / g_simTime / 1e6) : 0.0;

  auto computeDuration = [](const FlowSummary& st) {
    double rxSpan = (st.firstRx >= 0.0 && st.lastRx > st.firstRx) ? (st.lastRx - st.firstRx) : 0.0;
    double txSpan = (st.firstTx >= 0.0 && st.lastTx > st.firstTx) ? (st.lastTx - st.firstTx) : 0.0;
    double dur = (rxSpan > 0.0) ? rxSpan : txSpan;
    if (dur <= 0.0 && st.firstTx >= 0.0) {
      dur = std::max(0.0, g_simTime - st.firstTx);
    }
    return dur;
  };

  uint64_t totalTx = 0;
  uint64_t totalRx = 0;
  double totalDelaySec = 0.0;
  double sumThrMbps = 0.0;
  uint32_t countedFlows = 0;

  for (const auto& st : g_flowSummaries) {
    if (g_dstOnlyGs && !st.dstIsGs) {
      continue;
    }
    totalTx += st.txPackets;
    totalRx += st.rxPackets;
    totalDelaySec += st.delaySum;

    if (st.rxPackets > 0) {
      double dur = computeDuration(st);
      if (dur > 0.0 && st.rxBytes > 0) {
        sumThrMbps += (static_cast<double>(st.rxBytes) * 8.0) / (dur * 1e6);
      }
      ++countedFlows;
    }
  }

  const uint64_t totalLost = (totalTx >= totalRx) ? (totalTx - totalRx) : 0;
  const double pdr = (totalTx > 0) ? static_cast<double>(totalRx) / static_cast<double>(totalTx) : 0.0;
  const double avgDelay = (totalRx > 0) ? (totalDelaySec / static_cast<double>(totalRx)) : 0.0;
  const double avgThr = (countedFlows > 0) ? (sumThrMbps / static_cast<double>(countedFlows)) : 0.0;

  bool writeCsv = !g_exportCsv.empty();
  if (writeCsv) {
    try {
      std::filesystem::path path(g_exportCsv);
      if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
      }
      std::ofstream csv(path, std::ios::out | std::ios::trunc);
      if (csv.is_open()) {
        csv << "flowId,tx,rx,lost,pdr,avgDelayMs,throughputMbps\n";
        csv.setf(std::ios::fixed, std::ios::floatfield);
        for (uint32_t fid = 0; fid < g_flowSummaries.size(); ++fid) {
          const auto& st = g_flowSummaries[fid];
          if (g_dstOnlyGs && !st.dstIsGs) {
            continue;
          }
          const uint64_t lost = (st.txPackets >= st.rxPackets) ? (st.txPackets - st.rxPackets) : 0;
          const double flowPdr = (st.txPackets > 0) ? static_cast<double>(st.rxPackets) / static_cast<double>(st.txPackets) : 0.0;
          const double delayMs = (st.rxPackets > 0) ? (st.delaySum / static_cast<double>(st.rxPackets) * 1000.0) : 0.0;
          double dur = computeDuration(st);
          const double thrMbps = (dur > 0.0 && st.rxBytes > 0)
                                    ? (static_cast<double>(st.rxBytes) * 8.0) / (dur * 1e6)
                                    : 0.0;
          csv << fid << ',' << st.txPackets << ',' << st.rxPackets << ',' << lost << ','
              << std::setprecision(5) << flowPdr << ','
              << std::setprecision(3) << delayMs << ','
              << std::setprecision(5) << thrMbps << '\n';
        }
      }
    } catch (...) {
      std::cerr << "[WARN] failed to write data CSV to " << g_exportCsv << "\n";
    }
  }

  std::ostringstream dataBlock, ctrlBlock;
  dataBlock << "\n==== DATA METRICS ====\n";
  dataBlock << "Nodes: " << N << "  ISLs: " << links.size()
            << "  Range(km): " << g_rangeKm
            << "  SPF-watchdog(s): " << g_spfInterval
            << "  Flows: " << g_flows << "\n";
  dataBlock << "TxPkts: " << totalTx
            << "  RxPkts: " << totalRx
            << "  Lost: " << totalLost
            << "  PDR: " << std::setprecision(5) << pdr
            << "  AvgDelay(s): " << avgDelay
            << "  AvgThroughput(Mbps/flow): " << avgThr << "\n";

  ctrlBlock << "\n==== CONTROL (proxy) ====\n";
  ctrlBlock << "SpfRuns:   " << ospf->GetSpfRuns() << "\n";
  ctrlBlock << "CtrlBytesTx: " << totalCtrlBytes
            << "  CtrlAvgRate(Mbps): " << ctrlAvgMbps << "\n";

  // keep console output identical
  std::cout << dataBlock.str();
  std::cout << ctrlBlock.str();

  if (!g_exportCsv.empty()) {
    std::ofstream ccsv(g_exportCsv, std::ios::app);
    if (ccsv.is_open()) {
      ccsv << "CONTROL,,, ,,,"  // keep CSV shape minimal
           << "SpfRuns=" << ospf->GetSpfRuns()
           << ",CtrlBytesTx=" << totalCtrlBytes
           << ",CtrlAvgMbps=" << ctrlAvgMbps
           << "\n";
    } else {
      std::cerr << "[WARN] failed to append control metrics to " << g_exportCsv << "\n";
    }
  }

  // write tiny summary file if requested
  if (!g_exportSummary.empty()) {
    try {
      std::filesystem::path p(g_exportSummary);
      if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
      }
      std::ofstream out(g_exportSummary, std::ios::out | std::ios::trunc);
      if (out.is_open()) {
        out << "# FLAGS\n" << DumpFlagsOspfLite() << "\n";
        out << dataBlock.str() << "\n";
        out << ctrlBlock.str();
      }
    } catch (...) {
      std::cerr << "[WARN] failed to write exportSummary to " << g_exportSummary << "\n";
    }
  }

  // removed legacy one-line CSV print; summaries go to console and optional exportSummary file


  if (g_debugFlowEvents) {
    std::cout << "[FLOWDEBUG] maxScheduled=" << g_flowEventTracker.MaxConcurrent()
              << " pending=" << g_flowEventTracker.Pending()
              << " flows=" << g_flows << "\n";
  }

  Simulator::Destroy();
  delete g_vis;
  g_vis = nullptr;
  return 0;
}
// end of file
