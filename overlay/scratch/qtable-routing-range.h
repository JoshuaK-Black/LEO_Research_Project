#pragma once
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/random-variable-stream.h"
#include "ns3/log.h"
#include <unordered_map>
#include <functional>
#include <map>
#include <queue>
#include <vector>
#include <array>
#include <utility>
#include <sstream>
#include <limits>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <iostream>
#include <atomic>
#include <mutex>
#include <string>
#include <cstdio>

#ifndef RL_DIAG
#define RL_DIAG 1
#endif

namespace ns3 {

#if RL_DIAG
inline void RlDbgFormat(const char* fmt)
{
  NS_LOG_UNCOND(fmt);
}

template <typename... Args>
inline void RlDbgFormat(const char* fmt, Args&&... args)
{
  int size = std::snprintf(nullptr, 0, fmt, std::forward<Args>(args)...);
  if (size <= 0)
  {
    NS_LOG_UNCOND(fmt);
    return;
  }
  std::string buf(static_cast<size_t>(size) + 1, '\0');
  std::snprintf(buf.data(), static_cast<size_t>(size) + 1, fmt, std::forward<Args>(args)...);
  buf.resize(static_cast<size_t>(size));
  NS_LOG_UNCOND(buf);
}
#else
inline void RlDbgFormat(const char*, ...) {}
#endif

#ifndef RLDBG
#if RL_DIAG
#define RLDBG(...) ::ns3::RlDbgFormat(__VA_ARGS__)
#else
#define RLDBG(...) do {} while (0)
#endif
#endif

extern uint64_t g_rlOverridePkts;
// Counts packets forwarded using an RL override (packet-level usage)
extern uint64_t g_rlPktUsed;
extern bool g_logRl;
extern std::atomic<uint64_t> g_qrProbeTxPkts;
extern std::atomic<uint64_t> g_qrPackTxPkts;
extern std::atomic<uint64_t> g_qrFbTxPkts;
extern std::atomic<uint64_t> g_qrProbeTxBytes;
extern std::atomic<uint64_t> g_qrPackTxBytes;
extern std::atomic<uint64_t> g_qrFbTxBytes;

void QrCtrlResetCounters();
void QrCtrlPrintProxy(double windowSeconds);

class LeoRoutingGymEnv;

#ifndef QTABLE_ROUTING_LOG_COMPONENT_GUARD
#define QTABLE_ROUTING_LOG_COMPONENT_GUARD
NS_LOG_COMPONENT_DEFINE("QTableRouting");
#endif

class QTableRouting : public Ipv4RoutingProtocol
{
  friend class LeoRoutingGymEnv;
public:
  static TypeId GetTypeId()
  {
    static TypeId tid = TypeId("ns3::QTableRouting")
        .SetParent<Ipv4RoutingProtocol>()
        .SetGroupName("Internet")
        .AddConstructor<QTableRouting>();
    return tid;
  }

  QTableRouting();
  ~QTableRouting() override;

  void Configure(double alpha,
                 double gamma,
                 double eps,
                 Time probeInterval,
                 uint32_t probeFanout,
                 Time penaltyDrop,
                 uint16_t ctrlPort,
                 Ptr<Object> rangeMgr);

  void SetDestinationMask(const std::vector<bool>& mask);
  void SetSeedHopMs(double v);
  void SetEpsSchedule(double start, double final, double tau);
  void SetUpdateStride(uint32_t n);
  using RlPickFn = std::function<int(uint32_t, uint32_t, const std::vector<uint32_t>&)>;

  static void InstallRlPicker(RlPickFn cb);
  static bool RlEnabled();
  static void SetRlEnabled(bool on);
  static void SetNextHopOverride(uint32_t nodeId, uint32_t dstId, int nextHop);
  // === RL override support ===
  // Force next-hop for a given destination for 'ttlSeconds'.
  // If 'neiId' is UINT32_MAX, clears the override.
  void SetRlNextHop(uint32_t dstId,
                    uint32_t neiId,
                    double ttlSeconds,
                    const char* origin = "router",
                    uint64_t stamp = 0);
  // Returns true if an active RL override exists and gives a valid neighbor.
  bool GetRlNextHop(uint32_t dstId, uint32_t &neiOut);
  void ClearRlNextHop(uint32_t dstId);
  void PinTeacher(uint32_t dstId, double sec = 1.0);
  bool IsTeacherPinned(uint32_t dstId);
  static void SetRlUseDisabled(bool on);
  static bool IsRlUseDisabled();
  static void SetCurrentEnvStamp(uint64_t stamp);
  static uint64_t GetCurrentEnvStamp();
  // Tunables: probe RTT EWMA weight and next-hop switch hysteresis
  static void SetProbeEwmaW(double w) { s_probeEwmaW = std::clamp(w, 0.01, 0.50); }
  static void SetSwitchHyst(double f) { s_switchHyst = std::clamp(f, 0.0, 0.25); }
  // Optional: live override of alpha/epsilon (enable mid-run regime switches)
  static void SetAlphaEps(double a, double e)
  {
    s_alpha = std::clamp(a, 0.0, 1.0);
    s_eps   = std::clamp(e, 0.0, 1.0);
    s_overrideAE = true;
  }
  // Shortest-path seeding + refresh knobs
  void EnableSeedFromSpf(bool on) { m_seedFromSpf = on; }
  void SetSpSeedMs(double ms)     { if (std::isfinite(ms) && ms > 0.0) m_spSeedMs = ms; }
  void SetSpfRefresh(double sec)  { m_spfRefreshS = std::max(0.0, sec); }
  void EnableFreezeAfterMeasure(bool on) { m_freezeAfterMeasure = on; }
  void SetMeasureStart(double sec) { m_measureStartS = std::max(0.0, sec); }

  void RegisterNodeAddress(uint32_t nodeId, Ipv4Address nodeAddr);
  void AddAdjacency(uint32_t neiId, Ipv4Address myIfAddr, Ipv4Address neiIfAddr, uint32_t myOif);
  void KickRebuild();

  static void SetLogPolicy(uint64_t logEveryPackets, Time logEverySimTime);
  static void SetRebuildLogging(bool on) { s_logRebuild = on; }

  static void SetLinkUp(uint32_t a, uint32_t b, bool up)
  {
    S().linkUp[Key(a, b)] = up;
    for (auto* inst : Registry())
    {
      if (!inst)
      {
        continue;
      }
      if (inst->m_selfId == a || inst->m_selfId == b)
      {
        inst->OnLinkFlip(a, b, up);
      }
    }
  }

  static bool IsLinkUp(uint32_t a, uint32_t b)
  {
    auto it = S().linkUp.find(Key(a, b));
    return (it != S().linkUp.end()) ? it->second : false;
  }

  void SetIpv4(Ptr<Ipv4> ipv4) override;

  Ptr<Ipv4Route> RouteOutput(Ptr<Packet> p, const Ipv4Header& header,
                             Ptr<NetDevice> oif, Socket::SocketErrno& sockerr) override;

  bool RouteInput(Ptr<const Packet> p,
                  const Ipv4Header& header,
                  Ptr<const NetDevice> idev,
                  const UnicastForwardCallback& ucb,
                  const MulticastForwardCallback& mcb,
                  const LocalDeliverCallback& lcb,
                  const ErrorCallback& ecb) override;

  void NotifyInterfaceUp(uint32_t) override {}
  void NotifyInterfaceDown(uint32_t) override {}
  void NotifyAddAddress(uint32_t, Ipv4InterfaceAddress) override { RebuildFwdTable(); }
  void NotifyRemoveAddress(uint32_t, Ipv4InterfaceAddress) override { RebuildFwdTable(); }

  void PrintRoutingTable(Ptr<OutputStreamWrapper> stream, Time::Unit) const override;

  void PrintQTable(std::ostream& os, const std::vector<bool>* dstMask = nullptr) const;
  void DumpDropSummary(std::ostream& os) const;

  uint64_t GetProbesTx() const { return m_probesTx; }
  uint64_t GetProbesRx() const { return m_probesRx; }
  double GetNeighborCostMs(uint32_t neiId) const;

  // ---- Control overhead getters (totals) ----
  uint64_t GetCtrlPktsTx() const { return m_ctrlPktsTx; }
  uint64_t GetCtrlPktsRx() const { return m_ctrlPktsRx; }
  uint64_t GetCtrlPayloadTx() const { return m_ctrlPayloadTx; }
  uint64_t GetCtrlPayloadRx() const { return m_ctrlPayloadRx; }
  uint64_t GetCtrlWireTx() const { return m_ctrlWireTx; }
  uint64_t GetCtrlWireRx() const { return m_ctrlWireRx; }

  // ---- Control overhead getters (by type) ----
  uint64_t GetProbePktsTx() const { return m_probePktsTx; }
  uint64_t GetProbePktsRx() const { return m_probePktsRx; }
  uint64_t GetProbePayloadTx() const { return m_probePayloadTx; }
  uint64_t GetProbePayloadRx() const { return m_probePayloadRx; }
  uint64_t GetProbeWireTx() const { return m_probeWireTx; }
  uint64_t GetProbeWireRx() const { return m_probeWireRx; }

  uint64_t GetProbeAckPktsTx() const { return m_packPktsTx; }
  uint64_t GetProbeAckPktsRx() const { return m_packPktsRx; }
  uint64_t GetProbeAckPayloadTx() const { return m_packPayloadTx; }
  uint64_t GetProbeAckPayloadRx() const { return m_packPayloadRx; }
  uint64_t GetProbeAckWireTx() const { return m_packWireTx; }
  uint64_t GetProbeAckWireRx() const { return m_packWireRx; }

  uint64_t GetFeedbackPktsTx() const { return m_fbPktsTx; }
  uint64_t GetFeedbackPktsRx() const { return m_fbPktsRx; }
  uint64_t GetFeedbackPayloadTx() const { return m_fbPayloadTx; }
  uint64_t GetFeedbackPayloadRx() const { return m_fbPayloadRx; }
  uint64_t GetFeedbackWireTx() const { return m_fbWireTx; }
  uint64_t GetFeedbackWireRx() const { return m_fbWireRx; }

private:
  static uint64_t RlKey(uint32_t nodeId, uint32_t dstId)
  {
    return (static_cast<uint64_t>(nodeId) << 32) | static_cast<uint64_t>(dstId);
  }

  static std::atomic<bool> s_rlEnabled;
  static std::atomic<bool> s_rlUseDisabled;
  static std::atomic<uint64_t> s_currentEnvStamp;
  static RlPickFn s_rlPick;
  static std::unordered_map<uint64_t, int> s_forced;
  static std::mutex s_forcedMu;

  struct Adj
  {
    uint32_t    neiId{0};
    Ipv4Address myIfAddr;
    Ipv4Address neiIfAddr;
    uint32_t    myOif{0};
    double      ewmaCostMs{150.0};
  };

  struct FwdEntry
  {
    Ipv4Address nh;
    uint32_t    oif{0};
  };

  struct PendingProbe
  {
    Time sent;
    uint32_t neighbor{0};
  };

  enum CtrlType : uint8_t
  {
    CTRL_FEEDBACK  = 0xF1,
    CTRL_PROBE     = 0xA1,
    CTRL_PROBE_ACK = 0xA2
  };

  struct QTag : public Tag
  {
    static constexpr size_t kHist = 32;
    uint32_t prevNodeId{UINT32_MAX};
    uint32_t dstNodeId{UINT32_MAX};
    uint32_t lastHopId{UINT32_MAX};
    uint32_t sendStampMs{0};
    uint8_t  hopCount{0};
    uint8_t  historyLen{0};
    std::array<uint32_t, kHist> history{};

    QTag()
    {
      history.fill(UINT32_MAX);
    }

    static TypeId GetTypeId()
    {
      static TypeId tid = TypeId("ns3::QTableRouting::QTag")
          .SetParent<Tag>()
          .AddConstructor<QTag>();
      return tid;
    }

    TypeId GetInstanceTypeId() const override { return GetTypeId(); }

    uint32_t GetSerializedSize() const override
    {
      return 4u + 4u + 4u + 4u + 1u + 1u + static_cast<uint32_t>(kHist * sizeof(uint32_t));
    }

    void Serialize(TagBuffer i) const override
    {
      i.WriteU32(prevNodeId);
      i.WriteU32(dstNodeId);
      i.WriteU32(lastHopId);
      i.WriteU32(sendStampMs);
      i.WriteU8(hopCount);
      i.WriteU8(historyLen);
      for (size_t idx = 0; idx < kHist; ++idx)
      {
        i.WriteU32(history[idx]);
      }
    }

    void Deserialize(TagBuffer i) override
    {
      prevNodeId = i.ReadU32();
      dstNodeId  = i.ReadU32();
      lastHopId  = i.ReadU32();
      sendStampMs = i.ReadU32();
      hopCount   = i.ReadU8();
      historyLen = std::min<uint8_t>(i.ReadU8(), kHist);
      for (uint32_t idx = 0; idx < kHist; ++idx)
      {
        history[idx] = i.ReadU32();
      }
    }

    void Print(std::ostream& os) const override
    {
      os << "QTag(prev=" << prevNodeId << ", dst=" << dstNodeId
         << ", lastHop=" << lastHopId << ", hop=" << unsigned(hopCount) << ")";
    }

    void ResetHistory()
    {
      historyLen = 0;
      history.fill(UINT32_MAX);
      hopCount = 0;
    }

    void AppendVisit(uint32_t nodeId)
    {
      if (historyLen < kHist)
      {
        history[historyLen++] = nodeId;
      }
      else
      {
        for (size_t i = 1; i < kHist; ++i)
        {
          history[i - 1] = history[i];
        }
        history[kHist - 1] = nodeId;
      }
    }

    bool HasVisited(uint32_t nodeId) const
    {
      for (uint8_t i = 0; i < historyLen; ++i)
      {
        if (history[i] == nodeId)
        {
          return true;
        }
      }
      return false;
    }
  };

  void Start();
  void Stop();
  void AttachDropTrace();
  void ProbeTick();
  void HandleCtrlRx(Ptr<Socket> sock);
  void ProcessFeedback(const uint8_t* buf, uint32_t len);
  void ProcessProbe(const uint8_t* buf, uint32_t len, const Address& from);
  void ProcessProbeAck(const uint8_t* buf, uint32_t len);
  void SendFeedback(uint32_t prevNodeId, uint32_t destId, double costMs, double downstreamBestMs, uint8_t flags);
  void SendProbe(uint32_t neiId, const Adj& adj);
  void SendProbeAck(uint32_t seq, uint32_t requesterId, const Address& from);
  void NotifyDrop(const Ipv4Header& header, Ptr<const Packet> packet, Ipv4L3Protocol::DropReason reason, Ptr<Ipv4> ipv4, uint32_t interface);
  void ApplyCostUpdate(uint32_t dst,
                       uint32_t nei,
                       double costMs,
                       double downstreamBestMs,
                       const char* reason,
                       uint8_t flags,
                       bool forceUpdate = false);
  double& EnsureQValue(uint32_t dst, uint32_t nei);
  double GetQValue(uint32_t dst, uint32_t nei) const;
  double GetBestNeighborCost(uint32_t dst, uint32_t skipNei = UINT32_MAX) const;
  int32_t GetSpfNextHop(uint32_t dst) const;
  bool IsDestinationTracked(uint32_t dst) const;
  double CurrentEps() const;
  uint32_t LookupNodeId(Ipv4Address addr) const;
  Ipv4Address SelectSource(uint32_t oif) const;
  bool IsLocalAddress(Ipv4Address a) const;
  Ptr<Ipv4Route> BuildStaticRoute(Ipv4Address dst, Socket::SocketErrno& sockerr);
  void RebuildFwdTable();
  void MaybeLogDecision(uint32_t dst, uint32_t chosen, bool exploring, const std::vector<std::pair<uint32_t,double>>& candidates);
  void MaybeLogUpdate(uint32_t dst, uint32_t nei, double oldVal, double newVal, double costMs, double downstreamBestMs, const char* reason, uint8_t flags);
  static bool ShouldEmitLog(bool countPacket);
  void OnLinkFlip(uint32_t u, uint32_t v, bool up);

  struct Shared
  {
    std::unordered_map<uint64_t, bool> linkUp;
    std::map<uint32_t, Ipv4Address> nodeAddrs;
    struct LogLimiter
    {
      uint64_t packetStride{5000};
      Time simStride{Seconds(5.0)};
      uint64_t totalPackets{0};
      uint64_t lastPacketAtLog{0};
      Time lastLogSim{Time()};
      bool hasLogged{false};
    } log;
  };

  static uint64_t Key(uint32_t a, uint32_t b)
  {
    uint32_t lo = std::min(a, b);
    uint32_t hi = std::max(a, b);
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
  }

  static Shared& S()
  {
    static Shared s;
    return s;
  }

  static std::vector<QTableRouting*>& Registry()
  {
    static std::vector<QTableRouting*> r;
    return r;
  }

  static uint32_t NodeCount()
  {
    if (S().nodeAddrs.empty())
    {
      return 0;
    }
    return S().nodeAddrs.rbegin()->first + 1u;
  }

  Ptr<Ipv4>   m_ipv4;
  uint32_t    m_selfId{UINT32_MAX};
  Ipv4Address m_selfAddr{"0.0.0.0"};
  bool        m_started{false};

  std::map<uint32_t, Adj>          m_nei;
  std::map<Ipv4Address, FwdEntry>  m_fwd;
  std::map<uint32_t, Ipv4Address>  m_nodeAddrs;
  std::map<Ipv4Address, uint32_t>  m_addrToNode;
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, double>> m_q;
  std::unordered_map<uint32_t, uint32_t> m_lastChoice;

  // RL override table (per-destination temporary next-hop selection)
  struct RlOverrideEntry {
    uint32_t nextHop = std::numeric_limits<uint32_t>::max();
    double   expiry  = 0.0; // sim time seconds
    uint64_t stamp   = 0;   // env step stamp; 0 = legacy/unknown
  };
  std::unordered_map<uint32_t, RlOverrideEntry> m_rlOverride; // key: dstId
  std::unordered_map<uint32_t, double> m_teacherPin; // dstId -> expiry seconds

  double   m_alpha{0.2};
  double   m_gamma{0.9};
  double   m_eps{0.1};
  Time     m_probeInterval{Seconds(0.3)};
  uint32_t m_probeFanout{2};
  Time     m_penaltyDrop{MilliSeconds(500)};
  double   m_penaltyLoopMs{125.0};
  uint16_t m_ctrlPort{8899};
  EventId  m_probeEv;

  Ptr<Socket> m_ctrlSock;
  Ptr<UniformRandomVariable> m_rng01;
  Ptr<UniformRandomVariable> m_rngIdx;

  // ---- Control overhead accumulators ----
  uint64_t m_ctrlPktsTx{0};
  uint64_t m_ctrlPktsRx{0};
  uint64_t m_ctrlPayloadTx{0};
  uint64_t m_ctrlPayloadRx{0};
  uint64_t m_ctrlWireTx{0};
  uint64_t m_ctrlWireRx{0};

  uint64_t m_probePktsTx{0};
  uint64_t m_probePktsRx{0};
  uint64_t m_probePayloadTx{0};
  uint64_t m_probePayloadRx{0};
  uint64_t m_probeWireTx{0};
  uint64_t m_probeWireRx{0};

  uint64_t m_packPktsTx{0};
  uint64_t m_packPktsRx{0};
  uint64_t m_packPayloadTx{0};
  uint64_t m_packPayloadRx{0};
  uint64_t m_packWireTx{0};
  uint64_t m_packWireRx{0};

  uint64_t m_fbPktsTx{0};
  uint64_t m_fbPktsRx{0};
  uint64_t m_fbPayloadTx{0};
  uint64_t m_fbPayloadRx{0};
  uint64_t m_fbWireTx{0};
  uint64_t m_fbWireRx{0};

  static constexpr uint32_t kIpHdr = 20;  // IPv4 header bytes
  static constexpr uint32_t kUdpHdr = 8;  // UDP header bytes
  static constexpr uint32_t kL3L4   = kIpHdr + kUdpHdr;

  uint64_t m_probesTx{0};
  uint64_t m_probesRx{0};

  // --- SP seeding & fallback ---
  bool     m_seedFromSpf {false};
  double   m_spSeedMs    {3.0};     // base ms per hop for initial Q
  double   m_spfRefreshS {0.0};     // 0 = no periodic refresh
  std::vector<int32_t>  m_spNextHop;   // index by dst nodeId; value = neighbor nodeId (or -1)
  static constexpr int32_t INVALID_NH = -1;
  EventId  m_spfRefreshEv;
  // freeze-after-measure support
  bool     m_freezeAfterMeasure{false};
  double   m_measureStartS{0.0};
  EventId  m_freezeEv;
  static constexpr uint32_t kDropReasonBuckets = static_cast<uint32_t>(Ipv4L3Protocol::DROP_DUPLICATE) + 1;
  std::array<uint64_t, kDropReasonBuckets> m_dropByReason{};
  uint64_t m_decisionCount{0};
  double   m_updateLogThreshold{5.0};
  double   m_seedHopMs{10.0};
  double   m_defaultCostMs{200.0};
  double   m_maxCostMs{100000.0};
  double   m_epsStart{0.15};
  double   m_epsFinal{0.02};
  double   m_epsTau{30.0};
  std::vector<bool> m_dstMask;
  std::unordered_map<uint32_t, uint16_t> m_staticHops;
  uint32_t m_updateStride{1};
  uint64_t m_pktsSinceUpdate{0};

  uint16_t m_nextProbeSeq{1};
  std::unordered_map<uint16_t, PendingProbe> m_pendingProbes;
  bool     m_dropTraceConnected{false};

  inline static bool m_loggedGlobalConfig = false;
  inline static bool s_logRebuild = false;
  // Global knobs (with conservative defaults)
  inline static double s_probeEwmaW = 0.20;  // weight on new RTT sample
  inline static double s_switchHyst = 0.05;  // allowable % worse to keep current NH
  inline static bool   s_overrideAE = false; // if true, use s_alpha/s_eps
  inline static double s_alpha = 0.2;
  inline static double s_eps   = 0.1;
};

inline QTableRouting::QTableRouting()
{
  Registry().push_back(this);
}

inline QTableRouting::~QTableRouting()
{
  Stop();
  auto& r = Registry();
  r.erase(std::remove(r.begin(), r.end(), this), r.end());
}

inline void QTableRouting::Configure(double alpha,
                                     double gamma,
                                     double eps,
                                     Time probeInterval,
                                     uint32_t probeFanout,
                                     Time penaltyDrop,
                                     uint16_t ctrlPort,
                                     Ptr<Object>)
{
  m_alpha = s_overrideAE ? s_alpha : std::clamp(alpha, 0.0, 1.0);
  m_gamma = std::clamp(gamma, 0.0, 1.0);
  m_eps   = s_overrideAE ? s_eps   : std::clamp(eps,   0.0, 1.0);
  m_epsStart = m_eps;
  if (m_epsFinal > m_epsStart)
  {
    m_epsFinal = m_epsStart;
  }
  m_probeInterval = probeInterval;
  m_probeFanout   = std::max<uint32_t>(1u, probeFanout);
  m_penaltyDrop   = penaltyDrop;
  m_ctrlPort      = ctrlPort;
  m_penaltyLoopMs = std::max(25.0, 0.25 * m_penaltyDrop.GetMilliSeconds());

  if (!m_loggedGlobalConfig)
  {
    m_loggedGlobalConfig = true;
    NS_LOG_INFO("QTableRouting config alpha=" << m_alpha
                 << " gamma=" << m_gamma
                 << " eps=" << m_eps
                 << " probe=" << m_probeInterval.GetSeconds()
                 << " fanout=" << m_probeFanout
                 << " penaltyDropMs=" << m_penaltyDrop.GetMilliSeconds()
                 << " ctrlPort=" << m_ctrlPort);
  }

  if (m_ipv4 && !m_started)
  {
    Start();
  }
}

inline void QTableRouting::SetDestinationMask(const std::vector<bool>& mask)
{
  m_dstMask = mask;
  if (m_dstMask.empty())
  {
    return;
  }
  for (auto it = m_q.begin(); it != m_q.end();)
  {
    if (!IsDestinationTracked(it->first))
    {
      it = m_q.erase(it);
    }
    else
    {
      ++it;
    }
  }
  for (auto it = m_lastChoice.begin(); it != m_lastChoice.end();)
  {
    if (!IsDestinationTracked(it->first))
    {
      it = m_lastChoice.erase(it);
    }
    else
    {
      ++it;
    }
  }
}

inline void QTableRouting::SetSeedHopMs(double v)
{
  if (std::isfinite(v) && v > 0.0)
  {
    m_seedHopMs = v;
  }
}

inline void QTableRouting::SetEpsSchedule(double start, double final, double tau)
{
  m_epsStart = std::clamp(start, 0.0, 1.0);
  m_epsFinal = std::clamp(final, 0.0, 1.0);
  if (m_epsFinal > m_epsStart)
  {
    m_epsFinal = m_epsStart;
  }
  m_epsTau = std::max(1e-6, tau);
}

inline void QTableRouting::SetUpdateStride(uint32_t n)
{
  m_updateStride = std::max<uint32_t>(1, n);
  // Suppress noisy per-node log on every run
  // NS_LOG_INFO("QTableRouting(" << m_selfId << ") update stride = " << m_updateStride);
}

inline void QTableRouting::RegisterNodeAddress(uint32_t nodeId, Ipv4Address nodeAddr)
{
  m_nodeAddrs[nodeId] = nodeAddr;
  m_addrToNode[nodeAddr] = nodeId;
  S().nodeAddrs[nodeId] = nodeAddr;
  if (nodeId == m_selfId)
  {
    m_selfAddr = nodeAddr;
  }
}

inline void QTableRouting::AddAdjacency(uint32_t neiId, Ipv4Address myIfAddr, Ipv4Address neiIfAddr, uint32_t myOif)
{
  Adj a;
  a.neiId = neiId;
  a.myIfAddr = myIfAddr;
  a.neiIfAddr = neiIfAddr;
  a.myOif = myOif;
  m_nei[neiId] = a;
}

inline void QTableRouting::KickRebuild()
{
  RebuildFwdTable();
}

inline void QTableRouting::SetLogPolicy(uint64_t logEveryPackets, Time logEverySimTime)
{
  Shared& shared = S();
  shared.log.packetStride = logEveryPackets;
  shared.log.simStride = logEverySimTime;
  shared.log.totalPackets = 0;
  shared.log.lastPacketAtLog = 0;
  shared.log.lastLogSim = Time();
  shared.log.hasLogged = false;
}

inline void QTableRouting::SetIpv4(Ptr<Ipv4> ipv4)
{
  m_ipv4 = ipv4;
  Ptr<Node> node = ipv4 ? ipv4->GetObject<Node>() : nullptr;
  if (node)
  {
    m_selfId = node->GetId();
  }
  auto it = m_nodeAddrs.find(m_selfId);
  if (it != m_nodeAddrs.end())
  {
    m_selfAddr = it->second;
  }

  if (!m_rng01)
  {
    m_rng01 = CreateObject<UniformRandomVariable>();
    m_rng01->SetStream(0x7000 + m_selfId);
  }
  if (!m_rngIdx)
  {
    m_rngIdx = CreateObject<UniformRandomVariable>();
    m_rngIdx->SetStream(0x9000 + m_selfId);
  }

  AttachDropTrace();

  if (!m_started)
  {
    Simulator::ScheduleNow(&QTableRouting::Start, this);
  }
}

inline void QTableRouting::Start()
{
  if (m_started)
  {
    return;
  }
  m_started = true;

  if (!m_ctrlSock)
  {
    Ptr<Node> node = m_ipv4 ? m_ipv4->GetObject<Node>() : nullptr;
    if (node)
    {
      m_ctrlSock = Socket::CreateSocket(node, UdpSocketFactory::GetTypeId());
      InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), m_ctrlPort);
      int br = m_ctrlSock->Bind(local);
      NS_ASSERT_MSG(br == 0, "Control socket bind failed");
      m_ctrlSock->SetRecvCallback(MakeCallback(&QTableRouting::HandleCtrlRx, this));
    }
  }

  if (!m_probeInterval.IsZero())
  {
    m_probeEv = Simulator::Schedule(Seconds(0.20), &QTableRouting::ProbeTick, this);
  }

  if (m_fwd.empty())
  {
    Simulator::Schedule(MilliSeconds(200), &QTableRouting::RebuildFwdTable, this);
  }

  // Optional: schedule SP refresh (cheap path uses same rebuild for adjacency snapshot)
  if (m_spfRefreshS > 0.0)
  {
    m_spfRefreshEv = Simulator::Schedule(Seconds(m_spfRefreshS), [this]() {
      this->RebuildFwdTable();
      // reschedule
      if (m_spfRefreshS > 0.0)
      {
        this->m_spfRefreshEv = Simulator::Schedule(Seconds(m_spfRefreshS), [this]() {
          this->RebuildFwdTable();
        });
      }
    });
  }

  // Optional: freeze learning (alpha=0, eps=0) after measurement start
  if (m_freezeAfterMeasure)
  {
    Time when = Seconds(std::max(0.0, m_measureStartS));
    m_freezeEv = Simulator::Schedule(when, [this]() {
      this->m_alpha = 0.0;
      this->m_eps = 0.0;
      this->m_epsStart = 0.0;
      this->m_epsFinal = 0.0;
      // leave gamma unchanged
    });
  }
}

inline void QTableRouting::Stop()
{
  if (m_probeEv.IsPending())
  {
    Simulator::Cancel(m_probeEv);
  }
  if (m_spfRefreshEv.IsPending())
  {
    Simulator::Cancel(m_spfRefreshEv);
  }
  if (m_freezeEv.IsPending())
  {
    Simulator::Cancel(m_freezeEv);
  }

  if (m_ctrlSock)
  {
    m_ctrlSock->SetRecvCallback(MakeNullCallback<void, Ptr<Socket>>());
    m_ctrlSock->Close();
    m_ctrlSock = nullptr;
  }

  m_started = false;
}

inline void QTableRouting::AttachDropTrace()
{
  if (m_dropTraceConnected || !m_ipv4)
  {
    return;
  }
  Ptr<Ipv4L3Protocol> l3 = m_ipv4->GetObject<Ipv4L3Protocol>();
  if (!l3)
  {
    return;
  }
  l3->TraceConnectWithoutContext("Drop", MakeCallback(&QTableRouting::NotifyDrop, this));
  m_dropTraceConnected = true;
}

inline Ptr<Ipv4Route> QTableRouting::BuildStaticRoute(Ipv4Address dst, Socket::SocketErrno& sockerr)
{
  for (const auto& kv : m_nei)
  {
    const Adj& adj = kv.second;
    if (adj.neiIfAddr == dst)
    {
      Ptr<Ipv4Route> rt = Create<Ipv4Route>();
      rt->SetDestination(dst);
      rt->SetGateway(adj.neiIfAddr);
      rt->SetOutputDevice(m_ipv4->GetNetDevice(adj.myOif));
      rt->SetSource(SelectSource(adj.myOif));
      sockerr = Socket::ERROR_NOTERROR;
      return rt;
    }
  }

  auto it = m_fwd.find(dst);
  if (it == m_fwd.end())
  {
    sockerr = Socket::ERROR_NOROUTETOHOST;
    return nullptr;
  }

  Ptr<Ipv4Route> rt = Create<Ipv4Route>();
  rt->SetDestination(dst);
  rt->SetGateway(it->second.nh);
  rt->SetOutputDevice(m_ipv4->GetNetDevice(it->second.oif));
  rt->SetSource(SelectSource(it->second.oif));
  sockerr = Socket::ERROR_NOTERROR;
  return rt;
}

inline Ptr<Ipv4Route> QTableRouting::RouteOutput(Ptr<Packet> p, const Ipv4Header& header,
                                                 Ptr<NetDevice> /*oif*/, Socket::SocketErrno& sockerr)
{
  sockerr = Socket::ERROR_NOTERROR;
  if (!m_ipv4)
  {
    sockerr = Socket::ERROR_NOROUTETOHOST;
    return nullptr;
  }
  // Apply any global alpha/epsilon override on the fast path
  if (s_overrideAE)
  {
    m_alpha = s_alpha;
    m_eps   = s_eps;
  }

  Ipv4Address dstAddr = header.GetDestination();
  uint32_t dstNodeId = LookupNodeId(dstAddr);

  if (dstNodeId == UINT32_MAX || dstNodeId == m_selfId)
  {
    return BuildStaticRoute(dstAddr, sockerr);
  }

  if (!IsDestinationTracked(dstNodeId))
  {
    Ptr<Ipv4Route> fallback = BuildStaticRoute(dstAddr, sockerr);
    if (fallback)
    {
      return fallback;
    }
    sockerr = Socket::ERROR_NOROUTETOHOST;
    return nullptr;
  }

  QTag tag;
  bool hadTag = p->PeekPacketTag(tag);

  std::vector<uint32_t> upNeis;
  upNeis.reserve(m_nei.size());
  for (const auto& kv : m_nei)
  {
    if (IsLinkUp(m_selfId, kv.first))
    {
      upNeis.push_back(kv.first);
    }
  }

  if (upNeis.empty())
  {
    Ptr<Ipv4Route> fallback = BuildStaticRoute(dstAddr, sockerr);
    if (fallback)
    {
      return fallback;
    }
    sockerr = Socket::ERROR_NOROUTETOHOST;
    return nullptr;
  }

  auto buildOverrideRoute = [&](uint32_t neighbor) -> Ptr<Ipv4Route>
  {
    auto adjIt = m_nei.find(neighbor);
    if (adjIt == m_nei.end())
    {
      return nullptr;
    }
    Adj* adj = &adjIt->second;
    if (adj->myOif >= m_ipv4->GetNInterfaces())
    {
      return nullptr;
    }
    Ptr<NetDevice> outDev = m_ipv4->GetNetDevice(adj->myOif);
    if (!outDev)
    {
      return nullptr;
    }

    if (hadTag)
    {
      p->RemovePacketTag(tag);
    }
    else
    {
      tag.ResetHistory();
    }
    tag.prevNodeId = m_selfId;
    tag.dstNodeId  = dstNodeId;
    tag.lastHopId  = neighbor;
    tag.sendStampMs = Simulator::Now().GetMilliSeconds();
    if (!tag.HasVisited(m_selfId))
    {
      tag.AppendVisit(m_selfId);
    }
    if (tag.hopCount < 255)
    {
      ++tag.hopCount;
    }
    p->AddPacketTag(tag);

    ++g_rlOverridePkts;
    if (g_logRl)
    {
      std::cout << "[ROUTER][OVR] node=" << m_selfId
                << " dst=" << dstNodeId
                << " via=" << neighbor
                << " overrides=" << g_rlOverridePkts
                << "\n";
    }
    Ptr<Ipv4Route> rt = Create<Ipv4Route>();
    rt->SetDestination(dstAddr);
    rt->SetGateway(adj->neiIfAddr);
    rt->SetOutputDevice(outDev);
    rt->SetSource(adj->myIfAddr);
    return rt;
  };

  {
    int forcedNext = -1;
    {
      std::lock_guard<std::mutex> lk(s_forcedMu);
      auto it = s_forced.find(RlKey(m_selfId, dstNodeId));
      if (it != s_forced.end())
      {
        forcedNext = it->second;
        s_forced.erase(it);
      }
    }
    if (forcedNext >= 0)
    {
      uint32_t forcedNh = static_cast<uint32_t>(forcedNext);
      if (IsLinkUp(m_selfId, forcedNh) &&
          std::find(upNeis.begin(), upNeis.end(), forcedNh) != upNeis.end())
      {
        if (Ptr<Ipv4Route> rt = buildOverrideRoute(forcedNh))
        {
          return rt;
        }
      }
    }
  }

  // Prioritize cached RL override next-hop, if present (packet-level)
  {
    uint32_t rlNei = std::numeric_limits<uint32_t>::max();
    if (GetRlNextHop(dstNodeId, rlNei))
    {
      if (IsLinkUp(m_selfId, rlNei) &&
          std::find(upNeis.begin(), upNeis.end(), rlNei) != upNeis.end())
      {
        if (Ptr<Ipv4Route> rt = buildOverrideRoute(rlNei))
        {
          ++g_rlPktUsed;
          // NS_LOG_UNCOND("[ROUTER][USE_RL][FWD] node=" << m_selfId
          //                << " dst=" << dstNodeId
          //                << " nextHop=" << rlNei
          //                << " pktCount=" << g_rlPktUsed);
          // NS_LOG_UNCOND("[ROUTER][FWD] node=" << m_selfId
          //                << " dst=" << dstNodeId
          //                << " via=" << rlNei
          //                << " reason=rl");
          return rt;
        }
      }
    }
  }

  if (s_rlEnabled.load(std::memory_order_relaxed))
  {
    // Probe whether an RL override is currently present for this dst
    {
      uint32_t probeNh = std::numeric_limits<uint32_t>::max();
      (void) GetRlNextHop(dstNodeId, probeNh);
      (void) probeNh;
    }

    RlPickFn picker = s_rlPick;
    if (picker)
    {
      int rlChoice = picker(m_selfId, dstNodeId, upNeis);
      if (g_logRl)
      {
        std::cout << "[ROUTER][PICK] node=" << m_selfId
                  << " dst=" << dstNodeId
                  << " idx=" << rlChoice
                  << " K=" << upNeis.size()
                  << " pin=" << (IsTeacherPinned(dstNodeId) ? 1 : 0)
                  << "\n";
      }
      if (rlChoice >= 0)
      {
        uint32_t chosenOverride = static_cast<uint32_t>(rlChoice);
        if (IsLinkUp(m_selfId, chosenOverride) &&
            std::find(upNeis.begin(), upNeis.end(), chosenOverride) != upNeis.end())
        {
        if (Ptr<Ipv4Route> rt = buildOverrideRoute(chosenOverride))
        {
          // Packet-level confirmation: RL override actively used for forwarding
          ++g_rlPktUsed;
          // NS_LOG_UNCOND("[ROUTER][USE_RL][FWD] node=" << m_selfId
          //                << " dst=" << dstNodeId
          //                << " nextHop=" << chosenOverride
          //                << " pktCount=" << g_rlPktUsed);
          if (g_logRl)
          {
            std::cout << "[RL/APPLY] node=" << m_selfId
                      << " dst=" << dstNodeId
                      << " nh=" << chosenOverride
                      << "\n";
          }
          return rt;
        }
      }
    }
    }
  }

  if (hadTag && upNeis.size() > 1)
  {
    std::vector<uint32_t> filtered;
    filtered.reserve(upNeis.size());
    for (uint32_t nid : upNeis)
    {
      if (nid != tag.prevNodeId)
      {
        filtered.push_back(nid);
      }
    }
    if (!filtered.empty())
    {
      upNeis.swap(filtered);
    }
  }

  // SP fallback: if row is fresh/empty, prefer SP next-hop for this dst
  bool spFallback = false;
  bool exploring = false;
  uint32_t chosen = upNeis.front();
  auto rowIt = m_q.find(dstNodeId);
  bool freshRow = (rowIt == m_q.end()) || rowIt->second.empty();
  if (m_seedFromSpf && freshRow && dstNodeId < m_spNextHop.size())
  {
    int32_t spNh = m_spNextHop[dstNodeId];
    if (spNh != INVALID_NH && IsLinkUp(m_selfId, static_cast<uint32_t>(spNh)))
    {
      chosen = static_cast<uint32_t>(spNh);
      spFallback = true;
    }
  }

  // Ensure seed values before selection if not taking immediate SP fallback
  if (!spFallback)
  {
    for (uint32_t nid : upNeis)
    {
      EnsureQValue(dstNodeId, nid);
    }

    exploring = false; // exploration disabled
    // if (exploring) { ... } else
    {
      double bestCost = std::numeric_limits<double>::max();
      // Prefer SP next-hop on ties
      int32_t prefSp = (dstNodeId < m_spNextHop.size()) ? m_spNextHop[dstNodeId] : INVALID_NH;
      for (uint32_t nid : upNeis)
      {
        double q = GetQValue(dstNodeId, nid);
        if (q < bestCost)
        {
          bestCost = q;
          chosen = nid;
        }
        else if (std::abs(q - bestCost) < 1e-6)
        {
          if (static_cast<int32_t>(nid) == prefSp && static_cast<int32_t>(chosen) != prefSp)
          {
            chosen = nid;
          }
          else if (static_cast<int32_t>(chosen) != prefSp && nid < chosen)
          {
            chosen = nid;
          }
        }
      }
      // Hysteresis: keep current next-hop unless new one improves by s_switchHyst
      auto curIt = m_fwd.find(dstAddr);
      if (curIt != m_fwd.end())
      {
        const uint32_t currentNei = LookupNodeId(curIt->second.nh);
        if (currentNei != UINT32_MAX)
        {
          const double curCost = GetQValue(dstNodeId, currentNei);
          if (IsLinkUp(m_selfId, currentNei) && curCost <= bestCost * (1.0 + s_switchHyst))
          {
            chosen = currentNei;
          }
        }
      }
    }
  }

  Adj* adj = nullptr;
  auto adjIt = m_nei.find(chosen);
  if (adjIt != m_nei.end())
  {
    adj = &adjIt->second;
  }
  if (!adj)
  {
    sockerr = Socket::ERROR_NOROUTETOHOST;
    return nullptr;
  }

  if (adj->myOif >= m_ipv4->GetNInterfaces())
  {
    sockerr = Socket::ERROR_NOROUTETOHOST;
    return nullptr;
  }

  Ptr<NetDevice> outDev = m_ipv4->GetNetDevice(adj->myOif);
  if (!outDev)
  {
    sockerr = Socket::ERROR_NOROUTETOHOST;
    return nullptr;
  }

  if (hadTag)
  {
    p->RemovePacketTag(tag);
  }
  else
  {
    tag.ResetHistory();
  }
  // Forwarding trace disabled
  tag.prevNodeId = m_selfId;
  tag.dstNodeId  = dstNodeId;
  tag.lastHopId  = chosen;
  tag.sendStampMs = Simulator::Now().GetMilliSeconds();
  if (!tag.HasVisited(m_selfId))
  {
    tag.AppendVisit(m_selfId);
  }
  if (tag.hopCount < 255)
  {
    ++tag.hopCount;
  }
  p->AddPacketTag(tag);

  std::vector<std::pair<uint32_t,double>> cands;
  cands.reserve(upNeis.size());
  for (uint32_t nid : upNeis)
  {
    cands.emplace_back(nid, GetQValue(dstNodeId, nid));
  }
  MaybeLogDecision(dstNodeId, chosen, exploring, cands);

  Ptr<Ipv4Route> rt = Create<Ipv4Route>();
  rt->SetDestination(dstAddr);
  rt->SetGateway(adj->neiIfAddr);
  rt->SetOutputDevice(outDev);
  rt->SetSource(adj->myIfAddr);
  return rt;
}

inline bool QTableRouting::RouteInput(Ptr<const Packet> p,
                                      const Ipv4Header& header,
                                      Ptr<const NetDevice> /*idev*/,
                                      const UnicastForwardCallback& ucb,
                                      const MulticastForwardCallback& /*mcb*/,
                                      const LocalDeliverCallback& lcb,
                                      const ErrorCallback& ecb)
{
  Ipv4Address dst = header.GetDestination();

  if (dst == m_selfAddr || IsLocalAddress(dst))
  {
    if (!lcb.IsNull())
    {
      uint32_t iface = m_ipv4->GetInterfaceForAddress(dst);
      lcb(p, header, iface);
      return true;
    }
    return false;
  }

  uint32_t dstNodeId = LookupNodeId(dst);

  if (dstNodeId == UINT32_MAX || dstNodeId == m_selfId)
  {
    Socket::SocketErrno err;
    Ptr<Ipv4Route> rt = BuildStaticRoute(dst, err);
    if (rt && !ucb.IsNull())
    {
      ucb(rt, p->Copy(), header);
      return true;
    }
    if (!ecb.IsNull())
    {
      ecb(p, header, Socket::ERROR_NOROUTETOHOST);
    }
    return false;
  }

  QTag tag;
  const bool hadTag = p->PeekPacketTag(tag);
  if (hadTag)
  {
    const uint8_t kHopCap = static_cast<uint8_t>(std::min<uint32_t>(255u,
        2u * std::max<uint32_t>(8u, std::min<uint32_t>(NodeCount(), 32u))));
    if (tag.hopCount >= kHopCap)
    {
      if (tag.prevNodeId != UINT32_MAX)
      {
        auto adjIt = m_nei.find(tag.prevNodeId);
        if (adjIt != m_nei.end())
        {
          double loopCost = m_penaltyLoopMs + 0.5 * m_penaltyDrop.GetMilliSeconds();
          adjIt->second.ewmaCostMs = std::min(adjIt->second.ewmaCostMs + loopCost, m_maxCostMs);
          double downstreamBest = GetBestNeighborCost(tag.dstNodeId, tag.prevNodeId);
          SendFeedback(tag.prevNodeId, tag.dstNodeId, loopCost, downstreamBest, 0x01);
        }
      }
      if (!ecb.IsNull())
      {
        ecb(p, header, Socket::ERROR_NOROUTETOHOST);
      }
      return false;
    }
  }

  if (hadTag && tag.prevNodeId != UINT32_MAX && tag.prevNodeId != m_selfId)
  {
    double nowMs = Simulator::Now().GetMilliSeconds();
    double sendMs = static_cast<double>(tag.sendStampMs);
    double hopCost = std::max(0.0, nowMs - sendMs);
    bool loopDetected = tag.HasVisited(m_selfId);
    if (loopDetected)
    {
      hopCost += m_penaltyLoopMs;
    }

    auto adjIt = m_nei.find(tag.prevNodeId);
    if (adjIt != m_nei.end())
    {
      adjIt->second.ewmaCostMs = (1.0 - s_probeEwmaW) * adjIt->second.ewmaCostMs + s_probeEwmaW * hopCost;
    }

    double downstreamBest = (dstNodeId == m_selfId) ? 0.0 : GetBestNeighborCost(dstNodeId, tag.prevNodeId);
    uint8_t flags = loopDetected ? 0x01 : 0x00;
    SendFeedback(tag.prevNodeId, dstNodeId, hopCost, downstreamBest, flags);
  }

  Ptr<Packet> copy = p->Copy();
  Socket::SocketErrno err;
  Ptr<Ipv4Route> rt = RouteOutput(copy, header, nullptr, err);
  if (rt && !ucb.IsNull())
  {
    ucb(rt, copy, header);
    return true;
  }

  if (!ecb.IsNull())
  {
    ecb(p, header, err);
  }
  return false;
}

inline void QTableRouting::PrintRoutingTable(Ptr<OutputStreamWrapper> stream, Time::Unit) const
{
  std::ostream* os = stream->GetStream();
  *os << "QTableRouting node=" << m_selfId << " entries=" << m_fwd.size() << "\n";
  for (const auto& kv : m_fwd)
  {
    *os << "  dst " << kv.first << " -> nh " << kv.second.nh << " oif " << kv.second.oif << "\n";
  }
}

inline void QTableRouting::PrintQTable(std::ostream& os, const std::vector<bool>* dstMask) const
{
  os << "Node " << m_selfId << " selfAddr=" << m_selfAddr << " neighbors=" << m_nei.size() << "\n";
  for (const auto& nei : m_nei)
  {
    os << "  nei " << nei.first << " costEWMA=" << nei.second.ewmaCostMs << "\n";
  }
  for (const auto& row : m_q)
  {
    if (dstMask)
    {
      uint32_t dst = row.first;
      if (dst >= dstMask->size() || !(*dstMask)[dst])
      {
        continue;
      }
    }
    os << "  dst " << row.first << ":";
    for (const auto& val : row.second)
    {
      os << " (" << val.first << " -> " << val.second << "ms)";
    }
    os << "\n";
  }
}

inline void QTableRouting::DumpDropSummary(std::ostream& os) const
{
  auto reasonName = [](uint32_t idx) -> const char* {
    switch (idx)
    {
      case 0: return "DROP_UNSPEC";
      case Ipv4L3Protocol::DROP_TTL_EXPIRED: return "DROP_TTL_EXPIRED";
      case Ipv4L3Protocol::DROP_NO_ROUTE: return "DROP_NO_ROUTE";
      case Ipv4L3Protocol::DROP_BAD_CHECKSUM: return "DROP_BAD_CHECKSUM";
      case Ipv4L3Protocol::DROP_INTERFACE_DOWN: return "DROP_INTERFACE_DOWN";
      case Ipv4L3Protocol::DROP_ROUTE_ERROR: return "DROP_ROUTE_ERROR";
      case Ipv4L3Protocol::DROP_FRAGMENT_TIMEOUT: return "DROP_FRAGMENT_TIMEOUT";
      case Ipv4L3Protocol::DROP_DUPLICATE: return "DROP_DUPLICATE";
      default: return "DROP_UNKNOWN";
    }
  };

  bool printedHeader = false;
  for (uint32_t idx = 0; idx < m_dropByReason.size(); ++idx)
  {
    uint64_t count = m_dropByReason[idx];
    if (count == 0)
    {
      continue;
    }
    if (!printedHeader)
    {
      os << "[DROP] node=" << m_selfId;
      printedHeader = true;
    }
    os << ' ' << reasonName(idx) << '=' << count;
  }
  if (printedHeader)
  {
    os << std::endl;
  }
}

inline double& QTableRouting::EnsureQValue(uint32_t dst, uint32_t nei)
{
  auto& row = m_q[dst];
  auto [it, inserted] = row.try_emplace(nei, 0.0);
  double& value = it->second;
  if (inserted || value <= 0.0)
  {
    uint32_t hops = 1;
    auto hopIt = m_staticHops.find(dst);
    if (hopIt != m_staticHops.end() && hopIt->second > 0)
    {
      uint32_t pathHops = hopIt->second;
      if (pathHops > 0)
      {
        hops = std::max<uint32_t>(1u, pathHops - 1u);
      }
    }
    double seed = 0.0;
    if (m_seedFromSpf && dst < m_spNextHop.size())
    {
      bool isSpNext = (m_spNextHop[dst] == static_cast<int32_t>(nei));
      double unit = m_spSeedMs;
      // Give SP next hop a strictly lower initial cost than alternatives
      seed = (isSpNext ? (hops * unit) : ((hops + 1) * unit));
    }
    else
    {
      seed = hops * m_seedHopMs;
    }
    if (!std::isfinite(seed) || seed <= 0.0)
    {
      seed = m_defaultCostMs;
    }
    value = std::clamp(seed, 0.0, m_maxCostMs);
  }
  else
  {
    value = std::clamp(value, 0.0, m_maxCostMs);
  }
  return value;
}

inline double QTableRouting::GetQValue(uint32_t dst, uint32_t nei) const
{
  auto it = m_q.find(dst);
  if (it == m_q.end())
  {
    return m_defaultCostMs;
  }
  auto jt = it->second.find(nei);
  if (jt == it->second.end())
  {
    return m_defaultCostMs;
  }
  return jt->second;
}

inline double QTableRouting::GetBestNeighborCost(uint32_t dst, uint32_t skipNei) const
{
  if (!IsDestinationTracked(dst))
  {
    return m_defaultCostMs;
  }
  double best = std::numeric_limits<double>::infinity();
  bool found = false;
  for (const auto& kv : m_nei)
  {
    if (!IsLinkUp(m_selfId, kv.first) || kv.first == skipNei)
    {
      continue;
    }
    double q = GetQValue(dst, kv.first);
    if (!found || q < best)
    {
      best = q;
      found = true;
    }
  }
  if (found)
  {
  return best;
  }
  return m_defaultCostMs;
}

inline int32_t QTableRouting::GetSpfNextHop(uint32_t dst) const
{
  if (dst < m_spNextHop.size())
  {
    return m_spNextHop[dst];
  }
  return INVALID_NH;
}

inline double QTableRouting::GetNeighborCostMs(uint32_t neiId) const
{
  auto it = m_nei.find(neiId);
  if (it == m_nei.end())
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return it->second.ewmaCostMs;
}

inline bool QTableRouting::IsDestinationTracked(uint32_t dst) const
{
  if (m_dstMask.empty())
  {
    return true;
  }
  if (dst >= m_dstMask.size())
  {
    return false;
  }
  return m_dstMask[dst];
}

inline double QTableRouting::CurrentEps() const
{
  double tau = std::max(1e-6, m_epsTau);
  double t = Simulator::Now().GetSeconds();
  double decay = std::exp(-t / tau);
  double eps = m_epsFinal + (m_epsStart - m_epsFinal) * decay;
  eps = std::clamp(eps, std::min(m_epsStart, m_epsFinal), std::max(m_epsStart, m_epsFinal));
  return eps;
}

inline uint32_t QTableRouting::LookupNodeId(Ipv4Address addr) const
{
  auto it = m_addrToNode.find(addr);
  if (it != m_addrToNode.end())
  {
    return it->second;
  }
  for (const auto& kv : S().nodeAddrs)
  {
    if (kv.second == addr)
    {
      return kv.first;
    }
  }
  return UINT32_MAX;
}

inline Ipv4Address QTableRouting::SelectSource(uint32_t oif) const
{
  if (!m_ipv4)
  {
    return m_selfAddr;
  }
  if (oif < m_ipv4->GetNInterfaces() && m_ipv4->GetNAddresses(oif) > 0)
  {
    return m_ipv4->GetAddress(oif, 0).GetLocal();
  }
  return m_selfAddr;
}

inline bool QTableRouting::IsLocalAddress(Ipv4Address a) const
{
  if (!m_ipv4)
  {
    return false;
  }
  for (uint32_t i = 0; i < m_ipv4->GetNInterfaces(); ++i)
  {
    for (uint32_t j = 0; j < m_ipv4->GetNAddresses(i); ++j)
    {
      if (m_ipv4->GetAddress(i, j).GetLocal() == a)
      {
        return true;
      }
    }
  }
  return false;
}

inline void QTableRouting::RebuildFwdTable()
{
  m_fwd.clear();
  m_staticHops.clear();
  // Ensure SP next-hop buffer sized for all nodeIds; default invalid
  const uint32_t N_resize = NodeCount();
  if (m_spNextHop.size() != N_resize)
  {
    m_spNextHop.assign(N_resize, INVALID_NH);
  }
  else
  {
    std::fill(m_spNextHop.begin(), m_spNextHop.end(), INVALID_NH);
  }

  const uint32_t N = NodeCount();
  if (N == 0 || m_selfId == UINT32_MAX || m_selfId >= N)
  {
    return;
  }

  if (s_logRebuild)
  {
    std::cout << "[REBUILD] node=" << m_selfId
              << " linkUpEntries=" << S().linkUp.size()
              << " neighbors=" << m_nei.size() << std::endl;
  }

  std::unordered_map<uint32_t, std::vector<uint32_t>> adj;
  adj.reserve(S().linkUp.size());
  for (const auto& kv : S().linkUp)
  {
    if (!kv.second)
    {
      continue;
    }
    uint32_t a = static_cast<uint32_t>(kv.first >> 32);
    uint32_t b = static_cast<uint32_t>(kv.first & 0xffffffffu);
    adj[a].push_back(b);
    adj[b].push_back(a);
  }

  std::vector<int32_t> parent(N, -1);
  std::vector<uint8_t> seen(N, 0);
  std::queue<uint32_t> q;
  seen[m_selfId] = 1;
  q.push(m_selfId);

  while (!q.empty())
  {
    uint32_t u = q.front();
    q.pop();
    auto itAdj = adj.find(u);
    if (itAdj == adj.end())
    {
      continue;
    }
    for (uint32_t v : itAdj->second)
    {
      if (v >= N)
      {
        continue;
      }
      if (!seen[v])
      {
        seen[v] = 1;
        parent[v] = static_cast<int32_t>(u);
        q.push(v);
      }
    }
  }

  for (const auto& kv : S().nodeAddrs)
  {
    uint32_t dstId = kv.first;
    if (dstId == m_selfId)
    {
      continue;
    }
    if (dstId >= N || !seen[dstId])
    {
      continue;
    }

    uint32_t cur = dstId;
    uint32_t hopCount = 0;
    int32_t walker = static_cast<int32_t>(dstId);
    while (walker >= 0 && static_cast<uint32_t>(walker) != m_selfId)
    {
      int32_t parentId = parent[walker];
      if (parentId < 0)
      {
        hopCount = 0;
        break;
      }
      ++hopCount;
      walker = parentId;
    }
    if (hopCount > 0)
    {
      m_staticHops[dstId] = static_cast<uint16_t>(std::min<uint32_t>(hopCount, std::numeric_limits<uint16_t>::max()));
    }

    while (parent[cur] >= 0 && static_cast<uint32_t>(parent[cur]) != m_selfId)
    {
      cur = static_cast<uint32_t>(parent[cur]);
    }
    uint32_t nextHopId = (parent[dstId] < 0) ? dstId : cur;
    auto adjIt = m_nei.find(nextHopId);
    if (adjIt == m_nei.end())
    {
      continue;
    }

    // Record SP next-hop neighbor for this destination (by dst nodeId)
    if (dstId < m_spNextHop.size())
    {
      m_spNextHop[dstId] = static_cast<int32_t>(nextHopId);
    }

    FwdEntry fe;
    fe.nh = adjIt->second.neiIfAddr;
    fe.oif = adjIt->second.myOif;
    m_fwd[kv.second] = fe;
  }

  if (s_logRebuild)
  {
    std::cout << "[REBUILD] node=" << m_selfId
              << " fwdEntries=" << m_fwd.size() << std::endl;
  }
}

inline bool QTableRouting::ShouldEmitLog(bool countPacket)
{
  Shared& shared = S();
  if (countPacket)
  {
    ++shared.log.totalPackets;
  }

  const Time now = Simulator::Now();

  if (!shared.log.hasLogged)
  {
    shared.log.hasLogged = true;
    shared.log.lastLogSim = now;
    shared.log.lastPacketAtLog = shared.log.totalPackets;
    return true;
  }

  const bool packetReady = (shared.log.packetStride == 0) ||
                           (shared.log.totalPackets - shared.log.lastPacketAtLog >= shared.log.packetStride);
  const bool timeReady = shared.log.simStride.IsZero() ||
                         ((now - shared.log.lastLogSim) >= shared.log.simStride);

  if (packetReady && timeReady)
  {
    shared.log.lastPacketAtLog = shared.log.totalPackets;
    shared.log.lastLogSim = now;
    return true;
  }
  return false;
}

inline void QTableRouting::OnLinkFlip(uint32_t u, uint32_t v, bool up)
{
  uint32_t nei = UINT32_MAX;
  if (m_selfId == u)
  {
    nei = v;
  }
  else if (m_selfId == v)
  {
    nei = u;
  }

  if (nei == UINT32_MAX)
  {
    return;
  }

  for (auto& row : m_q)
  {
    double& cell = EnsureQValue(row.first, nei);
    if (!up)
    {
      cell = std::max(cell, m_maxCostMs);
    }
    else
    {
      cell = std::min(cell, m_defaultCostMs);
    }
  }

  auto it = m_nei.find(nei);
  if (it != m_nei.end())
  {
    if (!up)
    {
      it->second.ewmaCostMs = std::max(it->second.ewmaCostMs, m_maxCostMs);
    }
    else
    {
      it->second.ewmaCostMs = std::min(it->second.ewmaCostMs, m_defaultCostMs);
    }
  }

  KickRebuild();
}

inline void QTableRouting::SetRlNextHop(uint32_t dstId,
                                        uint32_t neiId,
                                        double ttlSeconds,
                                        const char* origin,
                                        uint64_t stamp)
{
  const char* src = origin ? origin : "router";
  double now = Simulator::Now().GetSeconds();
  if (neiId == std::numeric_limits<uint32_t>::max() || ttlSeconds <= 0.0)
  {
    RLDBG("[ROUTER][CLR] origin=%s node=%u dst=%u", src, m_selfId, dstId);
    m_rlOverride.erase(dstId);
    // NS_LOG_UNCOND("[ROUTER][OVR_MAP] size=" << m_rlOverride.size());
    return;
  }
  if (IsRlUseDisabled())
  {
    RLDBG("[ROUTER][SET] origin=%s node=%u dst=%u ignored (kill-switch)", src, m_selfId, dstId);
    return;
  }
  if (IsTeacherPinned(dstId))
  {
    RLDBG("[ROUTER][SET] origin=%s node=%u dst=%u ignored (teacher-pinned)", src, m_selfId, dstId);
    return;
  }
  RLDBG("[ROUTER][SET] origin=%s node=%u dst=%u nextHop=%u ttl=%.3f stamp=%llu",
        src,
        m_selfId,
        dstId,
        neiId,
        ttlSeconds,
        static_cast<unsigned long long>(stamp));
  RlOverrideEntry& entry = m_rlOverride[dstId];
  entry.nextHop = neiId;
  entry.expiry  = now + ttlSeconds;
  entry.stamp   = stamp;
  // NS_LOG_UNCOND("[ROUTER][OVR_MAP] size=" << m_rlOverride.size());
}

inline void QTableRouting::ClearRlNextHop(uint32_t dstId)
{
  RLDBG("[ROUTER][CLR] node=%u dst=%u", m_selfId, dstId);
  m_rlOverride.erase(dstId);
}

inline void QTableRouting::PinTeacher(uint32_t dstId, double sec)
{
  double now = Simulator::Now().GetSeconds();
  double hold = std::max(0.0, sec);
  if (hold <= 0.0)
  {
    m_teacherPin.erase(dstId);
    RLDBG("[ROUTER][PIN] node=%u dst=%u cleared", m_selfId, dstId);
    return;
  }
  double expiry = now + hold;
  m_teacherPin[dstId] = expiry;
  RLDBG("[ROUTER][PIN] node=%u dst=%u hold=%.3fs", m_selfId, dstId, hold);
}

inline bool QTableRouting::IsTeacherPinned(uint32_t dstId)
{
  auto it = m_teacherPin.find(dstId);
  if (it == m_teacherPin.end())
  {
    return false;
  }
  double now = Simulator::Now().GetSeconds();
  if (now > it->second)
  {
    m_teacherPin.erase(it);
    return false;
  }
  return true;
}

inline bool QTableRouting::GetRlNextHop(uint32_t dstId, uint32_t &neiOut)
{
  if (IsRlUseDisabled())
  {
    if (!m_rlOverride.empty())
    {
      RLDBG("[ROUTER][DISABLED] node=%u clearing %zu overrides", m_selfId, m_rlOverride.size());
      m_rlOverride.clear();
    }
    return false;
  }

  if (IsTeacherPinned(dstId))
  {
    auto pinnedIt = m_rlOverride.find(dstId);
    if (pinnedIt != m_rlOverride.end())
    {
      RLDBG("[ROUTER][PIN] node=%u dst=%u dropping RL override due to teacher pin",
            m_selfId,
            dstId);
      m_rlOverride.erase(pinnedIt);
    }
    return false;
  }

  auto it = m_rlOverride.find(dstId);
  if (it == m_rlOverride.end())
  {
    return false;
  }
  double now = Simulator::Now().GetSeconds();
  RlOverrideEntry& entry = it->second;
  if (now >= entry.expiry)
  {
    RLDBG("[ROUTER][TTL] expired override node=%u dst=%u", m_selfId, dstId);
    m_rlOverride.erase(it);
    return false;
  }
  uint64_t currentStamp = GetCurrentEnvStamp();
  if (entry.stamp != 0 && currentStamp != 0 && entry.stamp != currentStamp)
  {
    RLDBG("[ROUTER][STAMP] mismatch node=%u dst=%u entry=%llu current=%llu -> ignore",
          m_selfId,
          dstId,
          static_cast<unsigned long long>(entry.stamp),
          static_cast<unsigned long long>(currentStamp));
    m_rlOverride.erase(it);
    return false;
  }

  if (!IsLinkUp(m_selfId, entry.nextHop))
  {
    RLDBG("[ROUTER][WARN] invalid RL next hop node=%u dst=%u nh=%u", m_selfId, dstId, entry.nextHop);
    m_rlOverride.erase(it);
    return false;
  }

  neiOut = entry.nextHop;
  // RLDBG("[ROUTER][USE] node=%u dst=%u rlNextHop=%u stamp=%llu expiresIn=%.3f",
  //       m_selfId,
  //       dstId,
  //       neiOut,
  //       static_cast<unsigned long long>(entry.stamp),
  //       entry.expiry - now);
  return true;
}

inline void QTableRouting::SetRlUseDisabled(bool on)
{
  s_rlUseDisabled.store(on, std::memory_order_relaxed);
}

inline bool QTableRouting::IsRlUseDisabled()
{
  return s_rlUseDisabled.load(std::memory_order_relaxed);
}

inline void QTableRouting::SetCurrentEnvStamp(uint64_t stamp)
{
  s_currentEnvStamp.store(stamp, std::memory_order_relaxed);
}

inline uint64_t QTableRouting::GetCurrentEnvStamp()
{
  return s_currentEnvStamp.load(std::memory_order_relaxed);
}

inline void QTableRouting::MaybeLogDecision(uint32_t dst, uint32_t chosen, bool exploring, const std::vector<std::pair<uint32_t,double>>& candidates)
{
  ++m_decisionCount;
  auto prev = m_lastChoice.find(dst);
  bool changed = (prev == m_lastChoice.end()) || (prev->second != chosen);
  if (!exploring && !changed)
  {
    return;
  }

  m_lastChoice[dst] = chosen;

  if (!ShouldEmitLog(true))
  {
    return;
  }

  std::ostringstream oss;
  oss << "node=" << m_selfId << " dst=" << dst << " choose=" << chosen
      << (exploring ? " explore" : " exploit") << " q{";
  for (const auto& c : candidates)
  {
    oss << c.first << ":" << std::fixed << std::setprecision(2) << c.second << " ";
  }
  oss << "}";
  // Suppress verbose decision logs by default
  // NS_LOG_INFO(oss.str());
}

inline void QTableRouting::MaybeLogUpdate(uint32_t dst, uint32_t nei, double oldVal, double newVal, double costMs, double downstreamBestMs, const char* reason, uint8_t flags)
{
  double delta = std::fabs(newVal - oldVal);
  if (delta < m_updateLogThreshold)
  {
    return;
  }
  if (!ShouldEmitLog(false))
  {
    return;
  }
  std::ostringstream oss;
  oss << "node=" << m_selfId << " dst=" << dst << " nei=" << nei
      << " reason=" << reason
      << " costMs=" << costMs
      << " downMs=" << downstreamBestMs
      << " old=" << oldVal
      << " new=" << newVal
      << " flags=0x" << std::hex << unsigned(flags);
  // Suppress verbose update logs by default
  // NS_LOG_INFO(oss.str());
}

inline void QTableRouting::ApplyCostUpdate(uint32_t dst,
                                           uint32_t nei,
                                           double costMs,
                                           double downstreamBestMs,
                                           const char* reason,
                                           uint8_t flags,
                                           bool forceUpdate)
{
  if (!IsDestinationTracked(dst))
  {
    return;
  }
  if (!forceUpdate)
  {
    if (++m_pktsSinceUpdate % m_updateStride != 0)
    {
      return;
    }
  }
  else
  {
    ++m_pktsSinceUpdate;
  }
  double boundedCost = std::clamp(costMs, 0.0, m_maxCostMs);
  double boundedDown = std::clamp(downstreamBestMs, 0.0, m_maxCostMs);
  double& q = EnsureQValue(dst, nei);
  double old = q;
  double target = boundedCost + m_gamma * boundedDown;
  target = std::clamp(target, 0.0, m_maxCostMs);
  q = (1.0 - m_alpha) * q + m_alpha * target;
  q = std::clamp(q, 0.0, m_maxCostMs);
  MaybeLogUpdate(dst, nei, old, q, boundedCost, boundedDown, reason, flags);
}

inline void QTableRouting::SendFeedback(uint32_t prevNodeId, uint32_t destId, double costMs, double downstreamBestMs, uint8_t flags)
{
  if (!IsDestinationTracked(destId) || !m_ctrlSock)
  {
    return;
  }
  auto it = S().nodeAddrs.find(prevNodeId);
  if (it == S().nodeAddrs.end())
  {
    return;
  }
  uint8_t buf[1 + 4 + 4 + sizeof(double) * 2 + 1];
  buf[0] = CTRL_FEEDBACK;
  buf[1] = (destId >> 24) & 0xFF;
  buf[2] = (destId >> 16) & 0xFF;
  buf[3] = (destId >> 8) & 0xFF;
  buf[4] = destId & 0xFF;
  uint32_t senderId = m_selfId;
  buf[5] = (senderId >> 24) & 0xFF;
  buf[6] = (senderId >> 16) & 0xFF;
  buf[7] = (senderId >> 8) & 0xFF;
  buf[8] = senderId & 0xFF;
  std::memcpy(&buf[9], &costMs, sizeof(double));
  std::memcpy(&buf[9 + sizeof(double)], &downstreamBestMs, sizeof(double));
  buf[9 + sizeof(double) * 2] = flags;
  const uint32_t len = sizeof(buf);
  Ptr<Packet> pkt = Create<Packet>(buf, len);
  ++m_ctrlPktsTx;
  m_ctrlPayloadTx += len;
  m_ctrlWireTx    += (len + kL3L4);
  ++m_fbPktsTx;
  m_fbPayloadTx += len;
  m_fbWireTx    += (len + kL3L4);
  g_qrFbTxPkts.fetch_add(uint64_t{1}, std::memory_order_relaxed);
  g_qrFbTxBytes.fetch_add(static_cast<uint64_t>(len), std::memory_order_relaxed);
  InetSocketAddress remote(it->second, m_ctrlPort);
  m_ctrlSock->SendTo(pkt, 0, remote);
}

inline void QTableRouting::HandleCtrlRx(Ptr<Socket> sock)
{
  Address from;
  while (Ptr<Packet> pkt = sock->RecvFrom(from))
  {
    uint32_t len = pkt->GetSize();
    if (len == 0)
    {
      continue;
    }
    std::vector<uint8_t> buf(len);
    pkt->CopyData(buf.data(), len);
    uint8_t type = buf[0];
    ++m_ctrlPktsRx;
    m_ctrlPayloadRx += len;
    m_ctrlWireRx    += (len + kL3L4);
    switch (type)
    {
      case CTRL_FEEDBACK:
        ++m_fbPktsRx;
        m_fbPayloadRx += len;
        m_fbWireRx    += (len + kL3L4);
        ProcessFeedback(buf.data(), len);
        break;
      case CTRL_PROBE:
        ++m_probePktsRx;
        m_probePayloadRx += len;
        m_probeWireRx    += (len + kL3L4);
        ProcessProbe(buf.data(), len, from);
        break;
      case CTRL_PROBE_ACK:
        ++m_packPktsRx;
        m_packPayloadRx += len;
        m_packWireRx    += (len + kL3L4);
        ProcessProbeAck(buf.data(), len);
        break;
      default:
        break;
    }
  }
}

inline void QTableRouting::ProcessFeedback(const uint8_t* buf, uint32_t len)
{
  const uint32_t expected = 1 + 4 + 4 + sizeof(double) * 2 + 1;
  if (len < expected)
  {
    return;
  }
  uint32_t destId = (uint32_t(buf[1]) << 24) | (uint32_t(buf[2]) << 16) | (uint32_t(buf[3]) << 8) | uint32_t(buf[4]);
  uint32_t senderId = (uint32_t(buf[5]) << 24) | (uint32_t(buf[6]) << 16) | (uint32_t(buf[7]) << 8) | uint32_t(buf[8]);
  double costMs = 0.0;
  double downstreamBestMs = 0.0;
  std::memcpy(&costMs, &buf[9], sizeof(double));
  std::memcpy(&downstreamBestMs, &buf[9 + sizeof(double)], sizeof(double));
  uint8_t flags = buf[9 + sizeof(double) * 2];
  if (!IsDestinationTracked(destId))
  {
    return;
  }
  ApplyCostUpdate(destId, senderId, costMs, downstreamBestMs, "feedback", flags, false);
}

inline void QTableRouting::ProcessProbe(const uint8_t* buf, uint32_t len, const Address& from)
{
  if (len < 1 + 2 + 4)
  {
    return;
  }
  uint16_t seq = (uint16_t(buf[1]) << 8) | uint16_t(buf[2]);
  uint32_t requesterId = (uint32_t(buf[3]) << 24) | (uint32_t(buf[4]) << 16) | (uint32_t(buf[5]) << 8) | uint32_t(buf[6]);
  SendProbeAck(seq, requesterId, from);
  ++m_probesRx;
}

inline void QTableRouting::SendProbeAck(uint32_t seq, uint32_t requesterId, const Address& from)
{
  if (!m_ctrlSock)
  {
    return;
  }
  uint8_t buf[1 + 2 + 4];
  buf[0] = CTRL_PROBE_ACK;
  buf[1] = (seq >> 8) & 0xFF;
  buf[2] = seq & 0xFF;
  buf[3] = (m_selfId >> 24) & 0xFF;
  buf[4] = (m_selfId >> 16) & 0xFF;
  buf[5] = (m_selfId >> 8) & 0xFF;
  buf[6] = m_selfId & 0xFF;
  const uint32_t len = sizeof(buf);
  Ptr<Packet> pkt = Create<Packet>(buf, len);
  ++m_ctrlPktsTx;
  m_ctrlPayloadTx += len;
  m_ctrlWireTx    += (len + kL3L4);
  ++m_packPktsTx;
  m_packPayloadTx += len;
  m_packWireTx    += (len + kL3L4);
  g_qrPackTxPkts.fetch_add(uint64_t{1}, std::memory_order_relaxed);
  g_qrPackTxBytes.fetch_add(static_cast<uint64_t>(len), std::memory_order_relaxed);
  InetSocketAddress dst = InetSocketAddress::ConvertFrom(from);
  m_ctrlSock->SendTo(pkt, 0, dst);
}

inline void QTableRouting::ProcessProbeAck(const uint8_t* buf, uint32_t len)
{
  if (len < 1 + 2 + 4)
  {
    return;
  }
  uint16_t seq = (uint16_t(buf[1]) << 8) | uint16_t(buf[2]);
  auto it = m_pendingProbes.find(seq);
  if (it == m_pendingProbes.end())
  {
    return;
  }
  double rttMs = (Simulator::Now() - it->second.sent).GetMilliSeconds();
  auto adjIt = m_nei.find(it->second.neighbor);
  if (adjIt != m_nei.end())
  {
    adjIt->second.ewmaCostMs = (1.0 - s_probeEwmaW) * adjIt->second.ewmaCostMs + s_probeEwmaW * std::max(1.0, rttMs);
  }
  m_pendingProbes.erase(it);
}

inline void QTableRouting::ProbeTick()
{
  if (!m_started || m_nei.empty())
  {
    return;
  }

  const Time now = Simulator::Now();
  const Time staleInterval = m_probeInterval.IsZero() ? Seconds(1.0) : 2 * m_probeInterval;
  std::vector<uint16_t> expired;
  for (auto& kv : m_pendingProbes)
  {
    if (now - kv.second.sent >= staleInterval)
    {
      auto adjIt = m_nei.find(kv.second.neighbor);
      if (adjIt != m_nei.end())
      {
        adjIt->second.ewmaCostMs = std::min(adjIt->second.ewmaCostMs * 1.2, m_maxCostMs);
      }
      for (auto& row : m_q)
      {
        auto jt = row.second.find(kv.second.neighbor);
        if (jt != row.second.end())
        {
          jt->second = std::min(jt->second + 0.5 * m_penaltyDrop.GetMilliSeconds(), m_maxCostMs);
        }
      }
      expired.push_back(kv.first);
    }
  }
  for (uint16_t seq : expired)
  {
    m_pendingProbes.erase(seq);
  }

  std::vector<uint32_t> candidates;
  candidates.reserve(m_nei.size());
  for (const auto& kv : m_nei)
  {
    if (IsLinkUp(m_selfId, kv.first))
    {
      candidates.push_back(kv.first);
    }
  }

  if (candidates.empty())
  {
    m_probeEv = Simulator::Schedule(m_probeInterval, &QTableRouting::ProbeTick, this);
    return;
  }

  uint32_t probesToSend = std::min<uint32_t>(m_probeFanout, candidates.size());
  for (uint32_t n = 0; n < probesToSend; ++n)
  {
    uint32_t idx = (candidates.size() == 1) ? 0 : static_cast<uint32_t>(m_rngIdx->GetInteger(0, candidates.size() - 1));
    uint32_t neiId = candidates[idx];
    auto adjIt = m_nei.find(neiId);
    if (adjIt != m_nei.end())
    {
      SendProbe(neiId, adjIt->second);
    }
  }

  m_probeEv = Simulator::Schedule(m_probeInterval, &QTableRouting::ProbeTick, this);
}

inline void QTableRouting::SendProbe(uint32_t neiId, const Adj& adj)
{
  if (!m_ctrlSock)
  {
    return;
  }
  uint16_t seq = m_nextProbeSeq++;
  if (m_nextProbeSeq == 0)
  {
    m_nextProbeSeq = 1;
  }
  uint8_t buf[1 + 2 + 4];
  buf[0] = CTRL_PROBE;
  buf[1] = (seq >> 8) & 0xFF;
  buf[2] = seq & 0xFF;
  buf[3] = (m_selfId >> 24) & 0xFF;
  buf[4] = (m_selfId >> 16) & 0xFF;
  buf[5] = (m_selfId >> 8) & 0xFF;
  buf[6] = m_selfId & 0xFF;
  const uint32_t len = sizeof(buf);
  Ptr<Packet> pkt = Create<Packet>(buf, len);
  ++m_ctrlPktsTx;
  m_ctrlPayloadTx += len;
  m_ctrlWireTx    += (len + kL3L4);
  ++m_probePktsTx;
  m_probePayloadTx += len;
  m_probeWireTx    += (len + kL3L4);
  g_qrProbeTxPkts.fetch_add(uint64_t{1}, std::memory_order_relaxed);
  g_qrProbeTxBytes.fetch_add(static_cast<uint64_t>(len), std::memory_order_relaxed);
  InetSocketAddress remote(adj.neiIfAddr, m_ctrlPort);
  m_ctrlSock->SendTo(pkt, 0, remote);
  m_pendingProbes[seq] = PendingProbe{Simulator::Now(), neiId};
  ++m_probesTx;
}

inline void QTableRouting::NotifyDrop(const Ipv4Header& header,
                                      Ptr<const Packet> packet,
                                      Ipv4L3Protocol::DropReason reason,
                                      Ptr<Ipv4> ipv4,
                                      uint32_t)
{
  if (ipv4 != m_ipv4)
  {
    return;
  }
  const uint32_t idx = static_cast<uint32_t>(reason);
  if (idx < m_dropByReason.size())
  {
    ++m_dropByReason[idx];
  }
  QTag tag;
  if (!packet->PeekPacketTag(tag))
  {
    return;
  }
  if (tag.prevNodeId != m_selfId || tag.lastHopId == UINT32_MAX)
  {
    return;
  }
  uint32_t dstNodeId = tag.dstNodeId;
  if (!IsDestinationTracked(dstNodeId))
  {
    return;
  }
  double penalty = m_penaltyDrop.IsZero() ? m_maxCostMs : m_penaltyDrop.GetMilliSeconds();
  ApplyCostUpdate(dstNodeId, tag.lastHopId, penalty, m_defaultCostMs, "drop", 0x02, true);
}

inline void QTableRouting::InstallRlPicker(RlPickFn cb)
{
  s_rlPick = std::move(cb);
}

inline bool QTableRouting::RlEnabled()
{
  return s_rlEnabled.load(std::memory_order_relaxed);
}

inline void QTableRouting::SetRlEnabled(bool on)
{
  s_rlEnabled.store(on, std::memory_order_relaxed);
}

inline void QTableRouting::SetNextHopOverride(uint32_t nodeId, uint32_t dstId, int nextHop)
{
  std::lock_guard<std::mutex> lk(s_forcedMu);
  if (nextHop >= 0)
  {
    s_forced[RlKey(nodeId, dstId)] = nextHop;
  }
  else
  {
    s_forced.erase(RlKey(nodeId, dstId));
  }
}

inline std::atomic<bool> QTableRouting::s_rlEnabled{false};
inline std::atomic<bool> QTableRouting::s_rlUseDisabled{false};
inline std::atomic<uint64_t> QTableRouting::s_currentEnvStamp{0};
inline QTableRouting::RlPickFn QTableRouting::s_rlPick = nullptr;
inline std::unordered_map<uint64_t, int> QTableRouting::s_forced;
inline std::mutex QTableRouting::s_forcedMu;

} // namespace ns3
