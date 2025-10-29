// scratch/leo_rl_gym_env_impl.tcc

#include "leo_rl_gym_env.h"
#include "qtable-routing-range.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include <cstdint>
#include <string>
#include "ns3/log.h"
#include "ns3/simulator.h"

#if RL_DIAG
namespace {

std::string ToBitString(const std::vector<uint8_t>& vec)
{
  std::string out;
  out.reserve(vec.size());
  for (uint8_t v : vec)
  {
    out.push_back(v ? '1' : '0');
  }
  return out;
}

} // anonymous namespace
#endif

extern uint16_t g_openGymPort;

struct FlowSummary;
extern std::vector<FlowSummary> g_flowSummaries;

namespace ns3 {

extern bool g_logRl;
extern double g_rlStartS;

bool g_zeroRewardBeforeMeasure = false;

NS_OBJECT_ENSURE_REGISTERED(LeoRoutingGymEnv);
// Agent-attach flag used by the optional startup barrier
static std::atomic<bool> s_agentAttached{false};
bool LeoRoutingGymEnv::AgentAttached() { return s_agentAttached.load(); }
void LeoRoutingGymEnv::MarkAgentAttached() { s_agentAttached.store(true); }
TypeId LeoRoutingGymEnv::GetTypeId(void)
{
  static TypeId tid = TypeId("ns3::LeoRoutingGymEnv")
    .SetParent<OpenGymEnv>()
    .SetGroupName("OpenGym")
    .AddConstructor<LeoRoutingGymEnv>();
  return tid;
}

LeoRoutingGymEnv::LeoRoutingGymEnv() {}

LeoRoutingGymEnv::LeoRoutingGymEnv(Time stepTime,
                   uint32_t K,
                   Ptr<QTableRouting> router,
                   Ptr<RangeManager> rangeMgr,
                   int32_t trainDst,
                   double hopPenalty,
                   double qPenalty,
                   double dropPenalty,
                   double queueCapacityPkts,
                   double measureStart,
                   double simTime,
                   const std::vector<bool>* dstMask)
  : m_stepTime(stepTime),
    m_K(K),
    m_router(router),
    m_range(rangeMgr),
    m_lambdaHop(hopPenalty),
    m_lambdaQ(qPenalty),
    m_betaDrop(dropPenalty),
    m_queueCapacityPkts(std::max(1.0, queueCapacityPkts)),
    m_measureStart(measureStart),
    m_simTime(std::max(1e-6, simTime)),
    m_trainDst(trainDst),
    m_dstMask(dstMask)
{
  InitDestinationCycle();
}

LeoRoutingGymEnv::LeoRoutingGymEnv(Time stepTime,
                   uint32_t K,
                   const std::vector< Ptr<QTableRouting> >& routers,
                   Ptr<RangeManager> rangeMgr,
                   int32_t trainDst,
                   double hopPenalty,
                   double qPenalty,
                   double dropPenalty,
                   double queueCapacityPkts,
                   double measureStart,
                   double simTime,
                   const std::vector<bool>* dstMask)
  : m_stepTime(stepTime),
    m_K(K),
    m_router(nullptr),
    m_routers(routers),
    m_range(rangeMgr),
    m_lambdaHop(hopPenalty),
    m_lambdaQ(qPenalty),
    m_betaDrop(dropPenalty),
    m_queueCapacityPkts(std::max(1.0, queueCapacityPkts)),
    m_measureStart(measureStart),
    m_simTime(std::max(1e-6, simTime)),
    m_trainDst(trainDst),
    m_dstMask(dstMask)
{
  InitDestinationCycle();
}

Ptr<OpenGymSpace> LeoRoutingGymEnv::GetObservationSpace()
{
  // Mark agent attached on first space query to signal handshake
  s_agentAttached.store(true);
  const uint32_t F = 8;
  std::vector<uint32_t> obsShape{F};
  std::vector<uint32_t> maskShape{m_K};
  std::vector<uint32_t> neiShape{m_K * kNeighborFeatureDim};

  Ptr<OpenGymBoxSpace> obsSpace = CreateObject<OpenGymBoxSpace>(-1.0f, 1.0f, obsShape, "float");
  Ptr<OpenGymBoxSpace> maskSpace = CreateObject<OpenGymBoxSpace>(0.0f, 1.0f, maskShape, "uint8_t");
  Ptr<OpenGymBoxSpace> neiSpace = CreateObject<OpenGymBoxSpace>(0.0f, 1.0f, neiShape, "float");
  Ptr<OpenGymBoxSpace> queueSpace = CreateObject<OpenGymBoxSpace>(0.0f, 1.0f, maskShape, "float");
  float teacherMax = static_cast<float>(m_K > 0 ? static_cast<int32_t>(m_K) : 0);
  Ptr<OpenGymBoxSpace> teacherActionSpace = CreateObject<OpenGymBoxSpace>(-1.0f, teacherMax,
      std::vector<uint32_t>{1}, "int32_t");
  Ptr<OpenGymBoxSpace> teacherIdxSpace = CreateObject<OpenGymBoxSpace>(-1.0f, teacherMax,
      std::vector<uint32_t>{1}, "int32_t");
  Ptr<OpenGymBoxSpace> neighborIdSpace = CreateObject<OpenGymBoxSpace>(-1.0f, 1.0e6f,
      maskShape, "int32_t");
  Ptr<OpenGymBoxSpace> teacherNeighborSpace = CreateObject<OpenGymBoxSpace>(0.0f, 1.0e6f,
      std::vector<uint32_t>{1}, "int32_t");
  Ptr<OpenGymBoxSpace> teacherCostSpace = CreateObject<OpenGymBoxSpace>(0.0f, 5000.0f,
      std::vector<uint32_t>{1}, "float");
  Ptr<OpenGymDictSpace> dict = CreateObject<OpenGymDictSpace>();
  dict->Add("obs", obsSpace);
  dict->Add("mask", maskSpace);
  dict->Add("nei_feat", neiSpace);
  dict->Add("queue_norm", queueSpace);
  dict->Add("neighbor_ids", neighborIdSpace);
  dict->Add("teacher_action", teacherActionSpace);
  dict->Add("teacher_idx", teacherIdxSpace);
  dict->Add("teacher_neighbor", teacherNeighborSpace);
  dict->Add("teacher_cost", teacherCostSpace);
  return dict;
}

Ptr<OpenGymSpace> LeoRoutingGymEnv::GetActionSpace()
{
  // Discrete probe: expose a discrete action space for quick transport debugging.
  // Client can still send Box payloads; ExecuteActions handles both.
  uint32_t n = (m_K > 0) ? (m_K + 1) : 1; // 0..K (K can be mapped to -1 by client if desired)
  // NS_LOG_UNCOND("[ENV][SPACE] action=Discrete(K+1) K=" << m_K);
  return CreateObject<OpenGymDiscreteSpace>(static_cast<int>(n));
}

void LeoRoutingGymEnv::ScheduleNextStateRead()
{
  if (m_ev.IsPending())
  {
    m_ev.Cancel();
  }
  m_ev = Simulator::Schedule(m_stepTime, &LeoRoutingGymEnv::Step, this);
}

void LeoRoutingGymEnv::Step()
{
  uint32_t numNodes = static_cast<uint32_t>(m_routers.size());
  if (numNodes == 0)
  {
    numNodes = m_router ? 1u : 0u;
  }
  if (numNodes == 0)
  {
    numNodes = 1;
  }

  if (m_focusNode >= 0 && static_cast<uint32_t>(m_focusNode) < numNodes)
  {
    m_curNode = static_cast<uint32_t>(m_focusNode);
  }
  else
  {
    m_curNode = (m_curNode + 1) % numNodes;
  }

  if (m_trainDst < 0)
  {
    if (m_dstCycle.empty())
    {
      InitDestinationCycle();
    }
    if (!m_dstCycle.empty())
    {
      m_currentDst = m_dstCycle[m_dstCyclePos];
      m_dstCyclePos = (m_dstCyclePos + 1) % m_dstCycle.size();
    }
  }

  OpenGymEnv::Notify();
  ScheduleNextStateRead();
}

Ptr<OpenGymDataContainer> LeoRoutingGymEnv::GetObservation()
{
  // Reset per-step override counter at the beginning of a new observation window
  m_stepOverrides = 0;

  m_lastUpNeis = GetUpNeighbors(m_curNode);
  if (m_lastUpNeis.size() > m_K)
  {
    m_lastUpNeis.resize(m_K);
  }

  const uint32_t featureLen = 8;
  std::vector<float> obs;
  obs.reserve(featureLen);

  const double nowSec = Simulator::Now().GetSeconds();
  const double simRatio = (m_simTime > 0.0) ? std::clamp(nowSec / m_simTime, 0.0, 1.0) : 0.0;
  const double queuePkts = std::max(0.0, GetQueuePkts(m_curNode));
  const double queueRatio = (m_queueCapacityPkts > 0.0)
                                ? std::clamp(queuePkts / m_queueCapacityPkts, 0.0, 1.0)
                                : 0.0;
  const uint32_t dstId = m_currentDst;
  const float dstNorm = static_cast<float>((dstId % 256u) / 127.5f - 1.0f);

  obs.push_back(static_cast<float>(simRatio * 2.0 - 1.0));
  obs.push_back(static_cast<float>(queueRatio * 2.0 - 1.0));
  obs.push_back(dstNorm);

  const float degreeNorm = (m_K > 0)
                               ? static_cast<float>(std::clamp(static_cast<double>(m_lastUpNeis.size()) /
                                                              static_cast<double>(m_K),
                                                              0.0,
                                                              1.0) * 2.0 - 1.0)
                               : -1.0f;
  obs.push_back(degreeNorm);

  const bool dstIsGs = (m_dstMask && dstId < m_dstMask->size() && (*m_dstMask)[dstId]);
  obs.push_back(dstIsGs ? 1.0f : -1.0f);

  const double ewma = (m_lastUpNeis.empty())
                          ? 0.0
                          : std::clamp(GetEwmaDelayToDstMs(m_curNode, m_lastUpNeis.front(), dstId) / 200.0,
                                       0.0,
                                       1.0);
  obs.push_back(static_cast<float>(ewma * 2.0 - 1.0));

  const double vis = (m_lastUpNeis.empty())
                         ? 0.0
                         : std::clamp(GetVisRemaining(m_curNode, m_lastUpNeis.front()) / 10.0,
                                      0.0,
                                      1.0);
  obs.push_back(static_cast<float>(vis * 2.0 - 1.0));

  while (obs.size() < featureLen)
  {
    obs.push_back(0.0f);
  }

  std::vector<uint8_t> mask(m_K, 0u);
  for (size_t idx = 0; idx < m_lastUpNeis.size() && idx < mask.size(); ++idx)
  {
    mask[idx] = 1u;
  }

  auto clampUnit = [](double value) {
    if (!std::isfinite(value))
    {
      return 0.0;
    }
    return std::clamp(value, 0.0, 1.0);
  };

  std::vector<float> neiFeat(m_K * kNeighborFeatureDim, 0.0f);
  std::vector<float> queueNorms(m_K, 0.0f);
  const double queueDenom = std::max(1e-6, m_queueCapacityPkts);
  for (size_t idx = 0; idx < m_lastUpNeis.size() && idx < m_K; ++idx)
  {
    const uint32_t nei = m_lastUpNeis[idx];
    const double ewmaNorm = clampUnit(GetEwmaDelayToDstMs(m_curNode, nei, dstId) / 200.0);
    const double qPkts = std::max(0.0, GetQueuePkts(nei));
    const double qFrac = clampUnit(qPkts / queueDenom);
    const double visNorm = clampUnit(GetVisRemaining(m_curNode, nei) / 10.0);
    const double hopsNorm = clampUnit(static_cast<double>(GetSpfHopsVia(m_curNode, nei, dstId)) / 10.0);

    const size_t base = idx * kNeighborFeatureDim;
    neiFeat[base + 0] = static_cast<float>(ewmaNorm);
    neiFeat[base + 1] = static_cast<float>(qFrac);
    neiFeat[base + 2] = static_cast<float>(visNorm);
    neiFeat[base + 3] = static_cast<float>(hopsNorm);
    neiFeat[base + 4] = 1.0f;
    queueNorms[idx] = static_cast<float>(qFrac);
  }

  m_lastMask.assign(mask.begin(), mask.end());
  if (m_teacherPolicy == "qtable")
  {
    m_lastTeacherIdx = FindTeacherIndex(m_curNode, m_lastUpNeis);
  }
  else if (m_teacherPolicy == "ospf")
  {
    m_lastTeacherIdx = FindTeacherIndexOspf(m_curNode, m_lastUpNeis);
  }
  else
  {
    m_lastTeacherIdx = -1;
  }

  m_lastTeacherNeighbor = std::numeric_limits<uint32_t>::max();
  m_lastTeacherCost = std::numeric_limits<double>::quiet_NaN();
  if (m_lastTeacherIdx >= 0 && static_cast<size_t>(m_lastTeacherIdx) < m_lastUpNeis.size())
  {
    m_lastTeacherNeighbor = m_lastUpNeis[static_cast<size_t>(m_lastTeacherIdx)];
    Ptr<QTableRouting> router = (m_routers.size() > m_curNode && m_routers[m_curNode]) ? m_routers[m_curNode] : m_router;
    if (router && router->IsDestinationTracked(dstId))
    {
      m_lastTeacherCost = router->GetQValue(dstId, m_lastTeacherNeighbor);
    }
  }
  m_lastQueueNorm = queueNorms;

  RlDiagStep obsSnap;
  obsSnap.nodeId = m_curNode;
  obsSnap.dstId = m_currentDst;
  obsSnap.upNeiNodeIds = m_lastUpNeis;
  obsSnap.mask.assign(mask.begin(), mask.end());
  obsSnap.teacherIdx = m_lastTeacherIdx;
  obsSnap.teacherNodeId = m_lastTeacherNeighbor;
  m_lastDiag[Key(m_curNode, m_currentDst)] = obsSnap;

  // RLDBG("[ENV][OBS] node=%u dst=%u K=%zu mask=%s teacherId=%d teacherIdx=%d",
  //       obsSnap.nodeId,
  //       obsSnap.dstId,
  //       obsSnap.upNeiNodeIds.size(),
  //       ToBitString(obsSnap.mask).c_str(),
  //       (obsSnap.teacherNodeId == std::numeric_limits<uint32_t>::max() ? -1 : static_cast<int32_t>(obsSnap.teacherNodeId)),
  //       obsSnap.teacherIdx);

  if (m_lastTeacherIdx < 0)
  {
    ++m_cntTeacherMissing;
    // RLDBG("[ENV][TEACHER_MISS] node=%u dst=%u teacherGlobal=%d", m_curNode, m_currentDst,
    //       (m_lastTeacherNeighbor == std::numeric_limits<uint32_t>::max() ? -1 : static_cast<int32_t>(m_lastTeacherNeighbor)));
  }

  Ptr<OpenGymDictContainer> dict = CreateObject<OpenGymDictContainer>();
  Ptr<OpenGymBoxContainer<float>> obsBox = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>{featureLen});
  for (float v : obs)
  {
    obsBox->AddValue(v);
  }
  Ptr<OpenGymBoxContainer<uint8_t>> maskBox = CreateObject<OpenGymBoxContainer<uint8_t>>(std::vector<uint32_t>{m_K});
  for (uint8_t m : mask)
  {
    maskBox->AddValue(m);
  }
  Ptr<OpenGymBoxContainer<float>> neiBox = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>{m_K * kNeighborFeatureDim});
  for (float v : neiFeat)
  {
    neiBox->AddValue(v);
  }
  Ptr<OpenGymBoxContainer<float>> queueBox = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>{m_K});
  for (float q : queueNorms)
  {
    queueBox->AddValue(q);
  }
  Ptr<OpenGymBoxContainer<int32_t>> neighborBox = CreateObject<OpenGymBoxContainer<int32_t>>(std::vector<uint32_t>{m_K});
  for (size_t idx = 0; idx < m_K; ++idx)
  {
    int32_t val = (idx < m_lastUpNeis.size()) ? static_cast<int32_t>(m_lastUpNeis[idx]) : -1;
    neighborBox->AddValue(val);
  }
  Ptr<OpenGymBoxContainer<int32_t>> teacherActionBox = CreateObject<OpenGymBoxContainer<int32_t>>(std::vector<uint32_t>{1});
  teacherActionBox->AddValue(m_lastTeacherIdx);
  Ptr<OpenGymBoxContainer<int32_t>> teacherIdxBox = CreateObject<OpenGymBoxContainer<int32_t>>(std::vector<uint32_t>{1});
  teacherIdxBox->AddValue(m_lastTeacherIdx);
  Ptr<OpenGymBoxContainer<int32_t>> teacherNeighborBox = CreateObject<OpenGymBoxContainer<int32_t>>(std::vector<uint32_t>{1});
  int32_t teacherNeighborVal = (m_lastTeacherNeighbor == std::numeric_limits<uint32_t>::max())
                                   ? -1
                                   : static_cast<int32_t>(m_lastTeacherNeighbor);
  teacherNeighborBox->AddValue(teacherNeighborVal);
  Ptr<OpenGymBoxContainer<float>> teacherCostBox = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>{1});
  float teacherCostVal = std::isfinite(m_lastTeacherCost) ? static_cast<float>(m_lastTeacherCost) : 0.0f;
  teacherCostBox->AddValue(teacherCostVal);
  dict->Add("obs", obsBox);
  dict->Add("mask", maskBox);
  dict->Add("nei_feat", neiBox);
  dict->Add("queue_norm", queueBox);
  dict->Add("neighbor_ids", neighborBox);
  dict->Add("teacher_action", teacherActionBox);
  dict->Add("teacher_idx", teacherIdxBox);
  dict->Add("teacher_neighbor", teacherNeighborBox);
  dict->Add("teacher_cost", teacherCostBox);
  return dict;
}

float LeoRoutingGymEnv::GetReward()
{
  const double t = Simulator::Now().GetSeconds();
  // Gate rewards to measurement window
  if (t < m_measureStart)
  {
    return 0.0f;
  }

  // Placeholder reward shaping; these can be wired to real signals later
  const double avgDelay  = m_lastAvgDelay;   // seconds
  const double dropRate  = m_lastDropRate;   // 0..1
  const double meanQueue = m_lastMeanQueue;  // 0..1

  // Weights: bias toward PDR, keep delay reasonable, discourage override spam
  constexpr double wDelay = 1.0;
  constexpr double wDrop  = 8.0;
  constexpr double wQueue = 0.2;
  constexpr double wCtrl  = 5e-3;

  const double r = -(wDelay * avgDelay + wDrop * dropRate + wQueue * meanQueue)
                   - wCtrl * static_cast<double>(m_stepOverrides);
  return static_cast<float>(r);
}

bool LeoRoutingGymEnv::GetGameOver()
{
  return false;
}

bool LeoRoutingGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
  // High-signal entry log and null-action check
  // double now = Simulator::Now().GetSeconds();
  // NS_LOG_UNCOND("[ENV][ACT_ENTER] t=" << now);
  if (!action)
  {
    // NS_LOG_UNCOND("[ENV][ACT_NULL]");
    return false;
  }

  // Entry trace per step for timing/ordering sanity
  // std::cout << "[ENV][ENTER] step=" << (m_stepStamp + 1)
  //           << " t=" << Simulator::Now().GetSeconds() << "s\n";
  ++m_stepStamp;
  QTableRouting::SetCurrentEnvStamp(m_stepStamp);

  int32_t rawAction = -1;
  // Prefer Discrete first when action space is Discrete(K+1): map K -> -1 (teacher)
  if (auto disc = DynamicCast<OpenGymDiscreteContainer>(action))
  {
    uint32_t a = disc->GetValue();
    int idx = (a == static_cast<uint32_t>(m_K)) ? -1 : static_cast<int>(a);
    // NS_LOG_UNCOND("[ENV][ACT_DECODE] kind=disc val=" << a << " -> idx=" << idx);
    rawAction = static_cast<int32_t>(idx);
  }
  else if (auto box = DynamicCast<OpenGymBoxContainer<int32_t>>(action))
  {
    const std::vector<int32_t>& data = box->GetData();
    if (!data.empty())
    {
      rawAction = data.front();
    }
  }
  else if (auto boxf = DynamicCast<OpenGymBoxContainer<float>>(action))
  {
    const std::vector<float>& data = boxf->GetData();
    if (!data.empty() && std::isfinite(data.front()))
    {
      rawAction = static_cast<int32_t>(std::lrint(data.front()));
    }
  }
  else if (auto boxd = DynamicCast<OpenGymBoxContainer<double>>(action))
  {
    const std::vector<double>& data = boxd->GetData();
    if (!data.empty() && std::isfinite(data.front()))
    {
      rawAction = static_cast<int32_t>(std::llrint(data.front()));
    }
  }
  else
  {
    // RLDBG("[ENV][ACT] unsupported container type -> teacher");
    return false;
  }

  // Immediate visibility into what the agent sent over the wire.
  // RLDBG("[ENV][ACT] rawAction=%d", rawAction);
  // NS_LOG_UNCOND("[ENV][ACT] rawAction=" << rawAction);

  const uint32_t nodeId = m_curNode;
  const uint32_t dstId = ResolveDestination(nodeId);
  // Entry log for tracing decisions per-step (use members for clarity)
  // NS_LOG_UNCOND("[ENV][ACT_ENTER] step=" << m_stepStamp
  //                << " node=" << m_curNode << " dst=" << m_currentDst
  //                << " idx=" << rawAction << " K=" << m_K);
  // Keep the override alive long enough for the router to pick it.
  const double ttlSeconds = 1.0; // extended window for bring-up
  const uint64_t diagKey = Key(nodeId, dstId);

  auto snapIt = m_lastDiag.find(diagKey);
  if (snapIt == m_lastDiag.end())
  {
    // RLDBG("[ENV][ERR] no snapshot for node=%u dst=%u", nodeId, dstId);
    ++m_cntTeacherMissing;
    UseTeacherPath(nodeId, dstId);
    return true;
  }

  const RlDiagStep& snap = snapIt->second;
  // Pre-validation attempt trace (masked=-1 if OOB) — disabled
  if (snap.upNeiNodeIds.size() != snap.mask.size())
  {
    // NS_LOG_UNCOND("[ENV][WARN] mask-size mismatch neigh=" << snap.upNeiNodeIds.size()
    //                << " mask=" << snap.mask.size());
    // Do not early-return; continue and validate chosen index against both.
  }

  if (snap.teacherIdx < 0)
  {
    // RLDBG("[ENV][TEACHER_STATE] node=%u dst=%u teacher missing", nodeId, dstId);
  }

  if (rawAction < 0)
  {
    ++m_cntNegTeacher;
    // RLDBG("[ENV][ACT] node=%u dst=%u action=NEG teacherIdx=%d", nodeId, dstId, snap.teacherIdx);
    // Suppress auto-clearing RL and teacher pinning
    // Ptr<QTableRouting> router = ClearRlOverride(nodeId, dstId);
    // if (router) { router->PinTeacher(dstId, 0.0); }
    return true;
  }

  const size_t actIdx = static_cast<size_t>(rawAction);
  if (actIdx >= snap.upNeiNodeIds.size())
  {
    ++m_cntOobReject;
    // RLDBG("[ENV][REJ] node=%u dst=%u action=%d out-of-bounds (K=%zu)",
    //       nodeId, dstId, rawAction, snap.upNeiNodeIds.size());
    // NS_LOG_UNCOND("[ENV][REJ] reason=masked_or_oob node=" << m_curNode
    //                << " dst=" << m_currentDst
    //                << " idx=" << rawAction
    //                << " K=" << m_K);
    // Ptr<QTableRouting> router = ClearRlOverride(nodeId, dstId);
    // if (router) { router->PinTeacher(dstId, 0.0); }
    return true;
  }

  if (actIdx >= snap.mask.size() || !snap.mask[actIdx])
  {
    ++m_cntMaskedReject;
    // RLDBG("[ENV][REJ] node=%u dst=%u action=%d masked -> teacher", nodeId, dstId, rawAction);
    // NS_LOG_UNCOND("[ENV][REJ] reason=masked_or_oob node=" << m_curNode
    //                << " dst=" << m_currentDst
    //                << " idx=" << rawAction
    //                << " K=" << m_K);
    // Ptr<QTableRouting> router = ClearRlOverride(nodeId, dstId);
    // if (router) { router->PinTeacher(dstId, 0.0); }
    return true;
  }

  const uint32_t neiGlobalId = snap.upNeiNodeIds[actIdx];
  ++m_cntApplied;
  // RLDBG("[ENV][APPLY] node=%u dst=%u action=%d neighbourGlobal=%u ttl=%.3f",
  //       nodeId, dstId, rawAction, neiGlobalId, ttlSeconds);
  // Unconditional apply trace (helps correlate with router usage)
  // NS_LOG_UNCOND("[ENV][APPLY] node=" << m_curNode
  //                << " dst=" << m_currentDst
  //                << " idx=" << rawAction
  //                << " nei=" << neiGlobalId
  //                << " ttl=" << ttlSeconds);
  SetRlOverride(nodeId, dstId, neiGlobalId, ttlSeconds);
  // Count overrides applied during this Δt for control-cost in reward shaping
  ++m_stepOverrides;
  return true;
}

void LeoRoutingGymEnv::SetRlOverride(uint32_t nodeId, uint32_t dstId, uint32_t neiId, double ttlSec)
{
  Ptr<QTableRouting> router = GetRouterPtr(nodeId);
  if (!router)
  {
    // RLDBG("[ENV][WARN] missing router for node=%u dst=%u override=%u", nodeId, dstId, neiId);
    return;
  }
  // Bypass stamp-gating and give the router a bit more time to pick it.
  double hold = std::max(0.50, ttlSec);
  router->SetRlNextHop(dstId, neiId, hold, "env", /*stamp*/ 0);
  // RLDBG("[ENV][SET_RL v2] node=%u dst=%u nei=%u ttl=%.3f stamp=%llu",
  //        nodeId,
  //        dstId,
  //        neiId,
  //        hold,
  //        static_cast<unsigned long long>(m_stepStamp));
  // NS_LOG_UNCOND("[ENV][SET_RL v2] ttl>=0.50 stamp=0");
  if ((m_cntApplied % 50) == 0)
  {
    // NS_LOG_UNCOND("[ENV] overrides_applied=" << m_cntApplied);
  }
  // Mirror cache set unconditionally for easier grepping without extra flags.
  // RLDBG("[ENV][CACHE] set for %u,%u", nodeId, dstId);
  {
    std::lock_guard<std::mutex> lk(m_actionMu);
    m_actionMap[Key(nodeId, dstId)] = neiId;
    if (g_logRl)
    {
      std::cout << "[ENV][CACHE] set node=" << nodeId
                << " dst=" << dstId
                << " nei=" << neiId
                << " ttl=" << hold
                << "\n";
    }
  }
}

Ptr<QTableRouting> LeoRoutingGymEnv::ClearRlOverride(uint32_t nodeId, uint32_t dstId)
{
  Ptr<QTableRouting> router = GetRouterPtr(nodeId);
  if (!router)
  {
    RLDBG("[ENV][WARN] missing router for node=%u dst=%u clear", nodeId, dstId);
    return nullptr;
  }
  router->ClearRlNextHop(dstId);
  // RLDBG("[ENV][CLR_RL] node=%u dst=%u", nodeId, dstId);
  {
    std::lock_guard<std::mutex> lk(m_actionMu);
    m_actionMap.erase(Key(nodeId, dstId));
    if (g_logRl)
    {
      std::cout << "[ENV][CACHE] clear node=" << nodeId
                << " dst=" << dstId
                << "\n";
    }
  }
  return router;
}

void LeoRoutingGymEnv::UseTeacherPath(uint32_t nodeId, uint32_t dstId)
{
  ++m_cntTeacherApplied;
  // RLDBG("[ENV][TEACHER] node=%u dst=%u", nodeId, dstId);
  // Suppress auto-clearing RL and pinning teacher in UseTeacherPath
  // Ptr<QTableRouting> router = ClearRlOverride(nodeId, dstId);
  // if (router) { router->PinTeacher(dstId, 1.0); }
}

std::string LeoRoutingGymEnv::GetExtraInfo()
{
  std::ostringstream oss;
  oss << "{\"node\":" << m_curNode
      << ",\"dst\":" << ResolveDestination(m_curNode)
      << ",\"teacher_idx\":" << m_lastTeacherIdx
      << ",\"teacher_action\":" << m_lastTeacherIdx
      << ",\"teacher_neighbor\":" << ((m_lastTeacherNeighbor == std::numeric_limits<uint32_t>::max()) ? -1 : static_cast<int64_t>(m_lastTeacherNeighbor))
      << ",\"teacher_cost\":" << (std::isfinite(m_lastTeacherCost) ? m_lastTeacherCost : -1.0)
      << ",\"queue_teacher\":"
      << ((m_lastTeacherIdx >= 0 && static_cast<size_t>(m_lastTeacherIdx) < m_lastQueueNorm.size())
              ? static_cast<double>(m_lastQueueNorm[static_cast<size_t>(m_lastTeacherIdx)])
              : 0.0)
      << ",\"sim_time\":" << Simulator::Now().GetSeconds()
      << ",\"overrides_applied\":" << g_rlOverridePkts
      << "}";
  return oss.str();
}

void LeoRoutingGymEnv::DoDispose()
{
  // RLDBG("[ENV][COUNTS] applied=%llu negTeacher=%llu maskedReject=%llu oobReject=%llu teacherApplied=%llu teacherMissing=%llu",
  //       static_cast<unsigned long long>(m_cntApplied),
  //       static_cast<unsigned long long>(m_cntNegTeacher),
  //       static_cast<unsigned long long>(m_cntMaskedReject),
  //       static_cast<unsigned long long>(m_cntOobReject),
  //       static_cast<unsigned long long>(m_cntTeacherApplied),
  //       static_cast<unsigned long long>(m_cntTeacherMissing));
  if (m_ev.IsPending())
  {
    m_ev.Cancel();
  }
  if (m_heartbeatEv.IsPending())
  {
    m_heartbeatEv.Cancel();
  }
  OpenGymEnv::DoDispose();
}

void LeoRoutingGymEnv::Start()
{
  InitDestinationCycle();
  {
    std::lock_guard<std::mutex> lk(m_actionMu);
    m_actionMap.clear();
  }
  m_stepStamp = 0;
  m_prevEwmaMs.clear();
  m_lastMask.clear();
  m_lastTeacherIdx = -1;
  m_lastDiag.clear();
  m_cntNegTeacher = 0;
  m_cntMaskedReject = 0;
  m_cntOobReject = 0;
  m_cntApplied = 0;
  m_cntTeacherApplied = 0;
  m_cntTeacherMissing = 0;
  QTableRouting::InstallRlPicker([this](uint32_t node, uint32_t dst,
                                        const std::vector<uint32_t>& ups) -> int {
    if (!QTableRouting::RlEnabled())
    {
      return -1;
    }
    if (Simulator::Now().GetSeconds() < g_rlStartS)
    {
      return -1;
    }
    auto hasNeighbor = [&ups](uint32_t cand) -> bool {
      return std::find(ups.begin(), ups.end(), cand) != ups.end();
    };

    Ptr<QTableRouting> routerForNode = (m_routers.size() > node && m_routers[node]) ? m_routers[node] : m_router;
    if (routerForNode)
    {
      uint32_t overrideNh = 0;
      if (routerForNode->GetRlNextHop(dst, overrideNh) && hasNeighbor(overrideNh))
      {
        if (g_logRl)
        {
          std::cout << "[RL/PICK] node=" << node
                    << " dst=" << dst
                    << " -> ttl " << overrideNh
                    << "\n";
        }
        return static_cast<int>(overrideNh);
      }
    }

    std::lock_guard<std::mutex> lk(m_actionMu);

    auto it = m_actionMap.find(Key(node, dst));
    if (it != m_actionMap.end())
    {
      uint32_t cand = it->second;
      if (hasNeighbor(cand))
      {
        if (g_logRl)
        {
          std::cout << "[RL/PICK] node=" << node
                    << " dst=" << dst
                    << " -> exact " << cand
                    << "\n";
        }
        return static_cast<int>(cand);
      }
    }

    if (g_logRl)
    {
      std::cout << "[RL/PICK] node=" << node
                << " dst=" << dst
                << " miss (no cache or not up)\n";
    }
    return -1;
  });
  NS_LOG_UNCOND("[GYM] listening openGymPort=" << g_openGymPort);
  if (m_heartbeatEv.IsPending())
  {
    m_heartbeatEv.Cancel();
  }
  m_heartbeatEv = Simulator::Schedule(Seconds(5.0), &LeoRoutingGymEnv::LogHeartbeat, this);
  OpenGymEnv::Notify();
  ScheduleNextStateRead();
}

void LeoRoutingGymEnv::SetFocusNode(int32_t nodeId)
{
  m_focusNode = nodeId;
}

void LeoRoutingGymEnv::LogHeartbeat()
{
  NS_LOG_UNCOND("[GYM] heartbeat t=" << Simulator::Now().GetSeconds() << " s");
  m_heartbeatEv = Simulator::Schedule(Seconds(5.0), &LeoRoutingGymEnv::LogHeartbeat, this);
}

void LeoRoutingGymEnv::InitDestinationCycle()
{
  m_dstCycle.clear();
  if (m_trainDst >= 0)
  {
    m_currentDst = static_cast<uint32_t>(std::max<int32_t>(0, m_trainDst));
    m_dstCyclePos = 0;
    return;
  }

  uint32_t numNodes = static_cast<uint32_t>(m_routers.size());
  if (numNodes == 0)
  {
    numNodes = m_router ? 1u : 0u;
  }
  if (numNodes == 0)
  {
    numNodes = 1;
  }

  if (m_dstMask)
  {
    for (uint32_t idx = 0; idx < m_dstMask->size(); ++idx)
    {
      if ((*m_dstMask)[idx])
      {
        m_dstCycle.push_back(idx);
      }
    }
  }
  if (m_dstCycle.empty())
  {
    for (uint32_t idx = 0; idx < numNodes; ++idx)
    {
      m_dstCycle.push_back(idx);
    }
  }
  if (m_dstCycle.empty())
  {
    m_dstCycle.push_back(0);
  }
  m_dstCyclePos = 0;
  m_currentDst = m_dstCycle.front();
}

uint32_t LeoRoutingGymEnv::ResolveDestination(uint32_t) const
{
  if (m_trainDst >= 0)
  {
    return static_cast<uint32_t>(std::max<int32_t>(0, m_trainDst));
  }
  return m_currentDst;
}

Ptr<QTableRouting> LeoRoutingGymEnv::GetRouterPtr(uint32_t nodeId) const
{
  if (m_routers.size() > nodeId && m_routers[nodeId])
  {
    return m_routers[nodeId];
  }
  return m_router;
}

std::vector<uint32_t> LeoRoutingGymEnv::GetUpNeighbors(uint32_t nodeId) const
{
  if (!m_range)
  {
    return {};
  }
  auto ups = m_range->GetUpNeighbors(nodeId);
  if (ups.size() > m_K)
  {
    ups.resize(m_K);
  }
  return ups;
}

void LeoRoutingGymEnv::BuildObsForNode(uint32_t nodeId,
                                       std::vector<float>& obs,
                                       std::vector<uint32_t>& mask,
                                       std::vector<float>& neiFeat)
{
  obs.clear();
  mask.clear();
  neiFeat.assign(m_K * kNeighborFeatureDim, 0.0f);
  std::vector<float> queueNorms(m_K, 0.0f);
  constexpr uint32_t kFeatPerNei = 6;
  constexpr uint32_t kGlobals = 3;
  obs.reserve(m_K * kFeatPerNei + kGlobals);
  mask.reserve(m_K);

  const auto ups = GetUpNeighbors(nodeId);
  m_lastUpNeis = ups;
  if (m_lastUpNeis.size() > m_K)
  {
    m_lastUpNeis.resize(m_K);
  }
  const uint32_t degree = ups.size();
  const uint32_t dstId = ResolveDestination(nodeId);

  const double queueNormBase = std::max(1e-6, m_queueCapacityPkts);

  auto clampUnit = [](double v) {
    if (!std::isfinite(v))
    {
      return 0.0;
    }
    return std::clamp(v, 0.0, 1.0);
  };

  for (uint32_t idx = 0; idx < m_K; ++idx)
  {
    if (idx < degree)
    {
      const uint32_t nei = ups[idx];
      const double rttNorm = clampUnit(GetRttMs(nodeId, nei) / 200.0);
      const double qPkts = std::max(0.0, GetQueuePkts(nei));
      const double queueNorm = clampUnit(qPkts / queueNormBase);
      const double visNorm = clampUnit(GetVisRemaining(nodeId, nei) / 10.0);
      const double spfNorm = clampUnit(static_cast<double>(GetSpfHopsVia(nodeId, nei, dstId)) / 10.0);
      const double ewmaDelayNorm = clampUnit(GetEwmaDelayToDstMs(nodeId, nei, dstId) / 200.0);

      obs.push_back(1.0f);                                           // up flag
      obs.push_back(static_cast<float>(rttNorm));
      obs.push_back(static_cast<float>(queueNorm));
      obs.push_back(static_cast<float>(visNorm));
      obs.push_back(static_cast<float>(spfNorm));
      obs.push_back(static_cast<float>(ewmaDelayNorm));
      mask.push_back(1);

      const size_t base = idx * kNeighborFeatureDim;
      neiFeat[base + 0] = static_cast<float>(ewmaDelayNorm);
      neiFeat[base + 1] = static_cast<float>(queueNorm);
      neiFeat[base + 2] = static_cast<float>(visNorm);
      neiFeat[base + 3] = static_cast<float>(spfNorm);
      neiFeat[base + 4] = 1.0f;
      queueNorms[idx] = static_cast<float>(queueNorm);
    }
    else
    {
      for (uint32_t f = 0; f < kFeatPerNei; ++f)
      {
        obs.push_back(0.0f);
      }
      mask.push_back(0);
    }
  }

  int32_t teacherIdxLocal = -1;
  if (m_teacherPolicy == "qtable")
  {
    teacherIdxLocal = FindTeacherIndex(nodeId, m_lastUpNeis);
  }
  else if (m_teacherPolicy == "ospf")
  {
    teacherIdxLocal = FindTeacherIndexOspf(nodeId, m_lastUpNeis);
  }

  uint32_t teacherNodeId = std::numeric_limits<uint32_t>::max();
  if (teacherIdxLocal >= 0 && static_cast<size_t>(teacherIdxLocal) < m_lastUpNeis.size())
  {
    teacherNodeId = m_lastUpNeis[static_cast<size_t>(teacherIdxLocal)];
  }

  if (nodeId == m_curNode)
  {
    m_lastMask.assign(mask.begin(), mask.end());
    m_lastTeacherIdx = teacherIdxLocal;
    m_lastTeacherNeighbor = teacherNodeId;
    m_lastQueueNorm = queueNorms;
  }

  bool anyValid = false;
  for (uint32_t b : mask)
  {
    if (b > 0)
    {
      anyValid = true;
      break;
    }
  }
  if (!anyValid && !mask.empty())
  {
    mask[0] = 1u;
  }

  const float degreeNorm = (m_K > 0)
                                 ? static_cast<float>(std::clamp(static_cast<double>(degree) /
                                                                  static_cast<double>(m_K),
                                                                  0.0,
                                                                  1.0))
                                 : 0.0f;
  const float dstIsGs = (m_dstMask && dstId < m_dstMask->size() && (*m_dstMask)[dstId]) ? 1.0f : 0.0f;
  const double progressRaw = (m_simTime > 0.0)
                                 ? Simulator::Now().GetSeconds() / m_simTime
                                 : 0.0;
  const float progress = static_cast<float>(clampUnit(progressRaw));

  obs.push_back(degreeNorm);
  obs.push_back(dstIsGs);
  obs.push_back(progress);

}

void LeoRoutingGymEnv::SetTeacherPolicy(const std::string& policy)
{
  std::string lowered = policy;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);
  if (lowered == "qtable" || lowered == "ospf" || lowered == "none")
  {
    m_teacherPolicy = lowered;
  }
  else
  {
    std::cout << "[RL/ENV] unknown teacher policy '" << policy << "', defaulting to qtable\n";
    m_teacherPolicy = "qtable";
  }
}

int32_t LeoRoutingGymEnv::FindTeacherIndex(uint32_t nodeId, const std::vector<uint32_t>& upNeis) const
{
  if (upNeis.empty())
  {
    return -1;
  }
  Ptr<QTableRouting> router = (m_routers.size() > nodeId && m_routers[nodeId]) ? m_routers[nodeId] : m_router;
  if (!router)
  {
    return -1;
  }
  const uint32_t dstId = ResolveDestination(nodeId);
  if (!router->IsDestinationTracked(dstId))
  {
    return -1;
  }
  double bestCost = std::numeric_limits<double>::infinity();
  uint32_t bestNei = std::numeric_limits<uint32_t>::max();
  for (uint32_t nei : upNeis)
  {
    double q = router->GetQValue(dstId, nei);
    if (!std::isfinite(q))
    {
      continue;
    }
    if (q < bestCost)
    {
      bestCost = q;
      bestNei = nei;
    }
  }
  if (bestNei == std::numeric_limits<uint32_t>::max())
  {
    return -1;
  }
  for (size_t idx = 0; idx < upNeis.size(); ++idx)
  {
    if (upNeis[idx] == bestNei)
    {
      return static_cast<int32_t>(idx);
    }
  }
  return -1;
}

int32_t LeoRoutingGymEnv::FindTeacherIndexOspf(uint32_t nodeId, const std::vector<uint32_t>& upNeis) const
{
  if (upNeis.empty())
  {
    return -1;
  }
  Ptr<QTableRouting> router = (m_routers.size() > nodeId && m_routers[nodeId]) ? m_routers[nodeId] : m_router;
  if (!router)
  {
    return -1;
  }
  const uint32_t dstId = ResolveDestination(nodeId);
  if (!router->IsDestinationTracked(dstId))
  {
    return -1;
  }
  int32_t nh = router->GetSpfNextHop(dstId);
  if (nh < 0)
  {
    return -1;
  }
  const uint32_t nextHop = static_cast<uint32_t>(nh);
  for (size_t idx = 0; idx < upNeis.size(); ++idx)
  {
    if (upNeis[idx] == nextHop)
    {
      return static_cast<int32_t>(idx);
    }
  }
  return -1;
}

void LeoRoutingGymEnv::AccumulateStats(uint64_t &deliv, uint64_t &drops, double &delayMs)
{
  deliv = 0;
  drops = 0;
  delayMs = 0.0;

  for (const auto& st : ::g_flowSummaries)
  {
    deliv += st.rxPackets;
    const uint64_t tx = st.txPackets;
    if (tx > st.rxPackets)
    {
      drops += (tx - st.rxPackets);
    }
    delayMs += st.delaySum * 1000.0; // Stored in seconds
  }
}

double LeoRoutingGymEnv::ComputeReward(uint32_t nodeId, uint32_t dstId)
{
  uint64_t totDelivered = 0;
  uint64_t totDrops = 0;
  double totDelayMs = 0.0;
  AccumulateStats(totDelivered, totDrops, totDelayMs);

  uint64_t deltaDelivered = 0;
  if (totDelivered >= m_pktsDeliveredPrev)
  {
    deltaDelivered = totDelivered - m_pktsDeliveredPrev;
  }

  uint64_t deltaDrops = 0;
  if (totDrops >= m_pktsDroppedPrev)
  {
    deltaDrops = totDrops - m_pktsDroppedPrev;
  }

  m_pktsDeliveredPrev = totDelivered;
  m_pktsDroppedPrev = totDrops;
  m_delaySumPrevMs = totDelayMs;

  double reward = 0.0;
  if (deltaDelivered > 0)
  {
    reward += 1.0;
  }
  if (deltaDrops > 0)
  {
    reward -= m_betaDrop;
  }

  const uint64_t key = Key(nodeId, dstId);
  uint32_t chosenNei = std::numeric_limits<uint32_t>::max();
  {
    std::lock_guard<std::mutex> lk(m_actionMu);
    auto it = m_actionMap.find(key);
    if (it != m_actionMap.end())
    {
      chosenNei = it->second;
    }
  }

  if (chosenNei != std::numeric_limits<uint32_t>::max())
  {
    double curEwma = GetEwmaDelayToDstMs(nodeId, chosenNei, dstId);
    if (!std::isfinite(curEwma))
    {
      curEwma = 0.0;
    }
    double prevEwma = curEwma;
    auto prevIt = m_prevEwmaMs.find(key);
    if (prevIt != m_prevEwmaMs.end())
    {
      prevEwma = prevIt->second;
    }
    const double prog = prevEwma - curEwma;
    const double progNorm = std::clamp(prog / 50.0, -1.0, 1.0);
    reward += 0.1 * progNorm;

    double qPkts = GetQueuePkts(chosenNei);
    if (!std::isfinite(qPkts) || qPkts < 0.0)
    {
      qPkts = 0.0;
    }
    const double qFrac = std::clamp(qPkts / std::max(1e-6, m_queueCapacityPkts), 0.0, 1.0);
    reward -= m_lambdaQ * qFrac;

    m_prevEwmaMs[key] = curEwma;
  }
  else
  {
    m_prevEwmaMs.erase(key);
  }

  return std::clamp(reward, -1.0, 1.0);
}

double LeoRoutingGymEnv::GetRttMs(uint32_t nodeId, uint32_t neiId) const
{
  Ptr<QTableRouting> router = (m_routers.size() > nodeId && m_routers[nodeId]) ? m_routers[nodeId] : m_router;
  if (!router)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return router->GetNeighborCostMs(neiId);
}

double LeoRoutingGymEnv::GetQueuePkts(uint32_t) const
{
  return 0.0;
}

double LeoRoutingGymEnv::GetVisRemaining(uint32_t, uint32_t) const
{
  return 0.0;
}

uint32_t LeoRoutingGymEnv::GetSpfHopsVia(uint32_t, uint32_t, uint32_t) const
{
  return 0;
}

double LeoRoutingGymEnv::GetEwmaDelayToDstMs(uint32_t, uint32_t, uint32_t) const
{
  return 0.0;
}

} // namespace ns3
