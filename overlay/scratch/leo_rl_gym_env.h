// scratch/leo_rl_gym_env.h — Minimal ns3-gym environment skeleton for RL override
#pragma once

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/opengym-module.h"

#include "qtable-routing-range.h"

#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <atomic>

class RangeManager;

namespace ns3 {

struct RlDiagStep
{
  uint32_t nodeId{0};
  uint32_t dstId{0};
  std::vector<uint32_t> upNeiNodeIds;
  std::vector<uint8_t>  mask;
  int32_t teacherIdx{-1};
  uint32_t teacherNodeId{std::numeric_limits<uint32_t>::max()};
};

class LeoRoutingGymEnv : public OpenGymEnv
{
public:
  static TypeId GetTypeId(void);
  LeoRoutingGymEnv();
  LeoRoutingGymEnv(Time stepTime,
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
                   const std::vector<bool>* dstMask);
  // Overload: pass per-node routers so we can target m_curNode
  LeoRoutingGymEnv(Time stepTime,
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
                   const std::vector<bool>* dstMask);

  // OpenGymEnv API
  Ptr<OpenGymSpace> GetObservationSpace() override;
  Ptr<OpenGymSpace> GetActionSpace() override;
  // Expose obs and mask separately via Dict: {"obs": Box, "mask": Box}
  Ptr<OpenGymDataContainer> GetObservation() override;
  float GetReward() override;
  bool GetGameOver() override;
  bool ExecuteActions(Ptr<OpenGymDataContainer> action) override;
  std::string GetExtraInfo() override;

  void DoDispose() override;
  void Start();
  void LogHeartbeat();
  void SetTeacherPolicy(const std::string& policy);
  // Focus observations/actions on a specific node id; -1 to disable
  void SetFocusNode(int32_t nodeId);

  // Agent attach tracking for optional startup barrier
  static bool AgentAttached();
  static void MarkAgentAttached();

private:
  void ScheduleNextStateRead();
  void Step();
  void InitDestinationCycle();
  double ComputeReward(uint32_t nodeId, uint32_t dstId);
  uint32_t ResolveDestination(uint32_t nodeId) const;
  std::vector<uint32_t> GetUpNeighbors(uint32_t nodeId) const;
  void BuildObsForNode(uint32_t nodeId,
                      std::vector<float>& obs,
                      std::vector<uint32_t>& mask,
                      std::vector<float>& neiFeat);
  int32_t FindTeacherIndex(uint32_t nodeId, const std::vector<uint32_t>& upNeis) const;
  int32_t FindTeacherIndexOspf(uint32_t nodeId, const std::vector<uint32_t>& upNeis) const;
  static inline uint64_t Key(uint32_t a, uint32_t b)
  {
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
  }

  Ptr<QTableRouting> GetRouterPtr(uint32_t nodeId) const;

  // wiring
  Time m_stepTime;
  uint32_t m_K{0};
  Ptr<QTableRouting> m_router; // optional single router
  std::vector< Ptr<QTableRouting> > m_routers; // preferred per-node routers
  Ptr<RangeManager> m_range;

  // penalties
  double m_lambdaHop{0.0}, m_lambdaQ{0.0}, m_betaDrop{0.0};
  double m_queueCapacityPkts{1.0};
  double m_measureStart{0.0};
  double m_simTime{1.0};
  int32_t m_trainDst{-1};
  const std::vector<bool>* m_dstMask{nullptr};
  std::vector<uint32_t> m_dstCycle;
  uint32_t m_dstCyclePos{0};
  uint32_t m_currentDst{0};

  // round-robin over nodes
  uint32_t m_curNode{0};
  int32_t  m_focusNode{-1};
  uint64_t m_stepStamp{0};
  EventId  m_ev;
  EventId  m_heartbeatEv;
  double   m_lastInvalidLog{-1.0};
  std::vector<uint32_t> m_lastUpNeis;
  std::vector<uint8_t> m_lastMask;
  int32_t m_lastTeacherIdx{-1};
  uint32_t m_lastTeacherNeighbor{std::numeric_limits<uint32_t>::max()};
  double   m_lastTeacherCost{std::numeric_limits<double>::quiet_NaN()};
  std::vector<float> m_lastQueueNorm;
  std::string m_teacherPolicy{"qtable"};
  std::unordered_map<uint64_t, uint32_t> m_actionMap;
  std::mutex m_actionMu;
  std::unordered_map<uint64_t, double> m_prevEwmaMs;
  std::unordered_map<uint64_t, RlDiagStep> m_lastDiag;

  // book-keeping for reward between steps
  uint64_t m_pktsDeliveredPrev{0};
  uint64_t m_pktsDroppedPrev{0};
  double   m_delaySumPrevMs{0.0};

  // Reward shaping placeholders and control-cost accounting
  uint64_t m_stepOverrides{0};      // overrides applied during the last Δt
  double   m_lastAvgDelay{0.0};     // seconds over last Δt
  double   m_lastDropRate{0.0};     // drops_Δt / pkts_Δt
  double   m_lastMeanQueue{0.0};    // 0..1 normalized queue level

  static constexpr uint32_t kNeighborFeatureDim = 5;

  uint64_t m_cntNegTeacher{0};
  uint64_t m_cntMaskedReject{0};
  uint64_t m_cntOobReject{0};
  uint64_t m_cntApplied{0};
  uint64_t m_cntTeacherApplied{0};
  uint64_t m_cntTeacherMissing{0};

  void SetRlOverride(uint32_t nodeId, uint32_t dstId, uint32_t neiId, double ttlSec);
  Ptr<QTableRouting> ClearRlOverride(uint32_t nodeId, uint32_t dstId);
  void UseTeacherPath(uint32_t nodeId, uint32_t dstId);

  void AccumulateStats(uint64_t &deliv, uint64_t &drops, double &delayMs);
  double GetRttMs(uint32_t i, uint32_t j) const;
  double GetQueuePkts(uint32_t i) const;
  double GetVisRemaining(uint32_t i, uint32_t j) const;
  uint32_t GetSpfHopsVia(uint32_t i, uint32_t j, uint32_t d) const;
  double GetEwmaDelayToDstMs(uint32_t i, uint32_t j, uint32_t d) const;
};

} // namespace ns3
