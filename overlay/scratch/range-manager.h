#pragma once
#include "ns3/object.h"
#include "ns3/node.h"
#include "ns3/net-device.h"
#include "ns3/net-device-container.h"  // for NetDeviceContainer
#include "ns3/vector.h"                // for Vector
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <algorithm>

namespace ns3 {

// Minimal link-visibility manager (keeps all links "up" in this skeleton).
class RangeManager : public Object {
public:
  static TypeId GetTypeId() {
    static TypeId tid = TypeId("RangeManager").SetParent<Object>();
    return tid;
  }

  // Harness uses this to read which links are up
  std::unordered_map<uint64_t,bool> GetLinkBitmap() const { return m_up; }

  // Implemented in the harness .cc (below)
  void Setup(uint32_t n, double rangeKm, double tick);
  void Start();
  void Attach(Ptr<Node> a, Ptr<Node> b, NetDeviceContainer devs);
  void SetForceCleanLinks(bool on) { m_forceCleanLinks = on; }
  void SetStartDelay(double seconds) { m_startDelay = seconds; }

private:
  void Tick();
  static inline uint64_t Key(uint32_t a,uint32_t b){
    return (uint64_t)std::min(a,b)<<32 | std::max(a,b);
  }

  uint32_t m_n{0};
  double   m_range{1e6};
  double   m_tick{0.05};
  std::vector<Vector> m_pos;
  std::unordered_map<uint64_t, bool> m_up;
  std::unordered_map<uint64_t, NetDeviceContainer> m_links;
  bool m_forceCleanLinks{false};
  double m_startDelay{0.05};
};

} // namespace ns3
