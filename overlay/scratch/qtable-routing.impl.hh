#pragma once
#include "qtable-routing.h"
#include <queue>

namespace ns3 {

// =================== Static helpers ===================
uint64_t QTableRouting::Key(uint32_t a, uint32_t b) {
  if (a > b) std::swap(a, b);
  return (uint64_t(a) << 32) | uint64_t(b);
}
QTableRouting::Shared& QTableRouting::S() {
  static Shared s;
  return s;
}
uint32_t QTableRouting::NodeCount() {
  return S().nodeAddrs.empty() ? 0u : (1u + std::max_element(
      S().nodeAddrs.begin(), S().nodeAddrs.end(),
      [](auto &x, auto &y){ return x.first < y.first; })->first);
}

// =================== TypeId ===================
TypeId QTableRouting::GetTypeId() {
  static TypeId tid = TypeId("ns3::QTableRouting")
    .SetParent<Ipv4RoutingProtocol>()
    .SetGroupName("Internet")
    .AddConstructor<QTableRouting>();
  return tid;
}

// =================== Ctor/Dtor ===================
QTableRouting::QTableRouting() {}
QTableRouting::~QTableRouting() {
  Stop();
}

// =================== Harness wiring ===================
void QTableRouting::Configure(double alpha,
                              double gamma,
                              double eps,
                              Time probeInterval,
                              uint32_t probeFanout,
                              Time penaltyDrop,
                              uint16_t ctrlPort,
                              Ptr<Object> /*rangeMgr*/)
{
  m_alpha = alpha; m_gamma = gamma; m_eps = eps;
  m_probeInterval = probeInterval; m_probeFanout = probeFanout;
  m_penaltyDrop = penaltyDrop; m_ctrlPort = ctrlPort;
}

void QTableRouting::RegisterNodeAddress(uint32_t nodeId, Ipv4Address nodeAddr) {
  m_selfId = nodeId;
  m_selfAddr = nodeAddr;
  m_nodeAddrs[nodeId] = nodeAddr;
  // also publish globally for Dijkstra
  S().nodeAddrs[nodeId] = nodeAddr;
}

void QTableRouting::AddAdjacency(uint32_t neiId,
                                 Ipv4Address myIfAddr,
                                 Ipv4Address neiIfAddr,
                                 uint32_t myOif)
{
  Adj a;
  a.neiId = neiId; a.myIfAddr = myIfAddr; a.neiIfAddr = neiIfAddr; a.myOif = myOif;
  m_nei[neiId] = a;
}

void QTableRouting::SetLinkUp(uint32_t a, uint32_t b, bool up) {
  S().linkUp[Key(a,b)] = up;
}
bool QTableRouting::IsLinkUp(uint32_t a, uint32_t b) {
  auto it = S().linkUp.find(Key(a,b));
  return it != S().linkUp.end() && it->second;
}

// =================== Ipv4RoutingProtocol ===================
void QTableRouting::SetIpv4(Ptr<Ipv4> ipv4) {
  m_ipv4 = ipv4;

  // Create tiny probe socket (optional but handy)
  if (!m_sockRx) {
    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
    m_sockRx = Socket::CreateSocket(m_ipv4->GetObject<Node>(), tid);
    InetSocketAddress local(m_selfAddr, m_ctrlPort);
    m_sockRx->Bind(local);
    m_sockRx->SetRecvCallback([this](Ptr<Socket> s){
      Address from;
      Ptr<Packet> p = s->RecvFrom(from);
      if (!p) return;
      uint8_t b[8] = {0};
      uint32_t n = p->CopyData(b, sizeof(b));
      if (n < 1) return;
      if (b[0] == 0xA1) { // PROBE
        // reply immediately
        uint8_t r[1] = {0xA2}; // ACK
        Ptr<Packet> ack = Create<Packet>(r, sizeof(r));
        s->SendTo(ack, 0, from);
        m_probesRx++;
      }
    });
  }

  Start();
}

void QTableRouting::Start() {
  if (!m_probeEv.IsPending() && m_probeInterval.GetSeconds() > 0.0) {
    m_probeEv = Simulator::Schedule(m_probeInterval, &QTableRouting::ProbeTick, this);
  }
}

void QTableRouting::Stop() {
  if (m_probeEv.IsPending()) m_probeEv.Cancel();
  if (m_sockRx) { m_sockRx->SetRecvCallback(MakeNullCallback<void, Ptr<Socket>>()); m_sockRx->Close(); m_sockRx = nullptr; }
}

Ptr<Ipv4Route> QTableRouting::RouteOutput(Ptr<Packet> /*p*/, const Ipv4Header& header,
                                          Ptr<NetDevice> /*oif*/, Socket::SocketErrno& sockerr)
{
  sockerr = Socket::ERROR_NOROUTETOHOST;
  auto it = m_fwd.find(header.GetDestination());
  if (it == m_fwd.end()) {
    // rebuild once quickly if missing
    RebuildFwdTable();
    it = m_fwd.find(header.GetDestination());
    if (it == m_fwd.end()) return nullptr;
  }

  Ptr<Ipv4Route> rt = Create<Ipv4Route>();
  rt->SetDestination(header.GetDestination());
  rt->SetGateway(it->second.nh);
  rt->SetOutputDevice(m_ipv4->GetNetDevice(it->second.oif));
  rt->SetSource(m_ipv4->GetAddress(it->second.oif, 0).GetLocal());
  sockerr = Socket::ERROR_NOTERROR;
  return rt;
}

bool QTableRouting::RouteInput(Ptr<const Packet> p,
                               const Ipv4Header& header,
                               Ptr<const NetDevice> idev,
                               const UnicastForwardCallback& ucb,
                               const MulticastForwardCallback& mcb,
                               const LocalDeliverCallback& lcb,
                               const ErrorCallback& ecb)

{
  // Local?
  for (uint32_t i=0;i<m_ipv4->GetNInterfaces();++i) {
    for (uint32_t j=0;j<m_ipv4->GetNAddresses(i);++j) {
      if (header.GetDestination() == m_ipv4->GetAddress(i,j).GetLocal()) {
        if (!lcb.IsNull()) lcb(p, header, i);
        return true;
      }
    }
  }

  // Forward
  auto it = m_fwd.find(header.GetDestination());
  if (it == m_fwd.end()) {
    RebuildFwdTable();
    it = m_fwd.find(header.GetDestination());
    if (it == m_fwd.end()) { if (!ecb.IsNull()) ecb(p, header, Socket::ERROR_NOROUTETOHOST); return false; }
  }
  Ptr<Ipv4Route> rt = Create<Ipv4Route>();
  rt->SetDestination(header.GetDestination());
  rt->SetGateway(it->second.nh);
  rt->SetOutputDevice(m_ipv4->GetNetDevice(it->second.oif));
  rt->SetSource(m_ipv4->GetAddress(it->second.oif, 0).GetLocal());
  if (!ucb.IsNull()) ucb(rt, p, header);
  return true;
}

void QTableRouting::PrintRoutingTable(Ptr<OutputStreamWrapper> stream, Time::Unit unit) const {
  std::ostream* os = stream->GetStream();
  *os << "QTableRouting node=" << m_selfId << " entries=" << m_fwd.size() << "\n";
  for (auto &kv : m_fwd) {
    *os << "  " << kv.first << " -> nh=" << kv.second.nh << " oif=" << kv.second.oif << "\n";
  }
}

// =================== Control / learning ===================
void QTableRouting::ProbeTick() {
  // Very small, safe probing: ping up to probeFanout neighbors
  if (!m_nei.empty()) {
    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
    Ptr<Socket> tx = Socket::CreateSocket(m_ipv4->GetObject<Node>(), tid);
    tx->Bind(InetSocketAddress(m_selfAddr, 0));

    uint32_t sent = 0;
    for (auto &kv : m_nei) {
      if (sent >= m_probeFanout) break;
      const Adj& a = kv.second;
      if (!IsLinkUp(m_selfId, a.neiId)) continue;
      // one-byte PROBE (0xA1)
      uint8_t b[1] = {0xA1};
      Ptr<Packet> probe = Create<Packet>(b, sizeof(b));
      Time t0 = Simulator::Now();
      tx->SendTo(probe, 0, InetSocketAddress(a.neiIfAddr, m_ctrlPort));
      m_probesTx++;

      // naive: expect immediate ack; sample ewma RTT if received within penalty window
      // (We keep it simple to avoid heavy timers per probe.)
      // In practice, with short identical delays, this just keeps a small EWMA value.
      double rttMs = static_cast<double>((Simulator::Now() - t0).GetMilliSeconds());
      a.ewmaRttMs = 0.9 * a.ewmaRttMs + 0.1 * std::max(1.0, rttMs);

      sent++;
    }
  }

  // Refresh forwarding table periodically (links may flip)
  RebuildFwdTable();
  m_probeEv = Simulator::Schedule(m_probeInterval, &QTableRouting::ProbeTick, this);
}

// Build a simple graph on current up links and run Dijkstra from self to all nodes.
// Translate first hop into next-hop IP (neighbor iface) + oif.
void QTableRouting::RebuildFwdTable() {
  const uint32_t N = NodeCount();
  if (N == 0) return;

  // Adjacency list with unit weights (you can replace with EWMA-based weights)
  std::vector<std::vector<uint32_t>> g(N);
  for (auto &kv : S().linkUp) {
    if (!kv.second) continue;
    uint32_t a = uint32_t(kv.first >> 32);
    uint32_t b = uint32_t(kv.first & 0xffffffffu);
    if (a < N && b < N) { g[a].push_back(b); g[b].push_back(a); }
  }

  // Dijkstra (unit weights)
  const double INF = 1e18;
  std::vector<double> dist(N, INF);
  std::vector<int> parent(N, -1);
  using P = std::pair<double,uint32_t>;
  std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
  dist[m_selfId] = 0.0; pq.push({0.0, m_selfId});
  while (!pq.empty()) {
    auto [d,u] = pq.top(); pq.pop();
    if (d > dist[u]) continue;
    for (auto v : g[u]) {
      if (dist[v] > d + 1.0) {
        dist[v] = d + 1.0;
        parent[v] = int(u);
        pq.push({dist[v], v});
      }
    }
  }

  // Translate to per-destination next-hop (via first hop on the path)
  m_fwd.clear();
  for (uint32_t dst = 0; dst < N; ++dst) {
    if (dst == m_selfId) continue;
    int u = int(dst), p = parent[u];
    if (p < 0) continue;
    while (p >= 0 && uint32_t(p) != m_selfId) { u = p; p = parent[u]; }
    uint32_t nh = (p < 0) ? uint32_t(u) : uint32_t(u);
    auto it = m_nei.find(nh);
    auto jt = S().nodeAddrs.find(dst);
    if (it == m_nei.end() || jt == S().nodeAddrs.end()) continue;

    FwdEntry fe; fe.nh = it->second.neiIfAddr; fe.oif = it->second.myOif;
    m_fwd[jt->second] = fe; // /32 host route to the loopback-ish address
  }
}

} // namespace ns3

