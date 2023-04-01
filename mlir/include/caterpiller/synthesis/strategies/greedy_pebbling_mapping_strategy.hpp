#pragma once

#include <set>
#include <stack>
#include <vector>
#include <unordered_map>

#include "mapping_strategy.hpp"

#include <mockturtle/views/topo_view.hpp>

namespace caterpillar
{
namespace mt = mockturtle;

template<class LogicNetwork>
class greedy_pebbling_mapping_strategy : public mapping_strategy<LogicNetwork>
{
    using node_t = typename LogicNetwork::node;
public:
    greedy_pebbling_mapping_strategy()
    {
        static_assert( mt::is_network_type_v<LogicNetwork>, "LogicNetwork is not a network type" );
        static_assert( mt::has_foreach_po_v<LogicNetwork>, "LogicNetwork does not implement the foreach_po method" );
        static_assert( mt::has_foreach_fanin_v<LogicNetwork>, "LogicNetwork does not implement the foreach_fanin method" );
        static_assert( mt::has_set_visited_v<LogicNetwork>, "LogicNetwork does not implement the set_visited method" );
        static_assert( mt::has_visited_v<LogicNetwork>, "LogicNetwork does not implement the visited method" );
        static_assert( mt::has_clear_visited_v<LogicNetwork>, "LogicNetwork does not implement the clear_visited method" );
    }

    virtual ~greedy_pebbling_mapping_strategy() = default;

    bool compute_steps( LogicNetwork const& ntk ) override
    {
        clear_containers();
        construct_connectivity(ntk);
        compute_connected_components(ntk);
        for (auto connected_component : connected_components) {
            greedy_pebble(connected_component, ntk);
        }
        // TODO
        return true;
    }

    void print_connected_component(std::ostream &os) {
        os << "******connected component******" << std::endl;
        int i = 0;
        for (auto component : connected_components) {
            print_to(os, component, std::to_string(i++));
        }
    }

    void print_to(std::ostream &os, std::set<node_t> s, std::string info = "") {
        if (info != "") os << info << ": ";
        for (auto e : s) {
            os << e << ", ";
        }
        os << std::endl;
    }

private:
    std::set<node_t> RN(std::set<node_t> a, LogicNetwork const& ntk) {
        std::set<node_t> result;
        for (auto node : a) {
            for (auto fanout : node_to_fanout[node]) {
                if (a.find(fanout) != a.end()) continue;
                if (result.find(fanout) != result.end()) continue;
                bool all_fanin_in = true;
                for (auto fanin : node_to_fanin[fanout]) {
                    if (a.find(fanin) == a.end()) all_fanin_in = false;
                }
                if (all_fanin_in) result.insert(fanout);
            }
        }
        ntk.foreach_pi( [&] ( auto pi ) {
            for (auto fanout : node_to_fanout[pi]) {
                if (a.find(fanout) == a.end()) result.insert(fanout);
            }
        } );
        return result;
    }

    std::set<node_t> RP(std::set<node_t> a, LogicNetwork const& ntk) {
        std::set<node_t> result;
        for (auto node : a) {
            for (auto fanin : node_to_fanin[node]) {
                if (ntk.is_constant(fanin) || ntk.is_pi(fanin)) continue;
                if (a.find(fanin) != a.end()) continue;
                result.insert(fanin);
            }
        }
        ntk.foreach_po( [&] ( auto sig, auto po_index ) {
            auto po = ntk.get_node(sig);
            if (a.find(po) == a.end()) result.insert(po);
        } );
        return result;
    }

    std::set<node_t> intersection(std::set<node_t> a, std::set<node_t> b) {
        std::set<node_t> result;
        std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::inserter(result, result.begin()));
        return result;
    }

    std::set<node_t> difference(std::set<node_t> a, std::set<node_t> b) {
        std::set<node_t> result;
        std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::inserter(result, result.begin()));
        return result;
    }

    void compute_steps_set(std::set<node_t>& s, mockturtle::topo_view<LogicNetwork>& view, bool reverse = false) {
        auto it = this->steps().end();
        if (reverse) {
            view.foreach_node( [&] ( auto node ) {
                if (view.is_constant(node) || view.is_pi(node)) return;
                if (s.find(node) == s.end()) return;
                if (pos.count(node)) return;
                it = this->steps().insert(it, {node, uncompute_action{}});
            } );
        } else {
            view.foreach_node( [&] ( auto node ) {
                if (view.is_constant(node) || view.is_pi(node)) return;
                if (s.find(node) == s.end()) return;
                it = this->steps().insert(it, {node, compute_action{}});
                it++;
            } );
        }
    }

    void greedy_pebble(std::set<node_t> nodes, LogicNetwork const& ntk, int beta = 0) {
        mockturtle::topo_view view{ntk};
        if (nodes.size() <= 4) {
            compute_steps_set(nodes, view);
            compute_steps_set(nodes, view, true);
            return;
        }
        if (!beta) beta = nodes.size() * 2 / 3;

        std::set<node_t> A;
        std::set<node_t> B = difference(nodes, A);  // V - A
        std::set<node_t> C = intersection(RN(A, ntk), nodes);
        int target_size = nodes.size() - beta;
        while (A.size() <= target_size) {
            int rn_max_size = 0, n_max_size = 0;
            node_t next;
            for (auto i : C) {
                std::set<node_t> i_A;  // {i} + A
                std::copy(A.begin(), A.end(), std::inserter(i_A, i_A.begin()));
                i_A.insert(i);
                std::set<node_t> rn_i_A = intersection(RN(i_A, ntk), nodes);
                std::set<node_t> n_i_B = intersection(B, node_to_fanout[i]);/*
                std::cout << i << std::endl;
                std::cout << "rn_i_A.size: " << rn_i_A.size() << std::endl;
                std::cout << "rn_max_size: " << rn_max_size << std::endl;
                std::cout << "n_i_B.size: " << n_i_B.size() << std::endl;
                std::cout << "n_max_size: " << n_max_size << std::endl;*/
                if (rn_i_A.size() > rn_max_size || (rn_i_A.size() == rn_max_size && n_i_B.size() > n_max_size)) {
                    next = i;
                    rn_max_size = rn_i_A.size();
                    n_max_size = n_i_B.size();
                }
            }
            A.insert(next);
            B = difference(nodes, A);  // V - A
            C = intersection(RN(A, ntk), nodes);/*
            print_to(std::cout, A, "A");
            print_to(std::cout, B, "B");
            print_to(std::cout, C, "C");*/
        }
        std::set<node_t> S = intersection(RP(difference(B, C), ntk), nodes);
        ntk.foreach_pi( [&] ( auto pi ) {
            for (auto fanout : node_to_fanout[pi]) {
                S.erase(fanout);
            }
        } );
        A = difference(A, S);
        B = difference(B, S);/*
        print_to(std::cout, A, "A");
        print_to(std::cout, B, "B");
        print_to(std::cout, S, "S");*/
        compute_steps_set(A, view);
        compute_steps_set(S, view);
        compute_steps_set(A, view, true);
        compute_steps_set(B, view);
        compute_steps_set(B, view, true);
        compute_steps_set(A, view);
        compute_steps_set(S, view, true);
        compute_steps_set(A, view, true);
    }

    void compute_connected_components(LogicNetwork const& ntk) 
    {
        ntk.clear_visited();
        uint32_t visited_flag = 0;
        ntk.foreach_po( [&] ( auto sig, auto po_index ) {
            std::set<node_t> connected_component;
            std::stack<node_t> sta;
            auto po = ntk.get_node(sig);
            pos.insert(po);
            sta.push(po);
            visited_flag++;
            while (!sta.empty()) {
                auto node = sta.top();
                sta.pop();
                if (ntk.is_constant(node) || ntk.is_pi(node)) continue;
                if (ntk.visited(node)) continue;
                connected_component.insert(node);
                ntk.set_visited(node, visited_flag);
                for (auto fanin : node_to_fanin[node]) {
                    if (!ntk.visited(fanin)) sta.push(fanin);
                }
                for (auto fanout : node_to_fanout[node]) {
                    if (!ntk.visited(fanout)) sta.push(fanout);
                }
            }
            if (connected_component.empty()) visited_flag--;
            else connected_components.push_back(connected_component);
        } );
        ntk.clear_visited();
        std::sort(connected_components.begin(), connected_components.end(), [] (std::set<node_t> a, std::set<node_t> b) { return a.size() > b.size(); } );
    }

    void construct_connectivity(LogicNetwork const& ntk) {
        // topo_view rules out the nodes unreachable from pos
        mockturtle::topo_view view{ntk};
        view.foreach_node( [&] ( auto node ) {
            if (node_to_fanin.find(node) == node_to_fanin.end()) node_to_fanin[node] = std::set<node_t>();
            ntk.foreach_fanin( node, [&] ( auto fanin ) {
                node_to_fanin[node].insert(fanin.index);
                if (node_to_fanout.find(fanin.index) == node_to_fanout.end()) node_to_fanout[fanin.index] = std::set<node_t>();
                node_to_fanout[fanin.index].insert(node);
            } );
        } );
    }

    inline void clear_containers() {
        connected_components.clear();
        node_to_fanin.clear();
        node_to_fanout.clear();
        pos.clear();
    }
    std::vector<std::set<node_t>> connected_components;
    std::unordered_set<node_t> pos;
    std::unordered_map<node_t, std::set<node_t>> node_to_fanin;
    std::unordered_map<node_t, std::set<node_t>> node_to_fanout;
};

} // namespace caterpillar
