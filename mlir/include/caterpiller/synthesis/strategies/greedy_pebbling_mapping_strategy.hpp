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

        // topo_view rules out the nodes unreachable from pos
        mockturtle::topo_view view{ntk};
        _ntk = ntk;
        _view = view;
        construct_connectivity();

        // validate target size
        if (recursive_target_ratio) recursive_target_size = int(_view.size() * recursive_target_ratio);
        if (recursive_target_size <= 5) recursive_target_size = 5;

        // set start, target and all nodes
        std::set<node_t> start;
        std::set<node_t> target;
        std::set<node_t> nodes;
        view.foreach_node( [&] ( auto node ) {
            if (view.is_constant(node) || view.is_pi(node)) start.insert(node);
            nodes.insert(node);
            hasbeen_computed[node] = false;
        } );
        ntk.foreach_po( [&] ( auto sig, auto po_index ) {
            auto po = ntk.get_node(sig);
            if (!start.count(po)) target.insert(po);
        } );
        greedy_pebble(nodes, start, target);
        return true;
    }

    // target size under which recursion stops
    void set_target_size(int s) { recursive_target_size = s; }

    void set_target_ratio(double r) { recursive_target_ratio = r; }

    void print_connected_component(std::ostream &os, std::vector<std::set<node_t>> connected_components) {
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
    std::set<node_t> RN(std::set<node_t> a, std::set<node_t>& nodes) {
        std::set<node_t> result;
        for (auto node : fanout_set(a, nodes)) {
            if (!nodes.count(node)) continue;
            bool all_fanin_in = true;
            for (auto fanin : node_to_fanin[node]) {
                if (!a.count(fanin)) all_fanin_in = false;
            }
            if (all_fanin_in) result.insert(node);
        }
        return result;
    }

    std::set<node_t> RP(std::set<node_t> a, std::set<node_t>& nodes) {
        std::set<node_t> result;
        for (auto node : a) {
            for (auto fanin : node_to_fanin[node]) {
                if (!nodes.count(fanin)) continue;
                if (a.count(fanin)) continue;
                if (_ntk.is_constant(fanin) || _ntk.is_pi(fanin)) continue;
                result.insert(fanin);
            }
        }
        return result;
    }

    std::set<node_t> fanout_set(std::set<node_t> a, std::set<node_t>& nodes) {
        std::set<node_t> result;
        for (auto node : a) {
            for (auto fanout : node_to_fanout[node]) {
                if (a.count(fanout)) continue;
                if (nodes.count(fanout)) result.insert(fanout);
            }
        }
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

    std::set<node_t> set_union(std::set<node_t> a, std::set<node_t> b) {
        std::set<node_t> result;
        std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::inserter(result, result.begin()));
        return result;
    }

    void compute_node(node_t node, bool compute = true) {
        if (compute == hasbeen_computed[node]) return;
        if (compute) {
            this->steps().emplace_back(node, compute_action{});
            hasbeen_computed[node] = true;
        } else {
            this->steps().emplace_back(node, uncompute_action{});
            hasbeen_computed[node] = false;
        }
    }
    
    void uncompute_eagerly(node_t node, const std::set<node_t>& s, const std::set<node_t>& target, std::map<node_t, int>& ref_count, bool reverse = false) {
        if (_ntk.is_constant(node) || _ntk.is_pi(node)) return;
        for (auto fanin : node_to_fanin[node]) {
            if (!s.count(fanin)) continue;
            --ref_count[fanin];
            if (!ref_count[fanin]) {
                if (!reverse && target.count(fanin)) continue;
                compute_node(fanin, false);
                uncompute_eagerly(fanin, s, target, ref_count, reverse);
            }
        }
    }

    // use eager cleanup
    // target is a subset of s
    void compute_set_steps(std::set<node_t>& s, std::set<node_t>& target, bool reverse = false) {
        std::map<node_t, int> ref_count;
        _view.foreach_node( [&] ( auto node ) {
            if (!s.count(node)) return;
            ref_count[node] = intersection(node_to_fanout[node], s).size();
        } );
        _view.foreach_node( [&] ( auto node ) {
            if (_ntk.is_constant(node) || _ntk.is_pi(node)) return;
            if (!s.count(node)) return;
            if (target.count(node)) {
                if (reverse && ref_count[node]) return;
                compute_node(node, !reverse);
                uncompute_eagerly(node, s, target, ref_count, reverse);
            } else {
                compute_node(node);
            }
        } );
    }
    
    // if reverse = false, begin with pebbling configuration P = start and end with P = start + target. 
    // if reverse = true, begin with pebbling configuration P = start + target and end with P = start. 
    void greedy_pebble(std::set<node_t> nodes, std::set<node_t> start, std::set<node_t> target, bool reverse = false) {
        std::vector<std::set<node_t>> connected_components;
        compute_connected_components(nodes, connected_components);

        for (auto connected_component : connected_components) {
            auto live = live_node(connected_component, target);
            divide_and_conquer(live, start, intersection(live, target), reverse);
        }
    }
    
    // target <= nodes, intersection(nodes, start) = empty
    void divide_and_conquer(std::set<node_t> nodes, std::set<node_t> start, std::set<node_t> target, bool reverse = false) {
        if (nodes.size() - target.size() <= recursive_target_size) {
            compute_set_steps(nodes, target, reverse);
            return;
        }

        std::set<node_t> A;
        std::set<node_t> B = difference(nodes, A);  // V - A
        std::set<node_t> C = RN(start, nodes);
        int target_size = (nodes.size() - target.size()) >> 1;
        while (A.size() <= target_size) {
            int rn_max_size = 0, n_max_size = 0;
            int rn_now_size = C.size();
            node_t next;
            for (auto i : C) {
                int rn_size_i = rn_now_size - 1, n_size_i = 0;
                for (auto fanout : node_to_fanout[i]) {
                    if (!nodes.count(fanout)) continue;
                    bool all_fanin_in_A = true;
                    for (auto fanin_of_fanout : node_to_fanin[fanout]) {
                        if (i == fanin_of_fanout) continue;
                        if (A.count(fanin_of_fanout) || start.count(fanin_of_fanout)) continue;
                        all_fanin_in_A = false;
                    }
                    if (all_fanin_in_A) rn_size_i++;
                    if (B.count(fanout)) n_size_i++;
                }

                if (rn_size_i > rn_max_size || (rn_size_i == rn_max_size && n_size_i > n_max_size)) {
                    next = i;
                    rn_max_size = rn_size_i;
                    n_max_size = n_size_i;
                }
            }
            A.insert(next);
            B.erase(next);  // V - A

            // update C
            C.erase(next);
            for (auto fanout : node_to_fanout[next]) {
                if (!nodes.count(fanout)) continue;
                bool all_fanin_in_A = true;
                for (auto fanin_of_fanout : node_to_fanin[fanout]) {
                    if (next == fanin_of_fanout) continue;
                    if (A.count(fanin_of_fanout) || start.count(fanin_of_fanout)) continue;
                    all_fanin_in_A = false;
                }
                if (all_fanin_in_A) C.insert(fanout);
            }
            /*
            std::cout << next << " added to A" << std::endl;
            print_to(std::cout, A, "A");
            print_to(std::cout, B, "B");
            print_to(std::cout, C, "C");
            */
        }
        std::set<node_t> fanout_of_start = RN(start, nodes);
        C = difference(C, fanout_of_start);

        std::set<node_t> target_C = intersection(target, C);
        std::set<node_t> S = set_union(RP(difference(B, C), nodes), target_C);
        A = difference(A, S);
        B = difference(B, S);
        /*
        std::cout << "****** computation of s complete!!! ******" << std::endl;
        print_to(std::cout, A, "A");
        print_to(std::cout, B, "B");
        print_to(std::cout, S, "S");
        */
        std::set<node_t> target_A = intersection(target, A);
        std::set<node_t> target_B = intersection(target, B);
        std::set<node_t> target_S = intersection(target, S);
        std::set<node_t> none_target_A = difference(A, target_A);
        std::set<node_t> none_target_S = difference(S, target_S);
        if (reverse) {
            greedy_pebble(set_union(none_target_A, none_target_S), set_union(set_union(target_A, start), target_S), difference(S, target_S));
            greedy_pebble(B, set_union(S, start), target_B, true);
            greedy_pebble(set_union(A, S), start, set_union(S, target_A), true);
        } else {
            greedy_pebble(set_union(A, S), start, set_union(S, target_A));
            greedy_pebble(B, set_union(S, start), target_B);
            greedy_pebble(set_union(none_target_A, none_target_S), set_union(set_union(target_A, start), target_S), difference(S, target_S), true);
        }
    }

    // nodes that do lead to a target
    std::set<node_t> live_node(std::set<node_t>& nodes, std::set<node_t>& target) {
        std::set<node_t> alive;
        std::stack<node_t> sta;
        for (auto t : target) {
            if (nodes.count(t)) sta.push(t);
        }
        _ntk.clear_visited();
        while (!sta.empty()) {
            auto node = sta.top();
            sta.pop();
            if (!nodes.count(node)) continue;
            if (_ntk.is_constant(node) || _ntk.is_pi(node)) continue;
            if (_ntk.visited(node)) continue;
            alive.insert(node);
            _ntk.set_visited(node, 1);
            for (auto fanin : node_to_fanin[node]) {
                if (!_ntk.visited(fanin)) sta.push(fanin);
            }
        }
        _ntk.clear_visited();
        return alive;
    }

    void compute_connected_components(std::set<node_t>& nodes, std::vector<std::set<node_t>>& connected_components) 
    {
        _ntk.clear_visited();
        uint32_t visited_flag = 0;
        for (auto node : nodes) {
            std::set<node_t> connected_component;
            std::stack<node_t> sta;
            sta.push(node);
            visited_flag++;
            while (!sta.empty()) {
                auto node = sta.top();
                sta.pop();
                if (!nodes.count(node)) continue;
                if (_ntk.is_constant(node) || _ntk.is_pi(node)) continue;
                if (_ntk.visited(node)) continue;
                connected_component.insert(node);
                _ntk.set_visited(node, visited_flag);
                for (auto fanin : node_to_fanin[node]) {
                    if (!_ntk.visited(fanin)) sta.push(fanin);
                }
                for (auto fanout : node_to_fanout[node]) {
                    if (!_ntk.visited(fanout)) sta.push(fanout);
                }
            }
            if (connected_component.empty()) visited_flag--;
            else connected_components.push_back(connected_component);
        }
        _ntk.clear_visited();
        std::sort(connected_components.begin(), connected_components.end(), [] (std::set<node_t> a, std::set<node_t> b) { return a.size() > b.size(); } );
    }

    void construct_connectivity() {
        _view.foreach_node( [&] ( auto node ) {
            if (node_to_fanin.find(node) == node_to_fanin.end()) node_to_fanin[node] = std::set<node_t>();
            if (node_to_fanout.find(node) == node_to_fanout.end()) node_to_fanout[node] = std::set<node_t>();
            _view.foreach_fanin( node, [&] ( auto fanin ) {
                node_to_fanin[node].insert(fanin.index);
                if (node_to_fanout.find(fanin.index) == node_to_fanout.end()) node_to_fanout[fanin.index] = std::set<node_t>();
                node_to_fanout[fanin.index].insert(node);
            } );
        } );
    }

    inline void clear_containers() {
        node_to_fanin.clear();
        node_to_fanout.clear();
        hasbeen_computed.clear();
    }
    LogicNetwork _ntk;
    LogicNetwork _view;
    std::unordered_map<node_t, bool> hasbeen_computed;
    std::unordered_map<node_t, std::set<node_t>> node_to_fanin;
    std::unordered_map<node_t, std::set<node_t>> node_to_fanout;
    int recursive_target_size = 5;
    double recursive_target_ratio = 0.0;
};

} // namespace caterpillar
