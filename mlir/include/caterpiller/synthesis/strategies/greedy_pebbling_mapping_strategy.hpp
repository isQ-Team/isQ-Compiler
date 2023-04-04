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
        static_assert( mt::has_foreach_node_v<LogicNetwork>, "LogicNetwork does not implement the foreach_node method" );
        static_assert( mt::has_is_constant_v<LogicNetwork>, "LogicNetwork does not implement the is_constant method" );
        static_assert( mt::has_is_pi_v<LogicNetwork>, "LogicNetwork does not implement the is_pi method" );
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

        // set start, target and all nodes
        std::set<node_t> start;
        std::set<node_t> target;
        std::set<node_t> nodes;
        view.foreach_node( [&] ( auto node ) {
            if (view.is_constant(node) || view.is_pi(node)) start.insert(node);
            nodes.insert(node);
        } );
        ntk.foreach_po( [&] ( auto sig, auto po_index ) {
            auto po = ntk.get_node(sig);
            target.insert(po);
            pos.insert(po);
        } );
        greedy_pebble(nodes, start, target);
        /*
        compute_connected_components(ntk);
        for (auto connected_component : connected_components) {
            greedy_pebble(connected_component, ntk);
        }*/
        // TODO
        return true;
    }

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

    // target is a subset of s
    void compute_set_steps(std::set<node_t>& s, std::set<node_t>& target, bool reverse = false) {
        auto it = this->steps().end();
        _view.foreach_node( [&] ( auto node ) {
            if (_view.is_constant(node) || _view.is_pi(node)) return;
            if (!s.count(node)) return;
            if (target.count(node)) {
                if (reverse) {
                    if (pos.count(node)) return;
                    it = this->steps().insert(it, {node, uncompute_action{}});
                    it++;
                } else {
                    it = this->steps().insert(it, {node, compute_action{}});
                    it++;
                }
            } else {
                it = this->steps().insert(it, {node, compute_action{}});
                it++;
                it = this->steps().insert(it, {node, uncompute_action{}});
            }
        } );
        /*
        std::cout << "print steps: " << std::endl;
        auto it_t = this->steps().begin();
        while (it_t != this->steps().end()) {
            std::cout << it_t->first << std::endl;
            it_t++;
        }
        std::cout << "print steps ends here" << std::endl;*/
    }

    void greedy_pebble(std::set<node_t> nodes, std::set<node_t> start, std::set<node_t> target, bool reverse = false) {
        std::vector<std::set<node_t>> connected_components;
        compute_connected_components(nodes, connected_components);

        for (auto connected_component : connected_components) {
            divide_and_conquer(connected_component, start, intersection(connected_component, target), reverse);
        }
    }
    
    // target <= nodes, intersection(nodes, start) = empty
    void divide_and_conquer(std::set<node_t> nodes, std::set<node_t> start, std::set<node_t> target, bool reverse = false) {
        if (difference(nodes, target).size() <= 4) {
            compute_set_steps(nodes, target, reverse);
            return;
        }
        int beta = nodes.size() * 2 / 3;

        std::set<node_t> A;
        std::set<node_t> B = difference(nodes, A);  // V - A
        std::set<node_t> C = RN(start, nodes);
        int target_size = nodes.size() - beta;
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
                /*
                std::set<node_t> i_A;  // {i} + A
                std::copy(A.begin(), A.end(), std::inserter(i_A, i_A.begin()));
                i_A.insert(i);
                std::set<node_t> rn_i_A = RN(set_union(i_A, start), nodes);
                std::set<node_t> n_i_B = intersection(B, node_to_fanout[i]);
                
                if (rn_i_A.size() > rn_max_size || (rn_i_A.size() == rn_max_size && n_i_B.size() > n_max_size)) {
                    next = i;
                    rn_max_size = rn_i_A.size();
                    n_max_size = n_i_B.size();
                }*/
                /*
                std::cout << i << std::endl;
                std::cout << "rn_i_A.size: " << rn_i_A.size() << std::endl;
                std::cout << "rn_max_size: " << rn_max_size << std::endl;
                std::cout << "n_i_B.size: " << n_i_B.size() << std::endl;
                std::cout << "n_max_size: " << n_max_size << std::endl;
                */
                
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
            //C = RN(set_union(A, start), nodes);
            /*
            print_to(std::cout, A, "A");
            print_to(std::cout, B, "B");
            print_to(std::cout, C, "C");
            */
        }
        std::set<node_t> fanout_of_start = RN(start, nodes);
        C = difference(C, fanout_of_start);
        //C = RN(A, nodes);
        std::set<node_t> S = RP(difference(B, C), nodes);
        A = difference(A, S);
        B = difference(B, S);
        /*
        std::cout << "****** computation of s complete!!! ******" << std::endl;
        print_to(std::cout, A, "A");
        print_to(std::cout, B, "B");
        print_to(std::cout, S, "S");
        */
        greedy_pebble(set_union(A, S), start, set_union(S, intersection(target, A)));
        greedy_pebble(B, set_union(S, start), intersection(target, B), reverse);
        greedy_pebble(set_union(A, S), start, set_union(S, intersection(target, A)), true);
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
        pos.clear();
    }
    LogicNetwork _ntk;
    LogicNetwork _view;
    std::set<node_t> pos;
    std::unordered_map<node_t, std::set<node_t>> node_to_fanin;
    std::unordered_map<node_t, std::set<node_t>> node_to_fanout;
};

} // namespace caterpillar
