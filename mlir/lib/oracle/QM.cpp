#include "isq/oracle/QM.h"
#include "isq/oracle/graph.h"
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

using std::string;
using std::set;
using std::vector;
using std::pair;
using std::map;
using std::make_pair;
using namespace qm;
using namespace graph;

void myprint(vector<QMNode>& nodes){
    for (auto node: nodes){
        std::cout << node.bit << ": ";
        for (auto val : node.val){
            std::cout << val << ' ';
        }
        std::cout << '\n';
    }
}

void printgraph(Graph& ag){
    std::cout << "graph: \n";
    std::cout << "node: ";
    for (auto node: ag.nodes) std::cout << node << " ";

    std::cout << "\nedge: \n";
    for (auto edge: ag.egde){
        std::cout << edge.first << ' ' << edge.second << '\n';
    }
}

bool QM::oneBitDif(string& s1, string& s2){

    int cnt = 0;
    for (int i = 0; i < this->N; i++){
        if (s1[i] != s2[i]) cnt += 1;
    }
    return (cnt == 1);
}

bool QM::twoBitDif(string& s1, string& s2){

    int cnt = 0;
    for (int i = 0; i < this->N; i++){
        if (s1[i] == '-' && s2[i] != '-') return false;
        if (s2[i] == '-' && s1[i] != '-') return false;
        if (s1[i] != s2[i]) cnt += 1;
    }
    return (cnt == 2);
}

string QM::oneBitUnion(string& s1, string& s2){
    string s = "";
    for (int i = 0; i < this->N; i++){
        if (s1[i] != s2[i]){
            s += '-';
        }else{
            s += s1[i];
        }
    }
    return s;
}

vector<pair<string, string>> QM::twoBitUnion(string& s1, string& s2){
    
    vector<string> union_str;
    for (int i = 0; i < this->N; i++){
        if (s1[i] != s2[i]){
            union_str.push_back(s1.substr(0, i)+'-'+s1.substr(i+1));
            union_str.push_back(s2.substr(0, i)+'-'+s2.substr(i+1));
        }
    }

    vector<pair<string, string>> ans;
    ans.push_back(make_pair(union_str[0], union_str[3]));
    ans.push_back(make_pair(union_str[2], union_str[1]));
    return ans;
}

vector<QMNode> QM::simplify(set<int> A){
    
    vector<QMNode> ans;
    while (A.size() > 0){
        
        auto nodes = this->merge(A);
        for (auto node: nodes){
            ans.push_back(node);
            for (auto val: node.val){
                A.erase(val);
            }
        }
    }
    return ans;
}


vector<QMNode> QM::merge(set<int> A){

    vector<QMNode> nodes;
    for (auto val: A){
        nodes.push_back(QMNode(val, this->N));
    }
    
    while (true){
        // group first
        map<int, vector<QMNode>> group;
        for (int i = 0; i <= this->N+1; i++){
            vector<QMNode> qmnodes;
            group[i] = qmnodes;
        }
        for (auto node: nodes){
            group[node.onecnt].push_back(node);
        }

        vector<QMNode> new_nodes;
        // merge adjacent group
        for (int i = 0; i <= this->N; i++){
            if (group[i].size() == 0 || group[i+1].size() == 0) continue;
            for (auto &node1: group[i]){
                for (auto &node2: group[i+1]){
                    if (this->oneBitDif(node1.bit, node2.bit)){
                        string union_str = this->oneBitUnion(node1.bit, node2.bit);
                        vector<int> val_set = node1.val;
                        val_set.insert(val_set.end(), node2.val.begin(), node2.val.end());
                        new_nodes.push_back(QMNode(union_str, val_set));
                    }
                }
            }
        }
        if (new_nodes.size() == 0) break;
        nodes = new_nodes;
    }

    return this->getDisjointPoint(nodes);

}

vector<QMNode> QM::getDisjointPoint(vector<QMNode>& nodes){
    // build graph
    auto ag = Graph(false);
    int n = nodes.size();
    map<int, vector<int>> v;
    for (int i = 0; i < n; i++){
        ag.addNode(i);
        for (auto val: nodes[i].val){
            if (v.find(val) == v.end()){
                v.insert(make_pair(val, vector<int>({})));
            }
            
            for (auto nxt: v[val]){
                ag.addEdge(i, nxt);
            }

            v[val].push_back(i);
        }
    }
    // greedy : get disjoint point
    set<int> choose;
    while(ag.nodes.size() > 0){
        int d = 100000;
        vector<int> cand_idx;
        for (auto node: ag.nodes){
            if (ag.nodesMap[node].outDegree < d){
                d = ag.nodesMap[node].outDegree;
                cand_idx = {node};
            }
            else if (ag.nodesMap[node].outDegree == d)
            {
                cand_idx.push_back(node);
            }
            
        }
        
        srand((unsigned int)time(NULL));
        int idx = cand_idx[rand() % cand_idx.size()];
        choose.insert(idx);
        auto nxtlist = ag.nodesMap[idx].nxtNodeList;
        for (auto node: nxtlist) ag.removeNode(node);
        ag.removeNode(idx);
    }

    vector<QMNode> ans;
    for (auto idx: choose){
        ans.push_back(nodes[idx]);
    }

    return ans;
}

set<string> QM::optimize(vector<QMNode> nodes){
    // get nodes layers, optimize every layer
    map<int, set<string>> layers;
    for (int i = 0; i <= N+1; i++){
        layers.insert(make_pair(i, set<string>({})));
    }
    for (auto node: nodes){
        int cnt = 0;
        for (auto &ch: node.bit){
            if (ch == '-') cnt += 1;
        }
        layers[cnt].insert(node.bit);
    }
    
    set<string> res;
    for (int i = 0; i <= N; i++){
        if (layers[i].size() == 0) continue;
        auto ans = this->optimizeLayer(layers[i], layers[i+1]);
        //std::cout << i << ": \n[";
        for (auto &bit: ans.first){
        //    std::cout << bit << ' ';
            if (layers[i+1].count(bit) == 1){
                layers[i+1].erase(bit);
            }else{
                layers[i+1].insert(bit);
            }
        }
        //std::cout << "]\n[";
        for (auto &bit: ans.second){
        //    std::cout << bit << ' ';
            layers[i].erase(bit);
        }
        //std::cout << "]\n";
        res.insert(layers[i].begin(), layers[i].end());
    }

    return res;
}


pair<vector<string>, vector<string>> QM::optimizeLayer(set<string> now_layer, set<string> nxt_layer){
    // compare every two string in now_layer
    // choose onebit and twobit dif string pair to merge
    vector<string> layer;
    layer.assign(now_layer.begin(), now_layer.end());
    
    vector<QMNode> candidate;
    for (int x = 0; x < layer.size(); x++){
        for (int y = x+1; y < layer.size(); y++){
            if (this->oneBitDif(layer[x], layer[y])) candidate.push_back(QMNode("", {x, y}));
            if (this->twoBitDif(layer[x], layer[y])) candidate.push_back(QMNode("", {x, y}));
        }
    }
    // choose disjoint string to union
    auto choosen = this->getDisjointPoint(candidate);
    
    vector<string> new_layer;
    vector<string> old_layer;
    vector<vector<pair<string, string>>> temp;
    for (auto &node: choosen){
        int x = node.val[0];
        int y = node.val[1];
        // if one bit dif,union directly
        if (this->oneBitDif(layer[x], layer[y])){
            new_layer.push_back(this->oneBitUnion(layer[x], layer[y]));
        }
        // if two bit dif, two union res, calc union score, choose the one has max score
        // if score is same, add to temp and optimize in getMaxPair function
        if (this->twoBitDif(layer[x], layer[y])){
            auto newstr = this->twoBitUnion(layer[x], layer[y]);
            int score1 = this->getUnionScore(newstr[0].first, nxt_layer) + this->getUnionScore(newstr[0].second, nxt_layer);
            int score2 = this->getUnionScore(newstr[1].first, nxt_layer) + this->getUnionScore(newstr[1].second, nxt_layer);
            if (score1 > score2){
                new_layer.push_back(newstr[0].first);
                new_layer.push_back(newstr[0].second);
            }else if (score1 < score2)
            {
                new_layer.push_back(newstr[1].first);
                new_layer.push_back(newstr[1].second);
            }else{
                temp.push_back(newstr);
            }
            
        }
        old_layer.push_back(layer[x]);
        old_layer.push_back(layer[y]);
    }

    if (temp.size() > 0){
        auto res = this->getMaxPair(temp);
        new_layer.insert(new_layer.end(), res.begin(), res.end());
    }

    return make_pair(new_layer, old_layer);
}

int QM::getUnionScore(string& s, set<string>& layer){
    int score = 0;
    for (auto bit: layer){
        if (s == bit) score += 3;
        if (this->oneBitDif(s, bit)) score += 2;
        if (this->twoBitDif(s, bit)) score += 1;
    }
    return score;
}

vector<string> QM::getMaxPair(vector<vector<pair<string, string>>>& candidate){
    // if size >= 10, select randomly
    vector<string> ans;
    if (candidate.size() >= 10){
        srand((unsigned int)time(NULL));
        for (auto &pair: candidate){
            int ri = rand() % 2;
            ans.push_back(pair[ri].first);
            ans.push_back(pair[ri].second);
        }
    }else{
        // judge every choice, calc score and get the one has max score
        int n = candidate.size();
        int m = (1 << n);
        int best_score = -1;
        set<string> best_choosen;
        for (int i = 0; i < m; i++){
            int score = 0;
            set<string> choosen;
            for (int j = 0; j < n; j++){
                int idx = (i >> j) & 1;
                score += this->getUnionScore(candidate[j][idx].first, choosen);
                score += this->getUnionScore(candidate[j][idx].second, choosen);

                if (choosen.count(candidate[j][idx].first) == 1){
                    choosen.erase(candidate[j][idx].first);
                }
                else{
                    choosen.insert(candidate[j][idx].first);
                }
                if (choosen.count(candidate[j][idx].second) == 1){
                    choosen.erase(candidate[j][idx].second);
                }else{
                    choosen.insert(candidate[j][idx].second);
                }
            }

            if (score > best_score){
                best_score = score;
                best_choosen = choosen;
            }
        }

        ans.assign(best_choosen.begin(), best_choosen.end());
    }

    return ans;
}