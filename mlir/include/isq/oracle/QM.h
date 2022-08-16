#ifndef QM_H
#define QM_H

#include <string>
#include <set>
#include <vector>
#include <map>
#include <algorithm>

namespace qm{
    
    using std::string;
    using std::set;
    using std::vector;
    using std::pair;
    using std::map;
    
    struct QMNode
    {
        string bit;
        int onecnt;
        vector<int> val;

        QMNode() {};
        QMNode(string bit, vector<int> val): bit(bit), val(val), onecnt(0) {
            for (auto ch: bit){
                if (ch == '1') this->onecnt += 1;
            }
        };
        QMNode(int val, int N){
            this->val = {val};
            this->onecnt = 0;
            this->bit = "";
            int cnt = 0;
            while (val){
                if (val % 2 == 1){
                    this->bit += '1';
                    this->onecnt += 1;
                }else{
                    this->bit += '0';
                }
                val /= 2;
                cnt += 1;
            }
            for (int i = cnt; i < N; i++){
                this->bit += '0';
            }
            reverse(this->bit.begin(), this->bit.end());
        }
    };

    class QM{
        public:
            QM(int n): N(n) {};
            vector<QMNode> simplify(set<int> A);
            vector<QMNode> merge(set<int> A);
            set<string> optimize(vector<QMNode> nodes);

        private:
            int N;
            vector<int> A;
            bool oneBitDif(string& s1, string& s2);
            bool twoBitDif(string& s1, string& s2);
            string oneBitUnion(string& s1, string& s2);
            vector<pair<string, string>> twoBitUnion(string& s1, string& s2);
            vector<QMNode> getDisjointPoint(vector<QMNode>& nodes);
            pair<vector<string>, vector<string>> optimizeLayer(set<string> now_layer, set<string> nxt_layer);
            int getUnionScore(string& s, set<string>& layer);
            vector<string> getMaxPair(vector<vector<pair<string, string>>>& candidate);
    };
}

#endif