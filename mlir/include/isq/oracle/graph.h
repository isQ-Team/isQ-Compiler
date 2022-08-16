#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <set>
#include <map>

namespace graph{

    using std::map;
    using std::set;
    using std::vector;
    using std::pair;

    class NodeAttr{
        
        public:
            int inDegree;
            int outDegree;
            set<int> preNodeList;
            set<int> nxtNodeList;

            NodeAttr(): inDegree(0), outDegree(0) {};
    }; 

    class Graph{

        public:
            bool addNode(int node);
            bool addEdge(int s, int e);
            void removeEdge(int s, int e);
            void removeNode(int node);
            bool isDirect;
            map<int, NodeAttr> nodesMap;
            set<pair<int, int>> egde;
            set<int> nodes;

            Graph(bool direct): isDirect(direct) {};
            
    };
}


#endif