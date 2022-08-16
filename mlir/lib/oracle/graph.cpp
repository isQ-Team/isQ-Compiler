#include "isq/oracle/graph.h"
using namespace graph;
using std::make_pair;

bool Graph::addNode(int node){
    if (this->nodes.count(node) != 0) return false;
    this->nodes.insert(node);
    NodeAttr attr;
    this->nodesMap.insert(make_pair(node, attr));
    return true;
}

bool Graph::addEdge(int s, int e){
    if (this->nodes.count(s) == 0) return false;
    if (this->nodes.count(e) == 0) return false;

    auto p = make_pair(s, e);
    if (this->egde.count(p) != 0) return false;
    if (!this->isDirect){
        if (this->egde.count(make_pair(e, s)) != 0) return false;
    }

    this->egde.insert(p);
    this->nodesMap[s].outDegree += 1;
    this->nodesMap[s].nxtNodeList.insert(e);
    this->nodesMap[e].inDegree += 1;
    this->nodesMap[e].preNodeList.insert(s);

    if (!this->isDirect){
        this->egde.insert(make_pair(e, s));
        this->nodesMap[s].inDegree += 1;
        this->nodesMap[s].preNodeList.insert(e);
        this->nodesMap[e].outDegree += 1;
        this->nodesMap[e].nxtNodeList.insert(s);
    }
    return true;
}

void Graph::removeEdge(int s, int e){

    auto p = make_pair(s, e);
    if (this->egde.count(p) == 0) return;
    this->egde.erase(p);

    this->nodesMap[s].outDegree -= 1;
    this->nodesMap[s].nxtNodeList.erase(e);
    this->nodesMap[e].inDegree -= 1;
    this->nodesMap[e].preNodeList.erase(s);
}

void Graph::removeNode(int node){

    if (this->nodes.count(node) == 0) return;

    for (auto pre: this->nodesMap[node].preNodeList){
        this->nodesMap[pre].outDegree -= 1;
        this->nodesMap[pre].nxtNodeList.erase(node);
    }
    for (auto nxt: this->nodesMap[node].nxtNodeList){
        this->nodesMap[nxt].inDegree -= 1;
        this->nodesMap[nxt].preNodeList.erase(node);
    }
    
    this->nodes.erase(node);
    this->nodesMap.erase(node);

}