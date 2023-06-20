import random
import sys

graph = [[0, 12, 3],[11,6,10],[12,10,11],[5,7,0],[9,13,6],[3,11,9],[6,1,4],[10,3,13],[13,9,12],[4,8,5],[7,2,1],[1,5,2],[2,0,8],[8,4,7]]

max_depth = int(sys.argv[1])
#print("max depth: "+str(max_depth))
n = 14
stats = n * [0]
runtime = 1000
ans = 6

def gen_newcoin(coin):
    seed = random.random()
    if seed < 0.5:
        return (coin + 1) % 3
    else:
        return (coin + 2) % 3

def walk(step, now, coin):
    if step == max_depth:
        stats[now] += 1
        return
    newcoin = gen_newcoin(coin)
    walk(step+1, graph[now][newcoin], newcoin)

for i in range(runtime):
    walk(0, 0, 0)

#print(stats)
print(stats[ans]/runtime)
# print(graph[3][2])


