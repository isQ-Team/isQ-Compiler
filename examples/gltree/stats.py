import os
import json

sim_file = '~/prgs/gltree_size14.so'
freq_dict = {}

for iter in range (21, 25):
    print("iter: "+ str(iter))
    simulate_cmd = f'isqc simulate -i {iter} -d 0 --shots 100 {sim_file}'
    #print(simulate_cmd)
    res = os.popen(simulate_cmd).read()

    try:
        test_res = json.loads(res)
        exception = 0
                             
        for node in test_res:
            if node == '0110':
                # print('found!')
                freq_dict[iter] = test_res[node] / 100
    except:
        print('simulate error')
        print(res)

print(freq_dict)
