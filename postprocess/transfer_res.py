import os
import json


input_path = 'outputs/llama/new_result.bak.json'
output_path = 'outputs/llama/new_result.json'

old_res = json.load(open(input_path, 'r'))
res = {}

for d in old_res:
    assert len(d) == 1
    for k in d:
        res[k] = int(d[k])

json.dump(res, open(output_path, 'w'))