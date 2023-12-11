import numpy as np
import json

# problems = json.load(open('/home/puhao/thu/ml/data/scienceqa/problems.json'))
problems = json.load(open('/mnt/seagate12t/VQA/scienceqa/problems.json'))

# static the choices
choices = {}  # key: choice, value: count
for p in problems.values():
    cs = p['choices']
    for c in cs:
        if c not in choices:
            choices[c] = 0
        choices[c] += 1

# sort the choices
NUM_CLASSES = 1000
choices = sorted(choices.items(), key=lambda x: x[1], reverse=True)
# caculte the top 100 choices / all choices
top_num = choices[:NUM_CLASSES]
top_num = sum([x[1] for x in top_num])
all_num = sum([x[1] for x in choices])
print(top_num, all_num, top_num / all_num)

# static the top-choices can cover how many problems
top_choices = [x[0] for x in choices[:NUM_CLASSES]]
top_choices = set(top_choices)
cover = 0
for p in problems.values():
    cs = p['choices']
    is_cover = True
    for c in cs:
        if c not in top_choices:
            is_cover = False
            break
    if is_cover:
        cover += 1
print(cover, len(problems), cover / len(problems))
        