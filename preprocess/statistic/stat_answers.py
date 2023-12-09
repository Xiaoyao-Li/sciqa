import numpy as np
import json

problems = json.load(open('/home/puhao/thu/ml/data/scienceqa/problems.json'))

# static the choices
answers = {}  # key: choice, value: count
for p in problems.values():
    cs = p['choices']
    ans = cs[p['answer']]
    if ans not in answers:
        answers[ans] = 0
    answers[ans] += 1

# sort the choices
NUM_CLASSES = 1024
answers = sorted(answers.items(), key=lambda x: x[1], reverse=True)
# caculte the top 100 choices / all choices
top_num = answers[:NUM_CLASSES]
# print(top_num)
top_num = sum([x[1] for x in top_num])
all_num = sum([x[1] for x in answers])
print(top_num, all_num, top_num / all_num)

        