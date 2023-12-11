import numpy as np
import json

problems = json.load(open('/mnt/seagate12t/VQA/scienceqa/problems.json'))

# static the choices
hints = []
for p in problems.values():
    hint = p['hint']
    if hint != '':
        hints.append(hint)

print(len(hints), len(problems), len(hints) / len(problems))
