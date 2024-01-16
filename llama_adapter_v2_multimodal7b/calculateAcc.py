import numpy as np
import json

with open('./output_evaluation_old_512/test_problems_old_512.txt', 'r') as f:
    test_result_old = f.readlines()

test_result_old = [t.split('\t') for t in test_result_old]
# test_acc_old = 0
# for i, df in enumerate(test_result_old):
#     if df[4][1] != '!':
#         print('DF:', df)
#     if int(df[3]) == int(df[4][0]):
#         test_acc_old += 1
# test_acc_old /= len(test_result_old)

with open('./output_evaluation_new_512/test_problems_new_512.txt', 'r') as f:
    test_result_new = f.readlines()

test_result_new = [t.split('\t') for t in test_result_new]
# test_acc_new = 0
# none_df = 0
# for i, df in enumerate(test_result_new):
#     if df[4][1] != '!':
#         print('DF:', df)
#     if int(df[3]) == int(df[4][0]):
#         test_acc_new += 1
# test_acc_new /= len(test_result_new)


old_result_list = []
for i, df in enumerate(test_result_old):
    # c = df[0].split('\'')[1]
    old_result_list.append({df[0].split('\'')[1]: df[3]})

new_result_list = []
for j, df in enumerate(test_result_new):
    new_result_list.append({df[0].split('\'')[1]: df[3]})

json_old_result = json.dumps(old_result_list)
json_new_result = json.dumps(new_result_list)

with open('./old_result.json', 'w') as f:
    f.write(json_old_result)

with open('./new_result.json', 'w') as f:
    f.write(json_new_result)