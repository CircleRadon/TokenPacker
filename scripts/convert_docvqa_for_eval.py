import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--dst", type=str)
args = parser.parse_args()

all_answers = []
for line_idx, line in enumerate(open(args.src)):
    res = json.loads(line)
    question_id = res['questionId']
    text = res['answer'].rstrip('.')
    all_answers.append({"questionId": question_id, "answer": text})

with open(args.dst, 'w') as f:
    json.dump(all_answers, f)

