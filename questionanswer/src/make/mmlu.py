from datasets import load_dataset
import json
import pandas as pd
import pathlib
import string

def make_answer_list(arr):
    return '\n'.join([f'({l}) {s}' for l,s in zip(string.ascii_uppercase, arr)])

if __name__ == '__main__':
    
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    df = pd.DataFrame(ds['test'])

    outputs = []

    for config in df.category.unique():

        x = df[df.category == config]
        if len(x) > 100:
            x = x.sample(n=100, random_state=0)
        
        x['inputs'] = x['question'] + '\n\n' + x['options'].apply(make_answer_list)
        x['target'] = x['answer'].apply(lambda s: f'({s})')

        for i, row in x.iterrows():

            outputs.append({
                'inputs' : row.inputs,
                'subset' : config,
                'id' : len(outputs),
                'target' : row.target
            })
    
    pathlib.Path('data/').mkdir(exist_ok=True)
    with open('data/mmlu.jsonl', 'w') as out:
        for o in outputs:
            out.write(json.dumps(o) + '\n')

