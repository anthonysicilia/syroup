import json
import pandas as pd
import pathlib

def make_prompt(inputs, start_mc, before_mc=None):
    prompt = '\n'.join(inputs[1:start_mc])
    if before_mc is not None:
        prompt += before_mc
    for letter, inpt in zip(['(A)', '(B)', '(C)', '(D)', '(E)'], inputs[start_mc:]):
        prompt += f'\n{letter} {inpt}'
    return prompt

def get_target(target, zero_index=True):
    answers = ['(A)', '(B)', '(C)', '(D)', '(E)']
    if zero_index:
        return answers[int(target)]
    else:
        return answers[int(target) - 1]

def parse(inputs, target):
    inputs = [x.split('</')[0].split('>')[-1] for x in inputs.split('\n')]

    if inputs[0].startswith('[anli]:'):
        assert len(inputs) == 5
        return (
            'anli', 
            make_prompt(inputs, start_mc=3,
                before_mc='\n\nPick the best hypothesis for the given premise.'),
            get_target(target, zero_index=False)
        )
    elif inputs[0].startswith('[cosmosqa]:'):
        assert len(inputs) == 7
        return (
            'cosmosqa', 
            make_prompt(inputs, start_mc=3),
            get_target(target, zero_index=True)
        )
    elif inputs[0].startswith('[hellaswag]:'):    
        assert len(inputs) == 6
        return (
            'hellaswag', 
            make_prompt(inputs, start_mc=2, before_mc='...\n\nPick the best ending to the given context.'),
            get_target(target, zero_index=True)
        )
    elif inputs[0].startswith('[physicaliqa]:'):
        assert len(inputs) == 4
        return ('physicaliqa', 
            make_prompt(inputs, start_mc=2),
            get_target(target, zero_index=True)
        )
    elif inputs[0].startswith('[socialiqa]:'):
        assert len(inputs) == 6
        return (
            'socialiqa', 
            make_prompt(inputs, start_mc=3), 
            get_target(target, zero_index=False)
        )
    elif inputs[0].startswith('[winogrande]:'):
        assert len(inputs) == 4
        return (
            'winogrande', 
            make_prompt(inputs, start_mc=2,
                before_mc='\n\nFill in _ with the best option.'),
            get_target(target, zero_index=False)
        )
    
if __name__ == '__main__':

    paths = [
        'rainbow/anli/validation.anli.csv',
        'rainbow/cosmosqa/validation.cosmosqa.csv',
        'rainbow/hellaswag/validation.hellaswag.csv',
        'rainbow/physicaliqa/validation.physicaliqa.csv',
        'rainbow/socialiqa/validation.socialiqa.csv',
        'rainbow/winogrande/validation.winogrande.csv'
    ]

    outputs = []

    for p in paths:

        df = pd.read_csv(p)
        if len(df) > 500:
            df = df.sample(n=500, random_state=0)

        for i, row in df.iterrows():
            dataset, inputs, target = parse(row.inputs, row.targets)
            outputs.append({
                'inputs' : inputs,
                'subset' : dataset,
                'id' : len(outputs),
                'target' : target
            })
    
    pathlib.Path('data/').mkdir(exist_ok=True)
    with open('data/rainbow.jsonl', 'w') as out:
        for o in outputs:
            out.write(json.dumps(o) + '\n')
        

