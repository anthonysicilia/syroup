import json

if __name__ == '__main__':

    counters = {}

    for config in ['bbh', 'mmlu', 'rainbow']:

        with open(f'data/{config}.jsonl', 'r') as data:
            data = [json.loads(line) for line in data]
        
        for subset in set([x['subset'] for x in data]):
            cset = set([x['target'] for x in data if x['subset'] == subset])
            if subset in counters:
                raise ValueError('Repeat')
            counters[subset] = list(cset)
    
    with open('counters.json', 'w') as out:
        out.write(json.dumps(counters))
