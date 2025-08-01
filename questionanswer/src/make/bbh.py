from datasets import load_dataset
import json
import pandas as pd
import pathlib

all_configs = [
    'tracking_shuffled_objects_seven_objects',
    'salient_translation_error_detection',
    'tracking_shuffled_objects_three_objects',
    'geometric_shapes',
    'object_counting',
    # 'word_sorting',
    'logical_deduction_five_objects',
    'hyperbaton',
    'sports_understanding',
    'logical_deduction_seven_objects',
    'multistep_arithmetic_two',
    'ruin_names',
    'causal_judgement',
    'logical_deduction_three_objects',
    'formal_fallacies',
    'snarks',
    'boolean_expressions',
    'reasoning_about_colored_objects',
    # 'dyck_languages',
    'navigate',
    'disambiguation_qa',
    'temporal_sequences',
    'web_of_lies',
    'tracking_shuffled_objects_five_objects',
    'penguins_in_a_table',
    'movie_recommendation',
    'date_understanding'
]

if __name__ == '__main__':

    outputs = []

    for config in all_configs:

        ds = load_dataset("maveriq/bigbenchhard", config)
        df = pd.DataFrame(ds['train'])

        if len(df) > 100:
            df = df.sample(n=100, random_state=0)

        for i, row in df.iterrows():

            outputs.append({
                'inputs' : row.input,
                'subset' : config,
                'id' : len(outputs),
                'target' : row.target
            })
    
    pathlib.Path('data/').mkdir(exist_ok=True)
    with open('data/bbh.jsonl', 'w') as out:
        for o in outputs:
            out.write(json.dumps(o) + '\n')