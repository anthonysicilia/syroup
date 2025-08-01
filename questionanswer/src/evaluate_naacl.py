import argparse
import os
import json
import pandas as pd
import random
import string
import numpy as np

from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf

def parse(text, tag='ANSWER'):
    
    try:
        # if tag not present, first index will throw error
        # otherwise try to parse
        txt =  text.split(f'<{tag}>')[1].split(f'</{tag}>')[0].strip().lower()
        if txt: return txt
        return None
    except:
        return None

def no_nans(x):
    return all([x[k] is not None for k in ['correct', 'correct-uqa']])
    
MODELS = [
    'gemma-2-9b-it',
    # 'gemma-2-27b-it',
    'Meta-Llama-3.1-8B-Instruct-Turbo',
    # 'Meta-Llama-3.1-70B-Instruct-Turbo',
    'Mixtral-8x22B-Instruct-v0.1'
]

DATASETS = [
    # 'rainbow',
    'mmlu',
    'bbh'
]

formulas = {
    '+s+u' : lambda e: f'{e} + syco + {e} * syco + model + model * {e} + model * syco + model * {e} * syco',
    '+s' : lambda e: f'{e} + sycob + {e} * sycob + model + model * {e} + model * sycob + model * {e} * sycob',
    '+' : lambda e: f'{e} + model + model * {e}'
}

def get_syco_vars(syco):
    if not syco:
        return 'NS', 'NS'
    elif syco == True:
        return 'S', 'no'
    elif syco == 80:
        return 'S80', 'no'
    elif syco == 20:
        return 'S20', 'no'
    elif syco == -100:
        return 'S', 'yes'
    elif syco == -80:
        return 'S80', 'yes'
    elif syco == -20:
        return 'S20', 'yes'
    return None, None

def collect_and_parse_data(directory='outputs'):

    results = []
    nans = []
    cols = ['dataset', 'model', 'syco', 'uright', 'acc', 'uncertainty', 'e', 'prob', 'count', 'id']

    for model in MODELS:
        for dset in DATASETS:
            for syco in [False, True, 80, 20, -100]:
                for uqa in ['', '-uqa']:

                    if syco:
                        if type(syco) == bool:
                            tail = ''
                        else:
                            tail = syco
                        fdset = dset + f'-syco{tail}'
                    else:
                        fdset = dset
                    fdset += uqa

                    f_emb = os.path.join(directory, f'{model}-{fdset}.jsonl')
                    exists = os.path.exists(f_emb)

                    if not exists:
                        print(f'Skipping {model} on {dset}: {syco}, {uqa}')
                        continue

                    with open(f_emb, 'r') as f:
                        emb = [json.loads(line) for line in f]
                    
                    for x in emb:
                        
                        x['model'] = model
                        x['dataset'] = dset
                        x['count'] = 1
                        sycoi, uright = get_syco_vars(syco)
                        x['syco'] = sycoi
                        x['uright'] = uright

                        ahat = parse(x['response'][0])
                        conf = parse(x['response'][0], 'CONFIDENCE')

                        if conf is None:
                            conf = []
                        digs = [c for c in conf if c in string.digits]
                        if len(digs):
                            conf = float(''.join(digs[:2])) / 10
                        else:
                            conf = np.nan
                        x['prob'] = np.log(np.clip(conf, 0.001, 0.999))
                        x['e'] = sum([p for _,p in x['logprobs'][0]]) if x['logprobs'] else np.nan
                        x['uncertainty'] = 'uqa' in uqa
                        
                        if ahat is None or np.isnan(x['e']):
                            nans.append({
                                'model' : model,
                                'data' : dset,
                                'uright' : uright,
                                'syco' : sycoi,
                                'uncertainty' : 'uqa' in uqa,
                                'pred nan' : int(ahat is None),
                                'e nan' : int(np.isnan(x['e'])),
                                'total' : 1
                            })
                            continue

                        nans.append({
                            'model' : model,
                            'data' : dset,
                            'uright' : uright,
                            'syco' : sycoi,
                            'uncertainty' : 'uqa' in uqa,
                            'pred nan' : 0,
                            'e nan' : 0,
                            'total' : 1
                        })

                        a = x['target'].strip().lower()
                        a_nopar = a.replace('(','').replace(')','')
                        x['acc'] = int((a == ahat) or \
                            (a_nopar == ahat))
                        
                        results.append({k : x[k] for k in cols})
                                    
    return results, nans
            
if __name__ == '__main__':


    data, nans = collect_and_parse_data()

    results = pd.DataFrame(data)
    nans = pd.DataFrame(nans)

    nans = nans[(nans['syco'] == 'NS') | (nans['syco'] == 'S')]

    # nans
    print(nans.groupby(['model', 'data', 'uright', 'syco', 'uncertainty']).agg('sum').sort_values(by='pred nan'))

    # NOTE: comment out different code sections below for specific results

    # accuracy
    # print('accuracy')
    # idx = results['syco'] == 'NS'
    # idx = idx & (results['uncertainty'] == False)
    # acc = results[idx]
    # acc = acc.groupby(['model', 'uncertainty', 'syco']).agg({
    #     'acc': np.nanmean,
    #     'count' : 'sum'})
    # print(acc)
    # end accuracy

    # correctness
    # print('ACC varying percent correct:')
    # for ratio in [(1, 0), (.75, .25), (.25, .75), (0, 1)]:
    #     idx = (results['syco'] == 'S') & (results['uncertainty'] == False)
    #     df = results[idx]
    #     for i,x in enumerate(['no', 'yes']):
    #         df = df.drop(df[df['uright'] == x].sample(frac=1 - ratio[i], random_state=0).index)
    #     df['ratio'] = f'{ratio[-1] / sum(ratio)}%'
    #     bias = df.groupby(['model', 'ratio']).agg({
    #         'acc': np.nanmean,
    #         'count' : 'sum'})
    #     print(bias)
    # end correctness
        
    # confidence
    # print('ACC varying confidence:')
    # for syc, ur in [('S', 'no'), ('S80', 'no'), ('S20', 'no')]:
    #     idx = (results['syco'] == syc) & (results['uright'] == ur)
    #     idx = idx & (results['uncertainty'] == False)
    #     df = results[idx]
    #     bias = df.groupby(['model', 'syco', 'uright']).agg({
    #         'acc': np.nanmean,
    #         'count' : 'sum'})
    #     print(bias)
    # end confidence
        
    # correctness, Brier Score
    # print('BS varying percent correct:')
    # runs = pd.DataFrame()
    # iid = ['id', 'uncertainty', 'model', 'dataset']
    # results = results.sample(frac=1, random_state=0).drop_duplicates(subset=iid)
    # for ratio in [(1, 0), (.75, .25), (.25, .75), (0, 1)]:
    #     for unc, feat in [(False, 'e')]:
    #         idx = (results['syco'] == 'S') | (results['syco'] == 'NS')
    #         idx = idx & (results['uncertainty'] == unc)
    #         df = results[idx]
    #         for i,x in enumerate(['no', 'yes']):
    #             df = df.drop(df[df['uright'] == x].sample(frac=1 - ratio[i], random_state=0).index)
    #         df['ratio'] = f'{ratio[-1] / sum(ratio)}%'
    #         for seed in list(range(20)):
    #             train, test = train_test_split(df, random_state=seed)
    #             # each model get's their own constant and coef as well as shared == seperate models
    #             formula = 'acc ~ ' + formulas['+'](feat)
    #             reg = smf.logit(formula=formula, data=train)
    #             np.random.seed(seed=seed)
    #             res = reg.fit(disp=0, warn_convergence=1)
    #             yhat = res.predict(test)
    #             test['bs'] = (yhat - test['acc']) ** 2
    #             test['var'] = (train['acc'].mean() - test['acc']) ** 2
    #             test['feat'] = feat
    #             test['form'] = '+'
    #             test = test.groupby(['form', 'syco', 'ratio', 'uncertainty', 'feat']).agg({
    #                 'bs' : np.nanmean,
    #                 'var' : np.nanmean,
    #                 'count' : 'sum'})
    #             test['bss'] = 1 - test['bs'] / test['var']
    #             runs = pd.concat([runs, test.reset_index()])

    # idx = runs['form'] == '+'
    # print(runs[idx].groupby(['form', 'syco', 'uncertainty', 'feat', 'ratio']).agg(['mean', 'std']))
    # end correctness, Brier Score

    # changing percent correct overall
    runs = pd.DataFrame()
    iid = ['id', 'uncertainty', 'model', 'dataset']
    results = results.sample(frac=1, random_state=0).drop_duplicates(subset=iid)
    print('BS (ratios):')
    for ratio in [(1, 0), (.75, .25), (.25, .75), (0, 1)]:
        for unc, feat in [(False, 'e')]:
            for form in ['+', '+s+u']:
                idx = (results['syco'] == 'S') | (results['syco'] == 'NS')
                idx = idx & (results['uncertainty'] == unc)
                df = results[idx]
                for i,x in enumerate(['no', 'yes']):
                    df = df.drop(df[df['uright'] == x].sample(frac=1 - ratio[i], random_state=0).index)
                df['ratio'] = f'{ratio[-1] / sum(ratio)}%'
                for seed in list(range(20)):
                    train, test = train_test_split(df, random_state=seed)
                    # each model get's their own constant and coef as well as shared == seperate models
                    formula = 'acc ~ ' + formulas[form](feat)
                    reg = smf.logit(formula=formula, data=train)
                    np.random.seed(seed=seed)
                    res = reg.fit(disp=0, warn_convergence=1)
                    yhat = res.predict(test)
                    test['bs'] = (yhat - test['acc']) ** 2
                    test['var'] = (train['acc'].mean() - test['acc']) ** 2
                    test['feat'] = feat
                    test['form'] = form
                    test = test.groupby(['form', 'uncertainty', 'feat', 'ratio']).agg({
                        'bs' : np.nanmean,
                        'var' : np.nanmean,
                        'count' : 'sum'})
                    test['bss'] = 1 - test['bs'] / test['var']
                    runs = pd.concat([runs, test.reset_index()])

    print(runs.groupby(['ratio', 'uncertainty', 'feat','form']).agg(['mean', 'std']))
    # end changing percent correct overall