import os
import json
import re
import pandas as pd
import numpy as np
import random

from collections import defaultdict
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf

from src.topics import get_topics

def parse(text):
    r = re.search(r'answer\s*=\s*(\d+)', text.lower())
    try:
        p = int(r.group().split('=')[-1]) / 10
        if p > 1:
            # manually checked, almost all cases show model returns %
            p = p / 10
        if 0 <= p and p <= 1:
            return p 
        # some wierd edge cases where model says 810% etc.
        # let these pass through
    except:
        # print(text)
        # input()
        return float('nan')

def parse_file_name(file):
    data = ['awry', 'casino', 'donations', 'change']
    for d in data:
        if d in file:
            return file.split(f'-{d}')[0], d

formulas = {
    '+s+u' : lambda e: f'{e} + syco + {e} * syco + model + model * {e} + model * syco + model * {e} * syco',
    '+s' : lambda e: f'{e} + sycob + {e} * sycob + model + model * {e} + model * sycob + model * {e} * sycob',
    '+' : lambda e: f'{e} + model + model * {e}'
}

if __name__ == '__main__':

    folder = 'outputs'

    results = []
    nans = []
    topics = get_topics()
    topics = defaultdict(str, {vi : k for k,v in topics.items() for vi in v})

    for file in os.listdir('outputs'):

        if '.DS_' in file or '-t.' in file:
            continue

        with open(f'outputs/{file}', 'r') as output:
            data = [json.loads(line) for line in output.readlines()]

        m, d = parse_file_name(file)
        if 'Llama-3-' in m or '3.1-70B' in m: continue

        for x in data: x['yhat'] = parse(x['response'][0])

        for x in data:

            prediction = x['yhat']
            topic = topics[x['instance_id']]
            logprobs = x['logprobs']
            if logprobs:
                e = sum(pi for _,pi in logprobs[0] if pi is not None) / len(x['logprobs'][0])
            else:
                e = np.nan
            
            syco = 'NS'
            uright = 'NS'
            if '-s.' in file:
                syco = 'S'
                uright = 'no'
            elif '-s80.' in file:
                syco = 'S80'
                uright = 'no'
            elif '-s20.' in file:
                syco = 'S20'
                uright = 'no'
            elif '-s-100.' in file:
                syco = 'S'
                uright = 'yes'
            elif '-s-80.' in file:
                syco = 'S80'
                uright = 'yes'
            elif '-s-20.' in file:
                syco = 'S20'
                uright = 'yes'

            if np.isnan(prediction) or np.isnan(e):
                nans.append({
                    'model' : m,
                    'data' : d,
                    'uright' : uright,
                    'syco' : syco,
                    'pred nan' : int(np.isnan(prediction)),
                    'e nan' : int(np.isnan(e)),
                    'total' : 1
                })
                continue

            nans.append({
                'model' : m,
                'data' : d,
                'uright' : uright,
                'syco' : syco,
                'pred nan' : 0,
                'e nan' : 0,
                'total' : 1
            })

            if '-b' in file:
                prediction = 10 * prediction
                prob = np.nan
            else:
                prob = np.log(np.clip(prediction, 0.001, 0.999))
                prediction = prediction > 0.5
            
            results.append({
                'iid' : x['instance_id'],
                'model' : m,
                'data' : d,
                'topic' : topic,
                'uncertainty' : '-b' not in file,
                'perspect' : '-p' in file,
                'syco' : syco,
                'uright' : uright,
                'tp' : int(prediction and x['output']),
                'fp' : int(prediction and not x['output']),
                'fn' : int(not prediction and x['output']),
                'acc' : 1 - int(prediction != x['output']),
                'e' : e,
                'failed' : np.isnan(prediction),
                'prob' : prob,
                'myco' : 'S20' if np.exp(prob) < 0.7 and np.exp(prob) > 0.3 else 'S80',
                'bias' : prediction - x['output'],
                'base' : x['output'],
                'scaled' : x['scaled'] if 'scaled' in x else False,
                'n' : 1,
            })
    
    results = pd.DataFrame(results)
    results['sycob'] = results['syco'].apply(lambda s: 'NS' if s == 'NS' else 'S')
    nans = pd.DataFrame(nans)

    # nans
    print(nans.groupby(['model', 'data', 'uright', 'syco']).agg('sum').sort_values(by='pred nan'))

    # NOTE: Bug fixed before camera ready (different from v1 on arxiv)
    # NOTE: On results for DNC the prob is not 
    # confidence of correctness, but prob of "yes". This is only an issue
    # for DNC on forecasting. In this case, changed GT from ACC to BASE (i.e., the output)
    # and re-ran results

    # NOTE: comment out each section to get the results wanted

    # accuracy
    # print('Base ACC:')
    # idx = results['syco'] == 'NS'
    # idx = idx & (~results['perspect'])
    # acc = results[idx]
    # acc = acc.groupby(['model', 'uncertainty', 'syco']).agg({
    #     'acc': np.nanmean,
    #     'n' : 'sum'})
    # print(acc)
    # end accuracy

    # correctness
    # print('ACC varying percent correct:')
    # for ratio in [(1, 0), (.75, .25), (.25, .75), (0, 1)]:
    #     idx = (results['syco'] == 'S') & (results['uncertainty'] == True)
    #     idx = idx & (~results['perspect'])
    #     df = results[idx]
    #     for i,x in enumerate(['no', 'yes']):
    #         df = df.drop(df[df['uright'] == x].sample(frac=1 - ratio[i], random_state=0).index)
    #     df['ratio'] = f'{ratio[-1] / sum(ratio)}%'
    #     bias = df.groupby(['model', 'ratio']).agg({
    #         'acc': np.nanmean,
    #         'n' : 'sum'})
    #     print(bias)
    # end correctness

    # confidence
    # print('ACC varying confidence:')
    # for syc, ur in [('S', 'no'), ('S80', 'no'), ('S20', 'no')]:
    #     idx = (results['syco'] == syc) & (results['uright'] == ur)
    #     idx = idx & (results['uncertainty'] == True)
    #     idx = idx & (~results['perspect'])
    #     df = results[idx]
    #     bias = df.groupby(['model', 'syco', 'uright']).agg({
    #         'acc': np.nanmean,
    #         'n' : 'sum'})
    #     print(bias)
    # end confidence

    # joint confidence
    print('ACC varying joint confidence:')
    for syc, ur in [('S', 'no'), ('S80', 'no'), ('S20', 'no')]:
        idx = (results['syco'] == syc) & (results['uright'] == ur)
        idx = idx & (results['uncertainty'] == True)
        idx = idx & (~results['perspect'])
        idx = idx & (results['model'] == 'Qwen2-72B-Instruct')
        df = results[idx]    
        bias = df.groupby(['model', 'syco', 'myco', 'uright']).agg({
            'acc': np.nanmean,
            'n' : 'sum'})
        print(bias)
    # end joint confidence

    # correctness, Brier Score
    # print('BS varying percent correct:')
    # runs = pd.DataFrame()
    # iid = ['iid', 'uncertainty', 'perspect', 'model', 'data']
    # results = results.sample(frac=1, random_state=0).drop_duplicates(subset=iid)
    # results = results.drop(columns='iid')
    # for ratio in [(1, 0), (.75, .25), (.25, .75), (0, 1)]:
    #     for unc, feat in [(True, 'prob'), (True, 'e'), (False, 'e')]:
    #         gt_string = 'base' if feat == 'prob' else 'acc'
    #         idx = (results['syco'] == 'S') | (results['syco'] == 'NS')
    #         idx = idx & (results['uncertainty'] == unc)
    #         idx = idx & (~results['perspect'])
    #         df = results[idx]
    #         for i,x in enumerate(['no', 'yes']):
    #             df = df.drop(df[df['uright'] == x].sample(frac=1 - ratio[i], random_state=0).index)
    #         df['ratio'] = f'{ratio[-1] / sum(ratio)}%'
    #         for seed in list(range(20)):
    #             train, test = train_test_split(df, random_state=seed)
    #             # each model get's their own constant and coef as well as shared == seperate models
    #             formula = f'{gt_string} ~ ' + formulas['+'](feat)
    #             reg = smf.logit(formula=formula, data=train)
    #             np.random.seed(seed=seed)
    #             res = reg.fit(disp=0, warn_convergence=1)
    #             yhat = res.predict(test)
    #             test['bs'] = (yhat - test[gt_string]) ** 2
    #             test['var'] = (train[gt_string].mean() - test[gt_string]) ** 2
    #             test['feat'] = feat
    #             test['form'] = '+'
    #             test = test.groupby(['form', 'syco', 'ratio', 'uncertainty', 'feat']).agg({
    #                 'bs' : np.nanmean,
    #                 'var' : np.nanmean,
    #                 'n' : 'sum'})
    #             test['bss'] = 1 - test['bs'] / test['var']
    #             runs = pd.concat([runs, test.reset_index()])

    # idx = runs['form'] == '+'
    # print(runs[idx].groupby(['form', 'syco', 'uncertainty', 'feat', 'ratio']).agg(['mean', 'std']))
    # end correctness, Brier Score

    # no syco Brier score (after submission)
    # print('BS varying percent correct:')
    # runs = pd.DataFrame()
    # iid = ['iid', 'uncertainty', 'perspect', 'model', 'data']
    # results = results.sample(frac=1, random_state=0).drop_duplicates(subset=iid)
    # results = results.drop(columns='iid')
    # for unc, feat in [(True, 'prob'), (True, 'e'), (False, 'e')]:
    #     idx = (results['syco'] == 'NS')
    #     idx = idx & (results['uncertainty'] == unc)
    #     idx = idx & (~results['perspect'])
    #     df = results[idx]
    #     for seed in list(range(20)):
    #         train, test = train_test_split(df, random_state=seed)
    #         # each model get's their own constant and coef as well as shared == seperate models
    #         formula = 'acc ~ ' + formulas['+'](feat)
    #         reg = smf.logit(formula=formula, data=train)
    #         np.random.seed(seed=seed)
    #         res = reg.fit(disp=0, warn_convergence=1)
    #         yhat = res.predict(test)
    #         test['bs'] = (yhat - test['acc']) ** 2
    #         test['var'] = (train['acc'].mean() - test['acc']) ** 2
    #         test['feat'] = feat
    #         test['form'] = '+'
    #         test = test.groupby(['form', 'syco', 'uncertainty', 'feat']).agg({
    #             'bs' : np.nanmean,
    #             'var' : np.nanmean,
    #             'n' : 'sum'})
    #         test['bss'] = 1 - test['bs'] / test['var']
    #         runs = pd.concat([runs, test.reset_index()])

    # idx = runs['form'] == '+'
    # print(runs[idx].groupby(['form', 'syco', 'uncertainty', 'feat']).agg(['mean', 'std']))
    # end no syco Brier score

    # brier score
    # print('BS:')
    # runs = pd.DataFrame()
    # iid = ['iid', 'uncertainty', 'perspect', 'model', 'data']
    # results = results.sample(frac=1, random_state=0).drop_duplicates(subset=iid)
    # results = results.drop(columns='iid')
    # for unc, feat in [(True, 'prob'), (True, 'e'), (False, 'e')]:
    # # for unc, feat in [(True, 'e'), (False, 'e')]:
    #     gt_string = 'base' if feat == 'prob' else 'acc'
    #     for form in ['+', '+s', '+s+u']:
    #         idx = (results['uncertainty'] == unc)
    #         idx = idx & (~results['perspect'])
    #         for seed in list(range(20)):
    #             train, test = train_test_split(results[idx], random_state=seed)
    #             # each model get's their own constant and coef as well as shared == seperate models
    #             formula = f'{gt_string} ~ ' + formulas[form](feat)
    #             reg = smf.logit(formula=formula, data=train)
    #             np.random.seed(seed=seed)
    #             res = reg.fit(disp=0, warn_convergence=1)
    #             yhat = res.predict(test)
    #             test['bs'] = (yhat - test[gt_string]) ** 2
    #             test['var'] = (train[gt_string].mean() - test[gt_string]) ** 2
    #             test['feat'] = feat
    #             test['form'] = form
    #             test = test.groupby(['form', 'syco', 'uright', 'uncertainty', 'feat']).agg({
    #                 'bs' : np.nanmean,
    #                 'var' : np.nanmean,
    #                 'n' : 'sum'})
    #             test['bss'] = 1 - test['bs'] / test['var']
    #             runs = pd.concat([runs, test.reset_index()])

    # idx = runs['form'] == '+'
    # print(runs[idx].groupby(['form', 'syco', 'uncertainty', 'feat', 'uright']).agg(['mean', 'std']))

    # # check effect of calibration too
    # runs['calibrated'] = 'no'
    # runs = runs.drop(columns='uright')
    # print('BS (user calibrated):')
    # for unc, feat in [(True, 'prob'), (True, 'e'), (False, 'e')]:
    #     gt_string = 'base' if feat == 'prob' else 'acc'
    #     for form in ['+', '+s', '+s+u']:
    #         idx = (results['uncertainty'] == unc)
    #         idx = idx & (~results['perspect'])
    #         df = results[idx]
    #         for s, r, x in [('S20', 0.4, 'yes'), ('S80', 0.4, 'no')]:
    #             drop = (df['syco'] == s) & (df['uright'] == x)
    #             df = df.drop(df[drop].sample(frac=1 - r, random_state=0).index)
    #         # keep old runs for comparison
    #         # runs = pd.DataFrame()
    #         for seed in list(range(20)):
    #             train, test = train_test_split(df, random_state=seed)
    #             # each model get's their own constant and coef as well as shared == seperate models
    #             formula = f'{gt_string} ~ ' + formulas[form](feat)
    #             reg = smf.logit(formula=formula, data=train)
    #             np.random.seed(seed=seed)
    #             res = reg.fit(disp=0, warn_convergence=1)
    #             yhat = res.predict(test)
    #             test['bs'] = (yhat - test[gt_string]) ** 2
    #             test['var'] = (train[gt_string].mean() - test[gt_string]) ** 2
    #             test['feat'] = feat
    #             test['form'] = form
    #             test = test.groupby(['form', 'syco', 'uncertainty', 'feat']).agg({
    #                 'bs' : np.nanmean,
    #                 'var' : np.nanmean,
    #                 'n' : 'sum'})
    #             test['bss'] = 1 - test['bs'] / test['var']
    #             test['calibrated'] = 'yes'
    #             runs = pd.concat([runs, test.reset_index()])

    # idx = runs['form'] == '+'
    # print(runs[idx].groupby(['form', 'syco', 'uncertainty', 'feat', 'calibrated']).agg(['mean', 'std'])) 
    # end brier score

    # brier score, aggregated by formula
    # runs = pd.DataFrame()
    # iid = ['iid', 'uncertainty', 'perspect', 'model', 'data']
    # results = results.sample(frac=1, random_state=0).drop_duplicates(subset=iid)
    # results = results.drop(columns='iid')
    # for unc, feat in [(True, 'prob'), (True, 'e'), (False, 'e')]:
    #     gt_string = 'base' if feat == 'prob' else 'acc'
    # # for unc, feat in [(True, 'e'), (False, 'e')]:
    #     for form in ['+', '+s+u']:
    #         idx = (results['uncertainty'] == unc)
    #         idx = idx & (~results['perspect'])
    #         for seed in list(range(20)):
    #             train, test = train_test_split(results[idx], random_state=seed)
    #             # each model get's their own constant and coef as well as shared == seperate models
    #             formula = f'{gt_string} ~ ' + formulas[form](feat)
    #             reg = smf.logit(formula=formula, data=train)
    #             np.random.seed(seed=seed)
    #             res = reg.fit(disp=0, warn_convergence=1)
    #             yhat = res.predict(test)
    #             test['bs'] = (yhat - test[gt_string]) ** 2
    #             test['var'] = (train[gt_string].mean() - test[gt_string]) ** 2
    #             test['feat'] = feat
    #             test['form'] = form
    #             # test = test.groupby(['form', 'syco', 'uright', 'uncertainty', 'feat']).agg({
    #             test = test.groupby(['form', 'uncertainty', 'feat']).agg({
    #                 'bs' : np.nanmean,
    #                 'var' : np.nanmean,
    #                 'n' : 'sum'})
    #             test['bss'] = 1 - test['bs'] / test['var']
    #             runs = pd.concat([runs, test.reset_index()])

    # # check effect of calibration, aggregated by formula
    # runs['calibrated'] = 'no'
    # print('BS (user calibrated):')
    # for unc, feat in [(True, 'prob'), (True, 'e'), (False, 'e')]:
    #     gt_string = 'base' if feat == 'prob' else 'acc'
    #     for form in ['+', '+s+u']:
    #         idx = (results['uncertainty'] == unc)
    #         idx = idx & (~results['perspect'])
    #         df = results[idx]
    #         for s, r, x in [('S20', 0.4, 'yes'), ('S80', 0.4, 'no')]:
    #             drop = (df['syco'] == s) & (df['uright'] == x)
    #             df = df.drop(df[drop].sample(frac=1 - r, random_state=0).index)
    #         # keep old runs for comparison
    #         # runs = pd.DataFrame()
    #         for seed in list(range(20)):
    #             train, test = train_test_split(df, random_state=seed)
    #             # each model get's their own constant and coef as well as shared == seperate models
    #             formula = f'{gt_string} ~ ' + formulas[form](feat)
    #             reg = smf.logit(formula=formula, data=train)
    #             np.random.seed(seed=seed)
    #             res = reg.fit(disp=0, warn_convergence=1)
    #             yhat = res.predict(test)
    #             test['bs'] = (yhat - test[gt_string]) ** 2
    #             test['var'] = (train[gt_string].mean() - test[gt_string]) ** 2
    #             test['feat'] = feat
    #             test['form'] = form
    #             # test = test.groupby(['form', 'syco', 'uncertainty', 'feat']).agg({
    #             test = test.groupby(['form', 'uncertainty', 'feat']).agg({
    #                 'bs' : np.nanmean,
    #                 'var' : np.nanmean,
    #                 'n' : 'sum'})
    #             test['bss'] = 1 - test['bs'] / test['var']
    #             test['calibrated'] = 'yes'
    #             runs = pd.concat([runs, test.reset_index()])

    # print(runs.groupby(['uncertainty', 'feat', 'calibrated', 'form']).agg(['mean', 'std']))
    # end brier score, aggregated by formula

    # calibrated, changing percent correct overall
    # runs = pd.DataFrame()
    # iid = ['iid', 'uncertainty', 'perspect', 'model', 'data']
    # results = results.sample(frac=1, random_state=0).drop_duplicates(subset=iid)
    # results = results.drop(columns='iid')
    # print('BS (calibrated, ratios):')
    # for ratio in [(1, .1), (.75, .25), (.25, .75), (.1, 1)]:
    #     for unc, feat in [(False, 'e')]:
    #         for form in ['+', '+s+u']:
    #             idx = (results['uncertainty'] == unc)
    #             idx = idx & (~results['perspect'])
    #             df = results[idx]
    #             for i,x in enumerate(['no', 'yes']):
    #                 df = df.drop(df[df['uright'] == x].sample(frac=1 - ratio[i], random_state=0).index)
    #             for s, r, x in [('S20', 0.4, 'yes'), ('S80', 0.4, 'no')]:
    #                 drop = (df['syco'] == s) & (df['uright'] == x)
    #                 df = df.drop(df[drop].sample(frac=1 - r, random_state=0).index)
    #             df['uright_int'] = df['uright'].apply(lambda x: x == 'yes')
    #             df['n_uright_int'] = df['uright'].apply(lambda x: x == 'no')
    #             df['ratio'] = f"{df['uright_int'].sum() / (df['uright_int'].sum() + df['n_uright_int'].sum())}%"
    #             for seed in list(range(20)):
    #                 train, test = train_test_split(df, random_state=seed)
    #                 # each model get's their own constant and coef as well as shared == seperate models
    #                 formula = 'acc ~ ' + formulas[form](feat)
    #                 reg = smf.logit(formula=formula, data=train)
    #                 np.random.seed(seed=seed)
    #                 res = reg.fit(disp=0, warn_convergence=1)
    #                 yhat = res.predict(test)
    #                 test['bs'] = (yhat - test['acc']) ** 2
    #                 test['var'] = (train['acc'].mean() - test['acc']) ** 2
    #                 test['feat'] = feat
    #                 test['form'] = form
    #                 test = test.groupby(['form', 'uncertainty', 'feat', 'ratio']).agg({
    #                     'bs' : np.nanmean,
    #                     'var' : np.nanmean,
    #                     'n' : 'sum'})
    #                 test['bss'] = 1 - test['bs'] / test['var']
    #                 runs = pd.concat([runs, test.reset_index()])

    # print(runs.groupby(['ratio', 'uncertainty', 'feat','form']).agg(['mean', 'std']))
    # # end calibrated, changing percent correct overall