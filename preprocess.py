import os, sys
import json
import pandas as pd

def run_tnews(dataset):
    TNEWS_PATH = './tnews_public'
    inpath = os.path.join(TNEWS_PATH, dataset + '.json')
    outpath = os.path.join(TNEWS_PATH, dataset + '.tsv')

    label_map = {'100': 0, '101': 1, '102': 2, '103': 3, '104': 4, '106': 5, '107': 6, '108': 7, '109': 8, '110': 9, '112': 10, '113': 11, '114': 12, '115': 13, '116': 14}

    with open(inpath, 'r') as fin:
        records = []
        for line in fin:
            record = json.loads(line.strip())
            records.append(record)
        df = pd.DataFrame.from_records(records)
        if dataset != 'test':
            df['label'] = df['label'].map(lambda x: label_map.get(x))
        else:
            df['label'] = 0
        df[['label','sentence']].to_csv(outpath, sep='\t', index=False, header=None)


if __name__ == '__main__':
    dataset = sys.argv[1]  # train or test
    dataname = 'tnews'

    run_funcs = { 'tnews': run_tnews
                 }
    run_func = run_funcs[dataname]
    run_func(dataset)
