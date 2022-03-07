#!/home/sf/data/linux/py_env/transformers_ds/bin/python
from sklearn.model_selection import train_test_split
import os, pickle, copy, json, random
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import argparse
###################################################################################################################
#
# LIBS
#
###################################################################################################################
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
      
def  slice_dict(_indat_, slice_idxs):
    result = {}
    for cnt, idx in enumerate(slice_idxs):
        result[cnt] = _indat_[idx]
    return result
###################################################################################################################
#
# MAIN
#
###################################################################################################################
def main(param_fname):
    print('\n' + '='*25)
   # Read parameters
    if os.path.exists(param_fname):
        with open(param_fname, 'rb') as f:
            params = json.load(f)
    else:
        print(f'No such file {param_fname}')
        return -1


    if params['save_folder'] == 'here':
        save_folder = os.getcwd()
    else:
        save_folder = params['save_folder']
        
    if os.path.exists(save_folder):
        print(f'Save folder\n\t{save_folder}')
    else:
        print(f'No such folder {save_folder}')
        return -1

    pkl2load = os.path.join(params['src_folder'], params['file_name'])
    print(f'Loading "{pkl2load}"')
    with open(pkl2load, 'rb' ) as f:
        indata = pickle.load(f)

    max_seq_len = params['max_seq_len']
    min_seq_len = params['min_seq_len']
    print(f"Min seq len: {min_seq_len}")
    print(f"Max seq len: {max_seq_len}")
    
    filt_indat = {}
    cnt = 0
    for idx in tqdm(indata):
        seq_len = len( word_tokenize(indata[idx]['text']) )
        if seq_len <= max_seq_len and seq_len >= min_seq_len:
            filt_indat[cnt] = {}
            for key in indata[idx]:
                filt_indat[cnt][key] = copy.copy(indata[idx][key])
            cnt += 1

    tdf = pd.DataFrame(filt_indat).T
    df = tdf[['kw_scores','kw', 'title','text']]
    df.columns = ['scores','kws', 'title','text']
    df = df.dropna()
    print(f"Filtered data constitute {(len(df) / len(indata.keys()) * 100):.1f}% of the total input")

    seed = params['seed']
    seed_everything(seed)

    test_size = float(params['train_val'])
    print(f'Splitting the data\n\ttrain: {1-test_size}\n\ttest:  {test_size}')
    # split train
    train_idx, test_idx =  train_test_split(list(list(df.index)), test_size=test_size)
    # prepare
    indata = df.to_dict()
    indat_kws = {}
    for i in indata['kws'] :
        indat_kws[i] = indata['kws'][i]
    indat_title = indata['title']
    indata_text = indata['text']

    # train data
    train_data_kws = slice_dict(indat_kws, train_idx)
    train_data_txt = slice_dict(indata_text, train_idx)
    train_data_title = slice_dict(indat_title, train_idx)
    
    train_data = {}
    train_data['keywords'] = train_data_kws
    train_data['text'] = train_data_txt
    train_data['title'] = train_data_title

    # test data
    test_data_kws = slice_dict(indat_kws, test_idx)
    test_data_txt = slice_dict(indata_text, test_idx)
    test_data_title = slice_dict(indat_title, test_idx)

    test_data = {}
    test_data['keywords'] = test_data_kws
    test_data['text'] = test_data_txt
    test_data['title'] = test_data_title


    # # Save
    train_fpath = os.path.join(save_folder,f'train_{min_seq_len}_{max_seq_len}.pkl')
    test_fpath = os.path.join(save_folder,f'test_{min_seq_len}_{max_seq_len}.pkl')

    print(f'Saving train data in\n\t{train_fpath}')
    print(f'Saving test data in\n\t{test_fpath}')

    with open(train_fpath, 'wb') as f:
        pickle.dump(train_data, f)
        
    with open(test_fpath, 'wb') as f:
        pickle.dump(test_data, f)
    
    print('Done')
###################################################################################################################
#
# CLI
#
###################################################################################################################
# CLI paratemrs parser
arg_parser = argparse.ArgumentParser(description='Split the prepared data for traning and limit token size')

arg_parser.add_argument('-path',
                       metavar='path',
                       type=str,
                       help='Path to the parameters json')

args = arg_parser.parse_args()

param_fname = args.path
if __name__=='__main__':
    main(param_fname)
