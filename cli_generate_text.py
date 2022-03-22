from utils import NLP_Model, seed_everything

import os, json, random, pickle
import numpy as np
import pandas as pd
import datetime, torch
#######################################################################################################
#
# FUNCS
#
####################################################################################################### 
def save_to_csv(src_fld: str, fld_texts: str, file_name_pattern: str, df: pd.DataFrame):
    """
    Saves the Pandas DF to a csv file. 
    It appends to it if there is such file name!
    """
    _dst_path = os.path.join(src_fld, fld_texts)
    try:
        os.makedirs(_dst_path)
    except Exception as e:
        print(f"\t{e}")
    _dst_f = os.path.join(_dst_path, file_name_pattern)

    with open(_dst_f, 'a') as f:
        df.to_csv(f, mode='a', header=f.tell()==0, sep = '\t', index = False)
#######################################################################################################
#
# main()
#
#######################################################################################################  
def main(cfg_name):
    # read cfg
    with open(cfg_name,'rb') as f:
        config = json.load(f)

    # parameters 
    model_name_hf = config['config']['model_name_hf']
    model_path = config['config']['model_path']
    min_length = int(config['config']['min_length'])
    max_length = int(config['config']['max_length'])
    output_fld = config['config']['output_fld']
    output_core_name = config['config']['output_core_name']
    src_fld = config['config']['workdir']
    cache_dir = os.path.join(config['config']['cache_dir'], model_name_hf)

    device = config['config']['device']
    if device == '':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device is not specified. Roll back to available device: {device}')

    top_k = float(config['config']['top_k'])
    if top_k < 0:
        top_k = None

    top_p = float(config['config']['top_p'])
    if top_p < 0:
        top_p = None

    temperature = float(config['config']['temperature'])
    if temperature < 0:
        temperature = None

    repetition_penalty = float(config['config']['repetition_penalty'])
    if repetition_penalty < 0:
        repetition_penalty = None

    length_penalty = float(config['config']['length_penalty'])
    if length_penalty < 0:
        length_penalty = None

    num_return_sequences = int(config['config']['num_return_sequences'])
    if num_return_sequences < 0:
        num_return_sequences = None

    no_repeat_ngram_size = int(config['config']['no_repeat_ngram_size']  )
    if no_repeat_ngram_size < 0:
        no_repeat_ngram_size = None

    seed = int(config['config']['seed'])  
    if seed < 0:
        seed = 42

    SPECIAL_TOKENS  = {"bos_token": "<|BOS|>",
                       "eos_token": "<|EOS|>",
                       "unk_token": "<|UNK|>",                    
                       "pad_token": "<|PAD|>",
                       "sep_token": "<|SEP|>"}
                            
    # seed
    seed_everything(seed)
    device = torch.device('cuda') 
    dtstr = datetime.datetime.now().strftime('%Y-%m-%d') #%H-%M
    
    # set up the model
    nlp_model = NLP_Model(model = model_name_hf, 
                          dtype = torch.half, 
                          cache_dir = cache_dir, 
                          special_tokens = SPECIAL_TOKENS, 
                          load_model_path = model_path)
    model = nlp_model.model.half().to(device) 

    # Generation
    cnt = 0
    gen_txt = {}
    for key in config['prompts']:

        title = config['prompts'][key]['title']
        kws = config['prompts'][key]['keywords']
        prompt = NLP_Model.gen_prompt(title, kws, SPECIAL_TOKENS)
        model.eval();

        print(f'Generating for\n\t"{title}"')
        precursor = torch.tensor(nlp_model.tokenizer.encode(prompt)).unsqueeze(0)
        device = torch.device(device)
        precursor = precursor.to(device)

        sample_outputs = model.generate(
                                    precursor, 
                                    do_sample=True,   
                                    min_length = min_length, 
                                    max_length = max_length,
                                    top_k = top_k,                                 
                                    top_p = top_p,        
                                    temperature = temperature,
                                    repetition_penalty =repetition_penalty,
                                    length_penalty = length_penalty, 
                                    no_repeat_ngram_size = no_repeat_ngram_size,
                                    num_return_sequences = num_return_sequences
                                    )

        for i, sample_output in enumerate(sample_outputs):
            text = nlp_model.tokenizer.decode(sample_output, skip_special_tokens=True)
            bare_txt = text.split(title)[-1][len(kws):]
            gen_txt[cnt] = {
                'text' : bare_txt,
                'title': title,
                'kws'  : kws,        
            }
            cnt += 1


    # save
    print(f'Saving results in\n\t{os.path.join(src_fld, output_fld)}')
    df_get_txt = pd.DataFrame(gen_txt).T
    if output_core_name != '':
        file_name_pattern = f"{model_name_hf}_sampling_{output_core_name}.csv"
    else:
        file_name_pattern = f"{model_name_hf}_sampling.csv"

    save_to_csv(src_fld, output_fld, file_name_pattern, df_get_txt)
    print('Done')
    return 0
#######################################################################################################
#
# launch main()
#
####################################################################################################### 
if __name__ == '__main__':
    import argparse
    
    arg_parser = argparse.ArgumentParser(description='CLI Generate text based on conditions')
    arg_parser.add_argument('-cfg',
                           metavar='cfg',
                           type=str,
                           help='config file json')
    args = arg_parser.parse_args()
    param_fname = args.cfg
    print('******************* Starting generation *******************')
    main(param_fname)
