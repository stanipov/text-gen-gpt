from utils import NLP_Model, CTGDataset, seed_everything

import os, json, random, pickle
import numpy as np
import datetime

from transformers import TrainingArguments, Trainer

import torch



def convert_to_type(param_str, def_val, def_val_type):
    """
    Converts a value to a designated type (e.g. int of float)
    and prints error if fails and returns default
    """
    try:
        param_str = def_val_type(weight_decay)
    except Exception as e:
        print(e)
        param_str = def_val
    return param_str
#######################################################################################################
#
# MAIN
#
#######################################################################################################
def main(par_file):

    # read paramaters
    with open(par_file,'rb') as f:
        params = json.load(f)
    
    flds2float = ['lr','eps', 'warmup']
    for fld in flds2float:
        params[fld] = float(params[fld])

    flds2int = ['unfreeze_last_n', 'seed', 'epochs', 'warmup', 'train_batch_size', 'batch_update','max_len']
    for fld in flds2int:
        params[fld] = int(params[fld])
        
    # Load data
    with open(params['train'], 'rb') as f:
        train_data = pickle.load(f)
        
    with open(params['test'], 'rb') as f:
        test_data = pickle.load(f)


    model_alias           = params['model']
    print('\n===================================================================')
    print(f"Model: {model_alias}")
    
    EPOCHS          = params['epochs']
    LR              = params['lr']
    EPS             = params['eps']
    WARMUP_STEPS    = params['warmup']

    TRAIN_BATCHSIZE = params['train_batch_size']
    BATCH_UPDATE    = params['batch_update']
    
    UNFREEZE_LAST_N = params['unfreeze_last_n'] 
    
    SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                        "eos_token": "<|EOS|>",
                        "unk_token": "<|UNK|>",                    
                        "pad_token": "<|PAD|>",
                        "sep_token": "<|SEP|>"}
    MAXLEN          = params['max_len']
    SEED            = params['seed']
    device          = torch.device('cuda') 

    print(f"Max length: {MAXLEN}")
    seed_everything(SEED)

    cache_dir = os.path.join(params['cache_dir'], model_alias)
    
    model_preload = params.pop('model_preload', None)

    dtstr = datetime.datetime.now().strftime('%Y-%m-%d') 
    
    if '/' in model_alias:
        m_name = model_alias.split('/')[-1]
    else:
        m_name = model_alias
        
    model_name = f'{m_name}-{UNFREEZE_LAST_N}l-{EPOCHS}ep_{dtstr}'
    try:
        os.makedirs(os.path.join(os.getcwd(), model_name) )
    except:
        pass    
    model_path = os.path.join(os.getcwd(), *(model_name, 'trained_model'))
    print(f"Model path:\n\t{model_path}")

    print('Setting the tokenizer and model')
    nlp_model = NLP_Model(model_alias, torch.half, cache_dir, SPECIAL_TOKENS, model_preload)
    model = nlp_model.model.half().to(device)
    
    print(f"Total number of layers: {len(model.transformer.h)}")
    print(f"Un-freezing last {UNFREEZE_LAST_N}")
    nlp_model.unfreeze_last_n(UNFREEZE_LAST_N)
    print('Done')
    
    # Making the DataSet instances
    train_dataset = CTGDataset(train_data, nlp_model.tokenizer, SPECIAL_TOKENS, device, MAXLEN, randomize=True)
    test_dataset = CTGDataset(test_data, nlp_model.tokenizer, SPECIAL_TOKENS, device, MAXLEN, randomize=True)

    ds_config = params['ds_config']
    output_dir = os.path.join(os.getcwd(), model_name)
    log_dir = os.path.join(os.getcwd(), 'LOGS')
    
    try:
        os.makedirs(log_dir)
    except Exception as e:
        print(e)
          
    save_total_chkpts = params.pop('max_checkpts_num', None)
    if save_total_chkpts:
        save_total_chkpts = convert_to_type(save_total_chkpts, 1, int)

    weight_decay = params.pop('weight_decay', 0.01)
    if weight_decay:
        weight_decay = convert_to_type(weight_decay, 0.01, float)

    fp16 = params.pop('fp16_train', True)
    if fp16:
        fp16 = convert_to_type(fp16, True, bool)    

    fp16_full_eval = params.pop('fp16_full_eval', False)
    if fp16_full_eval:
        fp16_full_eval = convert_to_type(fp16_full_eval, False, bool)
         
    # Hugging Face trainer arguments
    training_args = TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = TRAIN_BATCHSIZE,
        per_device_eval_batch_size  = TRAIN_BATCHSIZE,
        gradient_accumulation_steps = BATCH_UPDATE,
        evaluation_strategy         = "epoch",
        fp16                        = fp16,
        fp16_full_eval              = fp16_full_eval,
        warmup_steps                = WARMUP_STEPS,    
        learning_rate               = LR,
        adam_epsilon                = EPS,
        weight_decay                = weight_decay,        
        save_total_limit            = save_total_chkpts,
        save_strategy               = 'epoch',
        load_best_model_at_end      = False,     
        logging_dir                 = log_dir + '/log.txt',
        logging_strategy            = 'steps',
        deepspeed                   = ds_config
    )

    # Instantiate the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,    
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=nlp_model.tokenizer
    )
    
    # Train
    trainer.train()
    
    # Save the model
    trainer.save_model(model_path) 
    
#######################################################################################################
#
# CLI
#
#######################################################################################################    
if __name__=='__main__':
    import sys
    
    args = sys.argv
    max_args = 100
    cnt = 0
    param_fname = None
    
    if len (args) > 1:
        while len(args) > 1 and cnt < max_args:
            
            item = args.pop(0)
            if '-cfg' in item:
                try:
                    param_fname = args.pop(0)
                except Exception as e:
                    print(e)
            cnt += 1

    if param_fname:        
        main(param_fname)
    else:
        print('No config is provided')
