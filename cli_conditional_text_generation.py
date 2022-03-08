from utils import NLP_Model, CTGDataset, seed_everything

import os, json, random, pickle
import numpy as np
import datetime

from transformers import TrainingArguments, Trainer

import torch
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
    print(f"Max length: {MAXLEN}")

    SEED   = params['seed']
    device = torch.device('cuda') 
    seed_everything(SEED)

    cache_dir = params['cache_dir']

    dtstr = datetime.datetime.now().strftime('%Y-%m-%d') #%H-%M
    
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
    nlp_model = NLP_Model(model_alias, torch.half, cache_dir, SPECIAL_TOKENS)
    model = nlp_model.model.to(device)
    
    print(f"Total number of layers: {len(model.transformer.h)}")
    print(f"Un-freezing last {UNFREEZE_LAST_N}")
    nlp_model.unfreeze_last_n(UNFREEZE_LAST_N)
    print('Done')


    # Making the DataSet instances
    train_dataset = CTGDataset(train_data, tokenizer, SPECIAL_TOKENS, device, MAXLEN, randomize=True)
    test_dataset = CTGDataset(test_data, tokenizer, SPECIAL_TOKENS, device, MAXLEN, randomize=True)

    # Training in a Jupyter Notebook
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '9994' # modify if RuntimeError: Address already in use
    #os.environ['RANK'] = "0"
    #os.environ['LOCAL_RANK'] = "0"
    #os.environ['WORLD_SIZE'] = "1" #deepspeed --num_gpus=1 new2_cli_conditional_text_generation.py --deepspeed ds_config.json
    #os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    #os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = "1"
    #os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:10'
    
    ds_config = params['ds_config']
    output_dir=os.path.join(os.getcwd(), (model_name))
    
    # Hugging Face trainer arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCHSIZE,
        per_device_eval_batch_size=TRAIN_BATCHSIZE,
        gradient_accumulation_steps=BATCH_UPDATE,
        evaluation_strategy="epoch",
        fp16=True,
        fp16_full_eval = False,
        warmup_steps=WARMUP_STEPS,    
        learning_rate=LR,
        adam_epsilon=EPS,
        weight_decay=0.01,        
        save_total_limit=1,
        save_strategy = 'epoch',
        load_best_model_at_end=False,     
        logging_dir = 'LOG_%s' % output_dir,
        logging_strategy = 'steps',
        deepspeed=ds_config
    )

    # Instantiate the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,    
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer
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
