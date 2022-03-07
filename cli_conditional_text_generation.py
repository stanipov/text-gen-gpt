#!/home/sf/data/linux/py_env/transformers_ds/bin/python

# deepspeed --num_gpus=1 new2_cli_conditional_text_generation.py --deepspeed ds_config.json

from sklearn.model_selection import train_test_split
import os, json, random, pickle
import numpy as np
import datetime
import argparse

from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining,TrainingArguments, Trainer

import torch
from torch.utils.data import Dataset
#######################################################################################################
#
# FUNCS
#
#######################################################################################################   
class MyShittyDataset(Dataset):
    
    def __init__(self, data, tokenizer, SPECIAL_TOKENS, torch_device, max_len, randomize=True):
        self.tokenizer = tokenizer
        self.keywords = data['keywords']
        self.data = data['text']
        self.title = data['title']
        self.spec_tok = SPECIAL_TOKENS
        self.device = torch_device
        self.randomize = randomize
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def join_keywords(keywords, randomize=True):
    
        if randomize:
            # if the keywords are empty, then skip shuffling
            # and return an empty line
            try:
                random.shuffle(keywords)
            except Exception as e:
                pass
            
        try:            
            _kws = ', '.join(keywords)
        except:
            _kws = ''
        return _kws
    
    def __getitem__(self, i):
        kws = self.join_keywords(self.keywords[i], self.randomize)

        tok_input = self.spec_tok['bos_token'] + self.title[i] + \
        self.spec_tok['sep_token'] + kws + self.spec_tok['sep_token'] + \
        self.data[i] + self.spec_tok['eos_token']
        
        encodings_dict = self.tokenizer(tok_input,                                   
                                   truncation=True, 
                                   max_length=self.max_len, 
                                   padding="max_length")   
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']
        
        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids), 
                'attention_mask': torch.tensor(attention_mask)}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
def get_tokenier(model, special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(model, return_token_type_ids=False) 
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print("Special tokens added")
    return tokenizer


def get_model(model, tokenizer, device, special_tokens=None, load_model_path=None):

    #GPT2LMHeadModel
    if special_tokens:
        config = AutoConfig.from_pretrained(model, 
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False, dtype=torch.half,)
    else: 
        config = AutoConfig.from_pretrained(model,                                     
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False, dtype=torch.half)    


    model = AutoModelForPreTraining.from_pretrained(model, config=config)

    if special_tokens:
        #Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    model.half().to(device)
    return model
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


    MODEL           = params['model']
    print('\n===================================================================')
    print(f"Model: {MODEL}")
    
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
    MAXLEN = params['max_len']
    print(f"Max length: {MAXLEN}")

    SEED   = params['seed']
    seed_everything(SEED)
    device = torch.device('cuda') 

    dtstr = datetime.datetime.now().strftime('%Y-%m-%d') #%H-%M
    model_name = f'{MODEL}-{UNFREEZE_LAST_N}l-{EPOCHS}ep_{dtstr}'
    try:
        os.makedirs(os.path.join(os.getcwd(), model_name) )
    except:
        pass    
    model_path = os.path.join(os.getcwd(), *(model_name, 'trained_model'))
    print(f"Model path:\n\t{model_path}")

    print('Setting the tokenizer')
    tokenizer = get_tokenier(MODEL, special_tokens=SPECIAL_TOKENS)
    print('Done')
    print('Setting the model')
    
    if params['model_preload'].lower() == 'none':
        model = get_model(MODEL, tokenizer, device,
                          special_tokens=SPECIAL_TOKENS,
                         )
    else:
        model = get_model(MODEL, tokenizer, device,
                          special_tokens=SPECIAL_TOKENS,
                          load_model_path = params['model_preload'].lower(),
                         )

    print(f"Total number of layers: {len(model.transformer.h)}")
    print(f"Un-freezing last {UNFREEZE_LAST_N}")
    
    # Unfreezing last N layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    for i, m in enumerate(model.transformer.h):        
        #Only un-freeze the last n transformer blocks
        if i+1 > len(model.transformer.h) - UNFREEZE_LAST_N:
            for parameter in m.parameters():
                parameter.requires_grad = True 

    for parameter in model.transformer.ln_f.parameters():        
        parameter.requires_grad = True

    for parameter in model.lm_head.parameters():        
        parameter.requires_grad = True
    print('Done')


    # Making the DataSet instances
    train_dataset = MyShittyDataset(train_data, tokenizer, SPECIAL_TOKENS, device, MAXLEN, randomize=True)
    test_dataset = MyShittyDataset(test_data, tokenizer, SPECIAL_TOKENS, device, MAXLEN, randomize=True)

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
    # CLI paratemrs parser
#    arg_parser = argparse.ArgumentParser(description='Train model')
#    arg_parser.add_argument('-path',
#                           metavar='path',
#                           type=str,
#                           help='Path to the parameters json')
#    args = arg_parser.parse_args()
#    param_fname = args.path
    
    param_fname = 'train_cfg_gpt2-l.json'
    
    import sys
    
    max_args = 100
    cnt = 0
    
    main(param_fname)
