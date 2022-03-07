import torch, os, random
import numpy as np
from transformers import GPTNeoForCausalLM, GPT2LMHeadModel, GPT2Tokenizer, GPTNeoConfig, GPT2Config, GPTNeoModel, Trainer
#######################################################################################################
#
# Language model
#
####################################################################################################### 
class NLP_Model:
    def __init__(self, model: str, dtype = torch.half, 
                 cache_dir: str = None, special_tokens: dict = None, 
                 load_model_path: str = None):
        
        # setting the tokenizer
        print('Setting the tokenizer')
        self.tokenizer = GPT2Tokenizer.from_pretrained(model, cache_dir = cache_dir)
        if special_tokens:
            self.tokenizer.add_special_tokens(special_tokens)
        print('Done')
            
        print('Setting the model')
        if special_tokens:
            if 'neo' in model:
                self._config = GPTNeoConfig.from_pretrained(model, cache_dir = cache_dir,
                                                            bos_token_id = self.tokenizer.bos_token_id,
                                                            eos_token_id = self.tokenizer.eos_token_id,
                                                            sep_token_id = self.tokenizer.sep_token_id,
                                                            pad_token_id = self.tokenizer.pad_token_id,
                                                            output_hidden_states = False,
                                                            dtype = dtype)
            if 'gpt2' in model:
                self._config = GPT2Config.from_pretrained(model, cache_dir = cache_dir,
                                            bos_token_id = self.tokenizer.bos_token_id,
                                            eos_token_id = self.tokenizer.eos_token_id,
                                            sep_token_id = self.tokenizer.sep_token_id,
                                            pad_token_id = self.tokenizer.pad_token_id,
                                            output_hidden_states = False,
                                            dtype = dtype)
        else:
            if 'neo' in model:
                self._config = GPTNeoConfig.from_pretrained(model, cache_dir = cache_dir,
                                                            pad_token_id = self.tokenizer.pad_token_id,
                                                            output_hidden_states = False,
                                                            dtype = dtype)
            if 'gpt2' in model: 
                self._config = GPT2Config.from_pretrained(model, cache_dir = cache_dir,
                                                          pad_token_id = self.tokenizer.pad_token_id,
                                                          output_hidden_states = False,
                                                          dtype = dtype)
        
        if 'neo' in model:
            self.model = GPTNeoForCausalLM.from_pretrained(model, 
                                                           config = self._config, 
                                                           cache_dir = cache_dir)
        if 'gpt2' in model:
            self.model = GPT2LMHeadModel.from_pretrained(model,
                                                         config = self._config,
                                                         cache_dir = cache_dir)
        
        # resize for the custom tokenizer
        if special_tokens:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # load pretrained model from file
        if load_model_path:
            self.model.load_state_dict(torch.load(load_model_path))
        print('Done')
        
        
    def unfreeze_last_n(self, N):
        """
        Unfreeze last N transformer heads in the model.
        Useful when limited in GPU memory        
        """
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        for i, m in enumerate(self.model.transformer.h):        
            if i+1 > len(self.model.transformer.h) - N:
                for parameter in m.parameters():
                    parameter.requires_grad = True 

        for parameter in self.model.transformer.ln_f.parameters():        
            parameter.requires_grad = True

        for parameter in self.model.lm_head.parameters():        
            parameter.requires_grad = True
    
    
    @staticmethod
    def gen_prompt(title: str, kws: str, special_toks: dict):
        """
        Generates prompt from title and keywords
        """
        return special_toks['bos_token'] + title + \
                special_toks['sep_token'] + kws + special_toks['sep_token']      
#######################################################################################################
#
# Dataset
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
#######################################################################################################
#
# Utilities
#
#######################################################################################################
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
