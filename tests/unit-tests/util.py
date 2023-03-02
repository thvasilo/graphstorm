import torch as th

from graphstorm.model.lm_model import TOKEN_IDX, ATT_MASK_IDX, TOKEN_TID_IDX

class Dummy: # dummy object used to create config objects
    # constructor
    def __init__(self, arg_dict):
        self.__dict__.update(arg_dict)

def create_tokens(tokenizer, input_text, max_seq_length, num_node, return_token_type_ids=False):
    input_text = input_text * num_node
    tokens = tokenizer(input_text,  max_length=max_seq_length,
                        truncation=True, padding='max_length', return_tensors='pt')
    # we only use TOKEN_IDX and VALID_LEN_IDX
    input_ids = tokens[TOKEN_IDX]
    attention_mask = tokens[ATT_MASK_IDX]
    valid_len = tokens[ATT_MASK_IDX].sum(dim=1)
    token_type_ids = tokens[TOKEN_TID_IDX] if return_token_type_ids else None
    return input_ids, valid_len, attention_mask, token_type_ids
