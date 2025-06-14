import tokenize

from altair import Padding
from modeling_gamma import PliGemmaForConditionalGeneration , PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os 
import torch

def load_hf_model(model_path: str, device: str) -> Tuple[PliGemmaForConditionalGeneration, AutoTokenizer]:
    
    tokenizer = AutoTokenizer.from_pretrained(model_path  , Padding_side = 'right')
    assert tokenizer.padding_side =='right'
    
    safetensors_files = glob.glob(os.path.join(model_path , '*.safetensors'))
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device='cpu') as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
                
    with open(os.path.join(model_path , 'config.json')) as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)
        
    model = PliGemmaForConditionalGeneration(config).to(device)
    
    # load the state dict of the model
    model.load_state_dict(tensors , strict=False)
    
    # tie weights
    model.tie_weights()             
    
    return (model , tokenizer)

   