from pyexpat import model
import token
from PIL import Image
from semver import process
from sphinx import ret
from streamlit import image
import torch
import fire

from processing_paligamma import PaliGemmaProcessor
from modeling_gamma import KVCache , PliGemmaForConditionalGeneration
from utils import load_hf_model
def move_inputs_to_device(model_inputs: dict , device:str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs

def _sample_top_p(probs : torch.Tensor, p : float):
    
    # Sort the probabilities in descending order and get the corresponding indices
    probs_sorted, indices = torch.sort(probs, descending=True)
    # Compute the cumulative sum of the sorted probabilities
    cumulative_probs = torch.cumsum(probs_sorted, dim=-1)
    # Create a mask for probabilities that exceed the top-p threshold
    mask = cumulative_probs - probs_sorted > p
    # Zero out probabilities that are masked
    probs_sorted[mask] = 0.0
    # Normalize the remaining probabilities
    probs_sorted /= probs_sorted.sum(dim=-1, keepdim=True)
    # Sample the next token from the filtered distribution
    next_token = torch.multinomial(probs_sorted, num_samples=1)
    # Map the sampled token back to the original indices
    next_token = indices.gather(-1, next_token)
    return next_token


def get_model_inputs(processor : PaliGemmaProcessor ,image_file_path ,  prompt : str , device : str):
    
    image = Image.open(image_file_path)
    images  = [image]
    prompts = [prompt]
    model_inputs = processor(text = prompts , images = images )
    model_inputs = move_inputs_to_device(model_inputs , device)
    return model_inputs
def test_inference(
    model : PliGemmaForConditionalGeneration , 
    processor : PaliGemmaProcessor ,
    device : str,
    prompt : str,
    image_file_path : str,
    max_token_to_generate : int,
    temperature : float,
    top_p : float,
    do_sample : bool,
):
    model_inputs = get_model_inputs(processor , prompt , image_file_path , device)
    input_ids = model_inputs['input_ids']
    attention_mask = model_inputs['attention_mask']
    pixel_values = model_inputs['pixel_values']
    
    kv_cache = KVCache() 
    
    
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []
    
    for _ in range(max_token_to_generate):
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            pixel_values = pixel_values,
            kv_cache = kv_cache,
        )
        kv_cache = outputs['kv_cache']
        next_token_logits = outputs['logits'][:, -1, :]
        
        if do_sample:
            next_token_logits = torch.softmax(next_token_logits / temperature , dim = -1)
            next_token = _sample_top_p(next_token_logits , top_p)
        else:
            next_token = torch.argmax(next_token_logits , dim = -1 , keepdim= True) 
        
        assert next_token.size() == (1, 1)
        next_token = next_token.usnqueeze(0)
        generated_tokens.append(next_token)
        
        if next_token.item() == stop_token:
            break
        input_ids = torch.cat([input_ids , next_token], dim = 1)      
    
    generated_tokens = torch.cat(generated_tokens, dim=1)
    decoded = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    print(prompt + decoded)     
        
def main(
    model_path : str = None,
    prompt : str = None,
    image_file_path : str = None,
    max_token_to_generate : int = 120,
    temperature : float = 0.8,
    top_p : float = 0.95,
    do_sample : bool = False,
    only_cpu : bool = False,
):
    device = 'cpu'
    if not only_cpu:
        if torch.cuda.is_available():
            device = 'cuda'
    elif torch.backends.mps.is_available(): 
        device = 'mps'

    print('device in use:', device)
    
    print(f'loading model')
    model , tokenizer = load_hf_model(model_path, device)
    model.to(device).eval()
    
    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer , num_image_tokens , image_size)
    
    print('Runing Inference')
    with torch.no_grad():
        
        test_inference(
            model,
            processor, 
            device,
            prompt , 
            image_file_path , 
            max_token_to_generate ,
            temperature ,
            top_p ,
            do_sample  ,
        )
        
if __name__ == "__main__":
    fire.Fire(main)        