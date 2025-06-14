from ast import Str
import token
from typing import Optional , List , Dict , Union, Tuple  , Iterable
from matplotlib.cbook import ls_mapper
from matplotlib.scale import scale_factory
import numpy as np
from PIL import Image
from sklearn.utils import resample
from streamlit import image
import torch

IMAGENET_STANDARD_MEAN = [0.5 , 0.5 , 0.5 ]
IMAGENET_STANDARD_STD = [0.5 , 0.5 , 0.5]
def add_image_tokens_to_prompt(image_token , image_seq_len , bos_token , prefix_token):
    return f'{image_token * image_seq_len}{bos_token}{prefix_token}\n'
def resize(
    image : Image , 
    size :Tuple [int , int ],
    resample :Image.Resampling = None,
    reducing_gap :Optional[int] = None,
 ) -> np.ndarray:
    height , width = size , 
    resize_image = image.resize( (height, width) , resample = resample ,reducing_gap = reducing_gap)
    return resize_image

def rescale(
    image : np.ndarray ,
    scale :float ,
    dtype : np.dtype=np.float32
    
) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def normalize(
    image : np.ndarray,
    mean : Union[float , Iterable[float]] , 
    std : Union[float , Iterable[float]] ,
) -> np.ndarray : 
    mean = np.array(mean , dtype= image.dtype)
    std = np.array(std , dtype= image.dtype)
    image = (image - mean) /std
    return image 
    

def process_images(images:List[Image.Image],
                  size :Dict[str , int ] = None , 
                  resample :Image.Resampling = None ,
                  image_mean : Optional[Union[float , List[float]]] = None, 
                  image_std : Optional[Union[float , List[float]]] = None, 
                  rescale_factor :float = None
)-> List[np.array]:
    height , width = size[0] , size[1]
    images = [
        resize(image = image , size = (height, width) , resample = resample) for image in images
    ]
    
    images = [np.array(image) for image in images]
    images = [rescale(image , scale = rescale_factor) for image in images]
    images = [normalize(image ,mean = image_mean , std = image_std) for image in images]
    images = [image.transpose(2, 0 , 1) for image in images] # move the channel to be the first  dimension . the model expect images in the format [channels , height , width]  
    return images 



class PaliGemmaProcessor:
    
    IMAGE_TOKEN = "<image>"
    
    def __init__(self, tokenizer , num_image_tokens: int , image_size : int):
        super().__init__()
        self.image_seq_len = num_image_tokens
        self.image_size = image_size
        
        tokens_to_add = {"additional_special_tokens ":[self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        
        EXTRA_TOKENS  =[
        f'<loc {i :04d}>' for i in range(1024) # these tokens are used for object detection(bounding box)
        ] 
        EXTRA_TOKENS  +=[
        f'<seg {i :03d}>' for i in range(128) # these tokens are used for object segementation
        ] 
        
        tokenizer.add_tokens(EXTRA_TOKENS) 
        
        self.image_token_id = tokenizer.convert_token_to_id(self.IMAGE_TOKEN)              
        # we will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        
        self.tokenizer = tokenizer
        
    def __call__(self,
                     text :List[str],
                     images: List[Image.Image],
                     padding: str ='longest',
                     trunction :bool = True,) -> dict:
        assert len(images) == 1 and len(text)==1 , f'recieved {len(images)} images for {len(text)} prompts'
            
        pixel_values = process_images(
                images , 
                size = (self.image_size , self.image_size), 
                resample = Image.Resampling.BICUBIC , 
                rescale_factor = 1 /255.0 ,
                image_mean = IMAGENET_STANDARD_MEAN ,
                image_std = IMAGENET_STANDARD_STD 
            )
        
        pixel_values = np.stack(pixel_values , axis = 0 )  # convert the list of numpy array to  a single numpy array with shape [batch_size, cahnnel , hieght , width]
        pixel_values = torch.tensor(pixel_values ) # convert the list of  numpy array to pytorch tensor 
        
        input_string_tokens_to_prompt =  [ 
            add_image_tokens_to_prompt(
            prefix_prompt= prompt , 
            bos_token = self.tokenizer.bos_token,
            image_seq_len = self.image_seq_len, 
            image_token = self.IMAGE_TOKEN
            )
             for prompt in text  ]
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         