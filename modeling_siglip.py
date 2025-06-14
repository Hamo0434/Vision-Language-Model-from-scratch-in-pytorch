from typing import Tuple , Optional
import torch 
import torch.nn as nn 
class SiglipVisionConfig:
    def __init__(self,
                 hidden_size = 786, 
                 intermediate_size = 3072 ,   # the size of the of linear layer used in feedforward network
                 num_hidden_layers = 12 , 
                 num_channel = 3 , 
                 image_size  = 224 , 
                 patch_size = 16 , 
                 layer_norm_eps = 1e-6 , 
                 attention_dropout = 0.0 , 
                 num_image_tokens :int = None, 
                 num_attention_heads  = 12 ,
                  **kwargs
                  ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_channel = num_channel
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens
        self.num_attention_heads = num_attention_heads
        
        
class SiglipVisionEmbedding(nn.Module):
    def __init__(self,config :SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embd_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
         
        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channel,
            out_channels= self.embd_dim ,
            kernal_size = self.patch_size, 
            stride = self.patch_size,
            padding = 'valid'
        )
        
        self.num_pathes= (self.image_size // self.patch_size) **2
        self.num_positions = self.num_pathes                         # how many positions we need  
        self.postions_embedding = nn.Embedding(self.num_positions, self.embd_dim)
        self.register_buffer(                                                           #
            'position_ids ' , 
            torch.arange(self.num_positions).expand((-1 , 1)),
            persistent = False ,
            )
        
    def forward(self, pixel_values:torch.FloatTensor)-> torch.torch:
            
        _ , _ , height , width  = pixel_values.shape 
        # convolve the patch_size  kernal over the image , with no overlaping 
        # the output  of the convolution will have shape [batch_size, embd_dim, num_pathes_h , num_pathes_w]
        # wher num_patchesh = height // patch_size etc.
        
        patch_embds = self.patch_embedding(pixel_values)
        # converting from this  [batch_size, embd_dim, num_pathes_h , num_pathes_w] to this [batch_size, embd_dim, num_patches ] where num_pathes = num_pathes_h * num_pathes_w
        embeddings = patch_embds.flatten(2)
        # adding the positions to the patches  , each positional encoding is a vector of size 
        embeddings = embeddings + self.postions_embedding(self.position_ids)
        
        return embeddings

class SiglipAttention(nn.Module):
    def __init__(self, config :SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embd_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embd_dim // self.num_heads
        self.scale = self.head_dim **-0.5              # 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout
        
        self.k_proj = nn.Linear(self.embd_dim , self.embd_dim)        # wk 
        self.v_proj = nn.Linear(self.embd_dim , self.embd_dim)
        self.q_proj = nn.Linear(self.embd_dim , self.embd_dim)
        self.out_proj = nn.Linear(self.embd_dim , self.embd_dim)
        
    def forward(self,hidden_states :torch.Tensor) -> Tuple[torch.Tensor , Optional[torch.tensor]]:
        batch__size , seq_len , _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)            # [batch_size , seq_len, embd_dim]
        key_states = self.k_proj(hidden_states)
        values_states = self.v_proj(hidden_states)
        
        # [batch_size , seq_len(num_patches), num_head  , head_dim ]
        query_states = query_states.view(batch__size , seq_len, self.num_heads , self.head_dim).transpose(1 , 2)
        key_states = key_states.view(batch__size , seq_len , self.num_heads , self.head_dim).transpose(1 , 2)
        values_states = values_states.view(batch__size , seq_len , self.num_heads , self.head_dim).transpose(1, 2)
        # calculate the attention weights 
        atten_weights = (torch.matmul(query_states , key_states.transpose(2, 3)) * self.scale)
        
        
        if atten_weights.size() != (batch__size , self.num_heads , seq_len , seq_len):
            raise ValueError(f'attention weights should be of size{(batch__size , self.num_heads , seq_len , seq_len)} , but is '
                               
                            f"{atten_weights.size()}")      
        
        # applying the softmax in attention weights  [batch__size, num_heads , patch_size  , patch_size]
        atten_weights = nn.functional.softmax(atten_weights , dim = -1 , dtype = torch.float32 ).to(query_states.dtype)
        atten_weights = nn.functional.dropout(atten_weights , p = self.scale , training = True)
        atten_output = torch.matmul(atten_weights, values_states.values)  
        
        if atten_output .size() != (batch__size , self.num_heads , seq_len , self.head_dim):
            raise ValueError(
                f"attention output should be {(batch__size, self.num_heads, seq_len ,self.head_dim )} but is:"
                f"{atten_output.size()}"
            )
        
        atten_output = atten_output.transpose(1, 2).contiguous()  #return it to the first dimention -> [batch_size , seq_len , num_heads , head_dim]   : contiguous here means we want to reshape 
        atten_output = atten_output.reshape(batch__size ,seq_len , self.embd_dim)
        
        atten_output = self.out_proj(atten_output)
        
        return atten_output , atten_weights
        
class SiglipMLP(nn.Module):
    def __init__(self, config :SiglipVisionConfig):
        super().__init()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size , config.intermediate_size)
        self.fc2 =nn.Linear(config.intermediate_size , config.hidden_size)
        
    def forward(self , hidden_States :torch.Tensor) -> torch.Tensor: 
        
        hidden_States  = self.fc1(hidden_States)
        
        hidden_States = nn.functional.gelu(hidden_States, approximate = 'tanh')
        hidden_States = self.fc2(hidden_States)
        
        return hidden_States
 
class SiglipEncoderLayer(nn.Module):

    def __init__(self, config :SiglipVisionConfig):
        
        super().__init__()
        self.embd_dim = config.hidden_size
        self.self._atten = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embd_dim , eps = config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embd_dim , eps = config.layer_norm_eps)
   
    def forward(self , 
                hidden_states :torch.Tensor)  -> torch.Tensor  :
        residual = hidden_states                        # residual [batch_size, num_patches , embd_dim]
        hidden_states = self.layer_norm1(hidden_states) #[batch_size, num_patches, embd_dim] -> [batch_size , num_patches, embd_dim]
        hidden_states = self.self_atten(hidden_states = hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        
        hidden_states = self.layer_norm2(hidden_states) 
        hidden_states = self.mlp(hidden_states) 
        hidden_states = hidden_states +residual
        
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config)  for _ in range(config.num_hidden_layers)])
        
    def forward(self, inputs_embds : torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embds
        
        for encoder_layer in self.layers :
            hidden_states = encoder_layer(hidden_states)
        return hidden_states
        
class  siglipVisionTransformer(nn.Module):
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        self.config = config
        embd_dim = config.hidden_size
        
        self.embeddings = SiglipVisionEmbedding(config)
        self.encoder = SiglipEncoder(config)
        self.post_layer_norm = nn.LayerNorm(embd_dim , eps = config.layer_norm_eps)
    
    def forward(self, pixel_values :torch.Tensor )  -> torch.Tensor :
        hidden_states = self.embeddings (pixel_values)
        last_hidden_state = self.encoder(inputs_embeds = hidden_states)
        last_hidden_state = self.post_layer_norm(last_hidden_state)
        
        

class SiglipVisionModel(nn.Module):
    def __init__(self, config = SiglipVisionConfig):
        super(SiglipVisionModel, self).__init__()
        self.config = config 
        self.vision_model = siglipVisionTransformer(config)
        
    def forward(self , pixel_values) -> Tuple:
        # [batch size, channels,  hight , width ] -> [batch size, num_batches , embed dimension]
        return self.vision_model(pixel_values  = pixel_values)
    
    
     