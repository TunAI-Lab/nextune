import math
import torch
import torch.nn as nn
from models.backbones.layers import modulate, AdaTransformerEnc



#################################################################################
#               Embedding Layer for Timesteps                                   #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    For denoising timesteps embedding in the final diffusion model
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=1000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                           Core NexTune Model                                  #
#################################################################################
class FinalLayer(nn.Module):
    """
    The final layer of the NexTune
    Used for adaLN, project x the to desired output size.
    """
    def __init__(self, hidden_size, n_mels):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(hidden_size, n_mels, bias=True)

    def forward(self, x, c):
        # (B, L, H), (B, H)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)      # (B, H), (B, H)
        x = modulate(self.norm(x), shift, scale)                     # (B, L, H)
        x = self.proj(x)                                             # (B, L, N)
        x = x.permute(0, 2, 1)                                       # (B, N, L)
        return x
    
class NexTune(nn.Module):
    """
    Diffusion backbone with Transformer layers.
    """
    def __init__(
        self,
        seq_length,
        hist_length,
        n_mels,
        use_ckpt_wrapper,
        hidden_size,
        num_heads,
        depth,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.t_embedder = TimestepEmbedder(hidden_size)         
        self.proj = nn.Linear(n_mels, hidden_size, bias=True)
        
        self.t_pos_embed = nn.Parameter(
            torch.zeros(1, hist_length+seq_length, hidden_size),
            requires_grad=True,
        )
        self.t_blocks = nn.ModuleList([
            AdaTransformerEnc(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size, n_mels)    
        self.hist_length = hist_length
        self.use_ckpt_wrapper = use_ckpt_wrapper
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation layers for t_block:
        for block in self.t_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.proj.weight, 0)
        nn.init.constant_(self.final_layer.proj.bias, 0)
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    
    def forward(self, x, t, h):
        """
        Forward pass of NexTune.
        - x: (B, N, L_x) tensor of future audio chunk where N:n_mels, L_x:seq_length
        - t: (B,) tensor of diffusion timesteps     
        - h: (B, N, L_h) tensor of history audio chunk where N:n_mels, L_h:hist_length
        """
        
        ##################### Cat and Proj ##########################
        # (B, N, L_x), (B, N, L_h)
        x = torch.cat((h, x), dim=2)                    # (B, N, L)
        x = x.permute(0, 2, 1)                          # (B, L, N)
        x = self.proj(x)                                # (B, L, H)
        #############################################################
        
        ###################### Embedders ############################
        # (B, t_max=1000)
        c = self.t_embedder(t)                          # (B, H)
        #############################################################
        
        ################# Temporal Attention ########################
        # (B, L, H), (B, H)
        x = x + self.t_pos_embed                        # (B, L, H)
        if self.use_ckpt_wrapper:
            for block in self.t_blocks:
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block),
                    x, c, use_reentrant=False,
                )                                       # (B, L, H)
        else:
            for block in self.t_blocks:
                x = block(x, c)                         # (B, L, H)
        #############################################################
        
        ##################### Final layer ###########################
        # (B, L, H), (B, H)
        x = self.final_layer(x, c)                      # (B, N, L)
        #############################################################
        
        return x[:, :, self.hist_length:]


#################################################################################
#                               NexTune Configs                                 #
#################################################################################
def NexTune_L(**kwargs):
    return NexTune(hidden_size=768, num_heads=16, depth=24, **kwargs)

def NexTune_B(**kwargs):
    return NexTune(hidden_size=512, num_heads=8, depth=16, **kwargs)

def NexTune_S(**kwargs):
    return NexTune(hidden_size=384, num_heads=4, depth=8, **kwargs)

NexTune_models = {
    'NexTune-L': NexTune_L,
    'NexTune-B': NexTune_B,
    'NexTune-S': NexTune_S,
}