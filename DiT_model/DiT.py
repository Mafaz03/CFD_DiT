import torch
from typing import Tuple, List
from einops import rearrange
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

with open(f"{ROOT}/config/config.json", "r") as file:
    config = json.load(file)

class TransformerLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model

        ff_hidden_dim = 4 * self.d_model

        # Layer norm for attention block
        self.att_norm = torch.nn.LayerNorm(self.d_model, elementwise_affine=False, eps=1E-6)

        self.attn_block = torch.nn.MultiheadAttention(embed_dim = d_model, num_heads = num_heads, batch_first = True)
        
        # Layer norm for mlp block
        self.ff_norm = torch.nn.LayerNorm(self.d_model, elementwise_affine=False, eps=1E-6)

        self.mlp_block = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, ff_hidden_dim),
            torch.nn.GELU(approximate='tanh'),
            torch.nn.Linear(ff_hidden_dim, self.d_model),
        )

        # Scale Shift Parameter predictions for this layer
        # 1. Scale and shift parameters for layernorm of attention (2 * d_model)
        # 2. Scale and shift parameters for layernorm of mlp (2 * d_model)
        # 3. Scale for output of attention prior to residual connection (d_model)
        # 4. Scale for output of mlp prior to residual connection (d_model)
        # Total 6 * d_model
        self.adaptive_norm_layer = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(self.d_model, 6 * self.d_model, bias=True)
        )

        ############################
        # DiT Layer Initialization #
        ############################

        torch.nn.init.xavier_uniform_(self.mlp_block[0].weight)
        torch.nn.init.constant_(self.mlp_block[0].bias, 0)
        torch.nn.init.xavier_uniform_(self.mlp_block[-1].weight)
        torch.nn.init.constant_(self.mlp_block[-1].bias, 0)

        torch.nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        torch.nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)

    def forward(self, x, condition):
        scale_shift_params = self.adaptive_norm_layer(condition).chunk(6, dim=1)

        pre_attn_shift, pre_attn_scale, post_attn_scale, pre_mlp_shift, pre_mlp_scale, post_mlp_scale = scale_shift_params
        
        out = x

        attn_norm_output = (self.att_norm(out) * (1 + pre_attn_scale.unsqueeze(1)) + pre_attn_shift.unsqueeze(1))

        attn_out, _ = self.attn_block(
                        attn_norm_output,
                        attn_norm_output,
                        attn_norm_output
                    )

        out = out + post_attn_scale.unsqueeze(1) * attn_out

        mlp_norm_output = (self.ff_norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) + pre_mlp_shift.unsqueeze(1))

        out = out + post_mlp_scale.unsqueeze(1) * self.mlp_block(mlp_norm_output)

        return out
    

def get_time_embedding(time_steps, temb_dim):

    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=temb_dim // 2,
        dtype=torch.float32,
        device=time_steps.device) / (temb_dim // 2))
    )

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps.unsqueeze(-1).repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb

def get_number_embedding(n: torch.Tensor, nemb_dim: int):
    """
    Fourier embedding for continuous values in [0, max_value].
    Scales input to [0, 1] first so frequencies are calibrated correctly.
    """
    # n = (n.float() - config["Stats"]["Re_Mean"]) / config["Stats"]["Re_Std"]       # normalize
    n = n.float()
    assert nemb_dim % 2 == 0
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=nemb_dim // 2,
        dtype=torch.float32,
        device=n.device) / (nemb_dim // 2))
    )

    # pos / factor
    # timesteps B -> [B, 1] -> B, nemb_dim
    n_emb = n.unsqueeze(-1).repeat(1, nemb_dim // 2) / factor
    n_emb = torch.cat([torch.sin(n_emb), torch.cos(n_emb)], dim=-1)
    return n_emb


def get_patch_position_embedding(pos_emb_dim, grid_size: Tuple, device):
    assert pos_emb_dim % 4 == 0, 'Position embedding dimension must be divisible by 4'

    grid_size_h, grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float32, device = device)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32, device = device)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0) # [2, grid_size[0], grid_size[1]]

    # grid_h_positions -> (Number of patch tokens,)
    grid_h_positions = grid[0].reshape(-1) # grid_size[0] * grid_size[1]
    grid_w_positions = grid[1].reshape(-1) # grid_size[0] * grid_size[1]

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=pos_emb_dim // 4,
        dtype=torch.float32,
        device=device) / (pos_emb_dim // 4))
    )

    grid_h_emb = grid_h_positions.unsqueeze(-1).repeat(1, pos_emb_dim // 4) / factor
    grid_h_emb = torch.cat([torch.sin(grid_h_emb), torch.cos(grid_h_emb)], dim=-1)
    # grid_h_emb -> (Number of patch tokens, pos_emb_dim // 2)

    grid_w_emb = grid_w_positions.unsqueeze(-1).repeat(1, pos_emb_dim // 4) / factor
    grid_w_emb = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_w_emb)], dim=-1)

    pos_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1)

    # pos_emb -> (Number of patch tokens, pos_emb_dim)
    return pos_emb # [grid_size[0] * grid_size[1], pos_emb_dim]



class PatchEmbedding(torch.nn.Module):
    def __init__(self, grid_height, grid_width, g_channels, patch_height, patch_width, d_model):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.g_channels = g_channels

        self.d_model = d_model

        self.patch_height = patch_height
        self.patch_width = patch_width

        # Input dimension for Patch Embedding FC Layer
        patch_dim = self.g_channels * self.patch_height * self.patch_width

        self.patch_embed = torch.nn.Sequential(torch.nn.Linear(patch_dim, self.d_model))

        ############################
        # DiT Layer Initialization #
        ############################

        torch.nn.init.xavier_uniform_(self.patch_embed[0].weight)
        torch.nn.init.constant_(self.patch_embed[0].bias, 0)

    def forward(self, x):
        grid_size_h = self.grid_height // self.patch_height
        grid_size_w = self.grid_width // self.patch_width

        # B, C, H, W -> B, (Patches along height * Patches along width), Patch Dimension
        # Number of tokens = Patches along height * Patches along width
        out = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)',
                        ph=self.patch_height,
                        pw=self.patch_width)

        # BxNumber of tokens x Patch Dimension -> B x Number of tokens x Transformer Dimension
        out = self.patch_embed(out)

        # Add 2d sinusoidal position embeddings
        pos_embed = get_patch_position_embedding(pos_emb_dim=self.d_model, grid_size=(grid_size_h, grid_size_w), device=x.device)
        out += pos_embed
        return out # [B, grid_height/patch_height * grid_width/patch_width, d_model]
    
class DiT(torch.nn.Module):
    def __init__(self, d_model, patch_size, grid_size, g_channels, timestep_emb_dim, number_emb_dim, num_layers, num_heads):
        super().__init__()

        num_layers       = num_layers
        self.grid_height = grid_size
        self.grid_width  = grid_size

        self.g_channels = g_channels
        
        self.d_model = d_model

        self.patch_height = patch_size
        self.patch_width  = patch_size

        self.timestep_emb_dim = timestep_emb_dim
        self.number_emb_dim = number_emb_dim

        # Number of patches along height and width
        self.nh = self.grid_height // self.patch_height
        self.nw = self.grid_width // self.patch_width

        # Patch Embedding Block
        self.patch_embed_layer = PatchEmbedding(grid_height  = self.grid_height, 
                                                grid_width   = self.grid_width, 
                                                g_channels   = self.g_channels, 
                                                patch_height = self.patch_height, 
                                                patch_width  = self.patch_width,
                                                d_model      = self.d_model)
        

        # Initial projection from sinusoidal time embedding
        self.t_proj = torch.nn.Sequential(
            torch.nn.Linear(self.timestep_emb_dim, self.d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(self.d_model, self.d_model)
        )

        self.n_proj = torch.nn.Sequential(
            torch.nn.Linear(self.number_emb_dim, self.d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(self.d_model, self.d_model)
        )
        

        # All Transformer Layers
        self.layers = torch.nn.ModuleList([
            TransformerLayer(d_model = d_model, num_heads = num_heads) for _ in range(num_layers)
        ])

        # Final normalization for unpatchify block
        self.norm = torch.nn.LayerNorm(self.d_model, elementwise_affine=False, eps=1E-6)

        # Scale and Shift parameters for the norm
        self.adaptive_norm_layer = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(self.d_model, 2 * self.d_model, bias=True)
        )

        # Final Linear Layer
        self.proj_out = torch.nn.Linear(self.d_model, self.patch_height * self.patch_width * self.g_channels)

        ############################
        # DiT Layer Initialization #
        ############################
        torch.nn.init.normal_(self.t_proj[0].weight, std=0.02)
        torch.nn.init.normal_(self.t_proj[2].weight, std=0.02)

        torch.nn.init.normal_(self.n_proj[0].weight, std=0.02)
        torch.nn.init.normal_(self.n_proj[2].weight, std=0.02)

        torch.nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        torch.nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)

        torch.nn.init.constant_(self.proj_out.weight, 0)
        torch.nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x, t, n):
        '''
        x: Post VAE contour
        t: time
        n: number
        '''

        # Patchify
        out = self.patch_embed_layer(x)

        # Compute Timestep representation
        # t_emb -> (Batch, timestep_emb_dim)
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.timestep_emb_dim)
        n_emb = get_number_embedding(torch.as_tensor(n), self.number_emb_dim)

        # (Batch, timestep_emb_dim) -> (Batch, d_model)
        c_emb = self.t_proj(t_emb) + self.n_proj(n_emb)

        # Go through the transformer layers
        for layer in self.layers:
            out = layer(out, c_emb)

        # Shift and scale predictions for output normalization
        pre_mlp_shift, pre_mlp_scale = self.adaptive_norm_layer(c_emb).chunk(2, dim=1)
        out = (self.norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) +
               pre_mlp_shift.unsqueeze(1))

        # Unpatchify
        # (B,patches,d_model) -> (B,patches,channels * patch_width * patch_height)
        out = self.proj_out(out)
        out = rearrange(out, 'b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)',
                        ph=self.patch_height,
                        pw=self.patch_width,
                        nw=self.nw,
                        nh=self.nh)
        return out
    
if __name__ == "__main__":
    dit = DiT(d_model        = 512,
            patch_size       = 2,
            grid_size        = 32,
            g_channels       = 4,
            timestep_emb_dim = 512,
            number_emb_dim   = 512,
            num_layers       = 12,
            num_heads        = 16)

    
    print(f"{sum(i.numel() for i in dit.parameters()):,}")
    grid = torch.rand(1, 4, 32, 32)
    t = torch.arange(0, 1, 1)
    n = torch.arange(0, 1, 1)
    print(dit(grid, t, n).shape)