
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class PhysioSiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Linear map to patches [Batch, Patches, PatchSize*Channels]
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        # t: [Batch]
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)

class NeuralFlow(nn.Module):
    """
    Neural Activity Flow Matching Model (S2N-Flow)
    Predicts velocity v(u, t, c) for neural activity u.
    Inputs: 
        u: [Batch, Voxels, 1] (Neural Activity State)
        t: [Batch] (Time)
        c: [Batch, ContextDim] (Visual Stimulus Embedding)
    """
    def __init__(
        self,
        num_voxels,
        in_channels=1, # u is 1D
        hidden_size=768,
        depth=12,
        num_heads=12,
        patch_size=64, # Number of voxels per patch
        context_dim=768
    ):
        super().__init__()
        self.num_voxels = num_voxels
        self.in_channels = in_channels
        self.patch_size = patch_size
        
        # Calculate number of patches
        # We pad num_voxels to be divisible by patch_size if needed
        self.pad_voxels = (patch_size - (num_voxels % patch_size)) % patch_size
        self.total_voxels = num_voxels + self.pad_voxels
        self.num_patches = self.total_voxels // patch_size
        
        # Input Embedding
        self.x_embed = nn.Linear(patch_size * in_channels, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        
        # Time & Context Embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.c_embedder = nn.Linear(context_dim, hidden_size) 
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            PhysioSiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        # Final Layer
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)

    def unpatchify(self, x):
        """
        x: [Batch, Patches, PatchSize * Channels]
        returns: [Batch, Voxels, Channels]
        """
        B, P, _ = x.shape
        x = x.view(B, P, self.patch_size, self.in_channels)
        x = x.view(B, P * self.patch_size, self.in_channels)
        return x[:, :self.num_voxels, :]

    def patchify(self, x):
        """
        x: [Batch, Voxels, Channels]
        returns: [Batch, Patches, PatchSize * Channels]
        """
        B, V, C = x.shape
        # Pad
        if self.pad_voxels > 0:
            pad = torch.zeros(B, self.pad_voxels, C, device=x.device)
            x = torch.cat([x, pad], dim=1)
        
        x = x.view(B, self.num_patches, self.patch_size * C)
        return x

    def forward(self, u, t, c):
        # u: [Batch, Voxels, 1]
        # t: [Batch] - Flow Time
        # c: [Batch, D] - Semantic Context
        
        x = self.patchify(u) # [B, P, PatchSize*1]
        x = self.x_embed(x)  # [B, P, D]
        x = x + self.pos_embed
        
        # Combine Time and Context
        t_emb = self.t_embedder(t) # [B, D]
        c_emb = self.c_embedder(c) # [B, D]
        
        cond = t_emb + c_emb # Additive conditioning
        
        for block in self.blocks:
            x = block(x, cond)
            
        x = self.final_layer(x, cond) # [B, P, PatchSize*1]
        x = self.unpatchify(x) # [B, V, 1]
        return x
