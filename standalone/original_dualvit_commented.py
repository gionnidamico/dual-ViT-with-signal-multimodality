import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np


"""
This function generates random bounding box coordinates on an image using the size and a lambda (lam) parameter. It is often used in techniques like CutMix, which enhances data augmentation by combining different image regions.

Parameters:
1. size: A tuple that describes the shape of the image, typically (batch_size, channels, height, width).
2. lam: A float that determines the proportion of the area to be covered by the bounding box.
3. scale (default=1): A scaling factor for the image dimensions, potentially to apply the bounding box on a smaller version.
"""
def rand_bbox(size, lam, scale=1): 
    # rescale width and height
    W = size[1] // scale
    H = size[2] // scale

    # calculate cut ratio (cut_rat determines the proportion of the bounding box size relative to the entire image)
    # The 1 - lam ensures the ratio relates to the remaining uncovered area)
    cut_rat = np.sqrt(1. - lam)
    # calculate width and height bounding box
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform - select random center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # calculate bounding box coordinates
    bbx1 = np.clip(cx - cut_w // 2, 0, W)	# clip to make coordinates stay in the bounding box of dimensions W x H
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2	# the four coordinates of the vertices of the bounding box
 
"""
This class implements Class Attention, a specialized self-attention mechanism often used in vision transformers (ViTs). 
It focuses on attending to the "class token" (which represents the global image embedding) based on interactions with the other tokens.
"""
class ClassAttention(nn.Module):
    def __init__(self, dim, num_heads):
        '''
        params:
        dim: The input feature dimension (embedding size).
        num_heads: Number of attention heads for multi-head attention.
        head_dim: Dimension per attention head, computed as dim // num_heads.
        self.scale: A scaling factor to stabilize the dot-product attention, computed as 1 / sqrt(head_dim).
        '''
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads        # Dimension per attention head    
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.kv = nn.Linear(dim, dim * 2)  # For computing keys and values
        self.q = nn.Linear(dim, dim)       # For computing queries
        self.proj = nn.Linear(dim, dim)    # For projecting the output back to `dim`
        self.apply(self._init_weights)     # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        '''
        Linear Layers: Initialized with truncated normal distribution for weights (std=0.02) and biases set to zero.
        '''
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        '''
        B = Batch size
        N = Number of tokens (patch tokens + class token)
        C = Embedding dimension (dim)
        '''
        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]         # Reshape and split into two tensors: keys (k) and values (v), each with shape (B, num_heads, N, head_dim)
        q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)    # compute queries: Select the class token (first token), then reshape it to (B, num_heads, 1, head_dim) to get q 
        # compute attention scores
        attn = ((q * self.scale) @ k.transpose(-2, -1))     # dot product to get similarity between query and keys; scaling stabilizes training
        attn = attn.softmax(dim=-1)         # softmax to get probabilities
        cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)    # attn @ v: Produces the weighted sum over v, creating an aggregated class embedding; then reshape to (B, 1, dim)
        cls_embed = self.proj(cls_embed)    # Project the class embedding back to the original dim
        return cls_embed

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

"""
The ClassBlock is a core component of a Vision Transformer (ViT). 
It builds on top of the ClassAttention mechanism and adds a feedforward network (MLP) 
and normalization to further process the class token
"""
class ClassBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = ClassAttention(dim, num_heads)
        self.mlp = FFN(dim, int(dim * mlp_ratio))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels # Xavier-like initialization for conv2d layers
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        '''
        x: Shape (B, N, dim), where:
        B = Batch size
        N = Number of tokens (class token + patch tokens)
        dim = Embedding dimension of each token
        '''
        cls_embed = x[:, :1]    # Extracts the class token from x; x[:, :1] returns the first token for each batch, shape: (B, 1, dim) - (The class token is intended to act as a global representation of the image)
        cls_embed = cls_embed + self.attn(self.norm1(x))
        
        # Feedforward Network (MLP) on Class Token
        '''
        Normalize the entire input x using self.norm1(x).
        ClassAttention is applied, attending to all tokens (N tokens) using the class token.
        This step refines the class token based on its interactions with other tokens.
        Residual Connection: The original class token (cls_embed) is added back to the attention output.
        Output shape: (B, 1, dim)
        '''
        cls_embed = cls_embed + self.mlp(self.norm2(cls_embed)) 
        
        # Concatenate Class Token with Remaining Tokens
        '''
        Concatenate the updated class token with the remaining patch tokens (x[:, 1:]).
        The shape of x[:, 1:] is (B, N-1, dim).
        The final shape of the output is (B, N, dim), where N is the total number of tokens (class + patches)
        '''
        return torch.cat([cls_embed, x[:, 1:]], dim=1)


class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)   # A depthwise convolutional layer that works on spatial features (height H and width W). It enhances spatial modeling by incorporating local pixel-level information.
                                                # Unlike MLPs, which operate independently on each token, the depthwise convolution allows tokens to interact locally.
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)          # Weight Initialization

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):  # Xavier-like initialization
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_() # bias set to zero

    def forward(self, x, H, W):
        '''
        Input:
        x: Shape (B, N, in_features), where:

        B = Batch size
        N = Number of tokens (or total patches)
        in_features = Embedding dimension of each token
        H, W:

        Height and width of the image (or the feature map) corresponding to the flattened input x.
        Used for reshaping x back into its spatial form for depthwise convolution.
        '''
        
        x = self.fc1(x)             # Shape of x changes from (B, N, in_features) to (B, N, hidden_features)
        
        # Depthwise Convolution 
        '''
        Before applying the convolution, x (shape (B, N, hidden_features)) is reshaped to (B, hidden_features, H, W), where:
        N = H × W (since N is the flattened number of patches).
        The DWConv performs spatial filtering across height and width, capturing spatial dependencies.
        After the convolution, the output is flattened back to (B, N, hidden_features).
        '''
        x = self.dwconv(x, H, W)
        
        x = self.act(x)
        x = self.fc2(x)     # Shape changes from (B, N, hidden_features) to (B, N, in_features).

        return x


class MergeFFN(nn.Module):
    """
    The MergeFFN class is an extension of the PVT2FFN design, 
    adding a proxy branch to process an additional component of the input called "semantics." 
    This allows the model to process two distinct parts of the input differently and then merge them. 
    """
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)   # depthwise convolution
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

        # Proxy Path (Semantic Tokens):
        self.fc_proxy = nn.Sequential(
            nn.Linear(in_features, 2*in_features),
            nn.GELU(),
            nn.Linear(2*in_features, in_features),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x, semantics = torch.split(x, [H*W, x.shape[1] - H*W], dim=1)   # The input x is split into:    
                                                                        #    x (Feature Tokens): First H*W tokens (corresponding to the spatial structure of the image).
                                                                        #    semantics (Semantic Tokens): remaning tokens
        semantics = self.fc_proxy(semantics)
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        x = torch.cat([x, semantics], dim=1)    # feature and semantic tokens are merged after being processed differently
        return x            # Returns a tensor of shape (B, N, in_features); N = feature tokens + semantic tokens 

class Attention(nn.Module):
    def __init__(self, dim, num_heads):         #dim: Total dimension of the input embeddings.
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads             # Each attention head operates on a subspace of size head_dim
        self.scale = head_dim ** -0.5           # scaling normalization

        self.q = nn.Linear(dim, dim)            # (B, N, dim) → (B, N, dim)
        self.kv = nn.Linear(dim, dim * 2)       # Jointly maps input to keys (K) and values (V), shape (B, N, dim) → (B, N, 2 * dim).
        self.proj = nn.Linear(dim, dim)         # Projects the concatenated attention output back to the input space, shape (B, N, dim) → (B, N, dim).
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape   # x: Input tensor of shape (B, N, C), B=batch_size, N=number_of_token, C=embedding dimension
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)            # q is reshaped to (B, N, num_heads, head_dim) and permuted to (B, num_heads, N, head_dim        
        
        # Key and Value Projections:                                                                                               # This structure allows each head to operate independently on its respective subspace  
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)   # kv is reshaped to separate keys (K) and values (V):
        k, v = kv[0], kv[1]                 # Shape of k and v: (B, num_heads, N, head_dim).   
        # Scaled Dot-Product Attention:
        attn = (q @ k.transpose(-2, -1)) * self.scale   # Compute attention scores by matrix multiplying queries (Q) with the transposed keys (K^T), followed by scaling
        attn = attn.softmax(dim=-1)                     # Apply softmax along the token dimension
        # Weighted Summation:
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # Multiply attention scores with values (V) to compute the weighted representation. Reshape and transpose back to (B, N, C).
        x = self.proj(x)            # Apply the final linear transformation to return to the original embedding space
        return x

class DualAttention(nn.Module):
    """
    Key Components

    Main Attention Path:
        q: Linear layer to project inputs into queries (Q).
        kv: Linear layer to project inputs into keys (K) and values (V).
        proj: Final linear projection after attention.

    Semantics Path:
        q_proxy: Linear layer for projecting the semantics into queries.
        kv_proxy: Linear layer for keys and values specific to the semantics pathway.
        mlp_proxy: Multi-layer perceptron to process the updated semantics.
        q_proxy_ln, p_ln, proxy_ln: Layer normalization layers for normalization before and after transformations.
    """
    def __init__(self, dim, num_heads, drop_path=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.q_proxy = nn.Linear(dim, dim)
        self.kv_proxy = nn.Linear(dim, dim * 2)
        self.q_proxy_ln = nn.LayerNorm(dim)

        self.p_ln = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()   #  Dropout probability for the stochastic depth operation

        self.mlp_proxy = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * dim, dim),
        )
        self.proxy_ln = nn.LayerNorm(dim)

        self.qkv_proxy = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim*3)
        )

        layer_scale_init_value = 1e-6
        # Learnable Scale Parameters:
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma3 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def selfatt(self, semantics):
        B, N, C = semantics.shape       # B: Batch size. N: Number of tokens (e.g., patches in vision tasks). C: Embedding dimension.
                                        # semantics: Auxiliary semantics tensor with shape (B, N_p, C). N_p: Number of semantics tokens.
        qkv = self.qkv_proxy(semantics).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)   #
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        semantics = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return semantics

    def forward(self, x, H, W, semantics):
        semantics = semantics + self.drop_path(self.gamma1 * self.selfatt(semantics))   # A self-attention operation (selfatt) is applied to the semantics tensor, and the result is added back (residual connection)
        B, N, C = x.shape
        B_p, N_p, C_p = semantics.shape

        # Cross-Attention (Semantics → Input):
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)    # Queries (Q) are derived from semantics.
        q_semantics = self.q_proxy(self.q_proxy_ln(semantics)).reshape(B_p, N_p, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv_semantics = self.kv_proxy(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)   # Keys (K) and values (V) are derived from the input (x).
        kp, vp = kv_semantics[0], kv_semantics[1]
        attn = (q_semantics @ kp.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        _semantics = (attn @ vp).transpose(1, 2).reshape(B, N_p, C) * self.gamma2
        semantics = semantics + self.drop_path(_semantics)
        semantics = semantics + self.drop_path(self.gamma3 * self.mlp_proxy(self.p_ln(semantics)))      # The semantics tensor is processed through a feedforward MLP block, normalized, and added back.

        # Cross-Attention (Input → Semantics):
        kv = self.kv(self.proxy_ln(semantics)).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, semantics
              
# This module integrates attention mechanisms, feedforward layers, and normalization in a residual block.
class MergeBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(dim, num_heads)
        self.mlp = MergeFFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        """
        Input x is normalized using norm1.
        Normalized input is passed through the attention mechanism (self.attn), which considers spatial dimensions (H, W).
        Output is scaled by gamma1 and added back to the input (residual connection). 
        """
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x), H, W))
        """
        Input x is normalized using norm1.
        Normalized input is passed through the attention mechanism (self.attn), which considers spatial dimensions (H, W).
        Output is scaled by gamma1 and added back to the input (residual connection).
        """
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x), H, W))
        return x

class DualBlock(nn.Module):
    """
    The DualBlock is an advanced module designed to process and refine features using a dual mechanism 
    that integrates dual attention and feedforward refinement, making it suitable for hierarchical or multi-modal transformers. 
    It operates on two streams of input: the primary feature x and the semantic features semantics.
    """
    def __init__(self, dim, num_heads, mlp_ratio, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        #  mlp_ratio=Ratio for the hidden dimensionality in the feedforward network.
        self.norm1 = norm_layer(dim)    # Applied before the attention block.
        self.norm2 = norm_layer(dim)    # Applied before the feedforward block.

        self.attn = DualAttention(dim, num_heads, drop_path=drop_path)  # Multi-head attention mechanism (defined in the Attention class).
        self.mlp = PVT2FFN(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()   #Stochastic depth for regularization.
        
        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, semantics): 
        """
        x: Primary feature tensor with shape (B, N, C), where:

            B: Batch size.
            N: Number of tokens.
            C: Embedding dimension.

        H, W: Spatial dimensions of the input.
        semantics: Semantic feature tensor with shape (B, N_sem, C)
        """
        _x, semantics = self.attn(self.norm1(x), H, W, semantics)   # The normalized x and semantics are processed by the DualAttention module.
        x = x + self.drop_path(self.gamma1 * _x)        # The attention-refined output _x is scaled by gamma1 and added back to the original input x (residual connection).
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x), H, W))     # The normalized x is passed through the feedforward network (self.mlp), which incorporates positional information using (H, W).
                                                                                # The FFN output is scaled by gamma2 and added back to x (residual connection).
        return x, semantics     # x: Refined primary feature tensor (B, N, C).
                                # semantics: Updated semantic feature tensor (B, N_sem, C).

class DownSamples(nn.Module):
    """
    The DownSamples class is a module designed to downsample feature maps in a hierarchical model, particularly in vision transformer architectures. 
    It achieves dimensionality reduction and normalization, preparing features for subsequent processing layers.
    """
    def __init__(self, in_channels, out_channels):
#         in_channels: Number of input channels in the feature map.
#         out_channels: Number of output channels after the convolutional layer.
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)    # This reduces the spatial resolution by half and increases the channel dimension to out_channels.
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """
        x: Input feature tensor with shape (B, C_in, H_in, W_in):

        B: Batch size.
        C_in: Number of input channels.
        H_in, W_in: Spatial dimensions of the input.
        """
        x = self.proj(x)            # Reduces the spatial dimensions of x from (H_in, W_in) to approximately (H_in / 2, W_in / 2) (due to stride=2).
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)    # Flattening:
                                            #     The feature map is reshaped to combine the spatial dimensions into a single dimension.
                                            #     Resulting shape: (B, C_out, H_out * W_out).

                                            # Transposing:
                                            #     The channels are moved to the last dimension to match the input format expected by LayerNorm.
                                            #     Resulting shape: (B, H_out * W_out, C_out).
        
        x = self.norm(x)
        return x, H, W

class Stem(nn.Module): # TODO
    """
    The Stem class is a module designed as the initial feature extraction layer in a vision model, 
    particularly for hierarchical or transformer-based architectures. 
    It consists of multiple convolutional layers followed by projection and normalization, preparing the input image for subsequent processing.
    """
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2,
                      padding=3, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )   # Reduce the spatial resolution via a stride of 2 in the first layer. Keep the resolution constant in subsequent layers.
        self.proj = nn.Conv2d(hidden_dim,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)    
        self.norm = nn.LayerNorm(out_channels)  # A convolutional layer with a stride of 2 that reduces spatial resolution and projects the feature map to out_channels.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """
        x: Input image tensor with shape (B, C_in, H_in, W_in):

        B: Batch size.
        C_in: Number of input channels (e.g., 3 for RGB images).
        H_in, W_in: Spatial dimensions of the input image.
        """
        x = self.conv(x)
        x = self.proj(x)
        _, _, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)
        # Flattening:

        #     Combines the spatial dimensions (H_proj, W_proj) into a single dimension.
        #     Resulting shape: (B, C_out, H_proj * W_proj).

        # Transposing:

        #     Rearranges the dimensions for Layer Normalization.
        #     Resulting shape: (B, H_proj * W_proj, C_out).
        x = self.norm(x)    # Normalizes the feature tensor along the channel dimension (C_out).
        return x, H, W

class SemanticEmbed(nn.Module):
    """
    The SemanticEmbed class is a module designed to embed high-level semantic features into a transformed representation. 
    It consists of a simple linear transformation followed by normalization, making it suitable for tasks 
    where semantic features need to be projected into a compatible space for downstream processing.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj_proxy = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, semantics):
        semantics = self.proj_proxy(semantics)  # Projects the input feature vector into the out_channels dimensional space.
        return semantics

class DualViT(nn.Module):
    def __init__(self, 
        in_chans = 3, # Number of input channels (e.g., 3 for RGB images).
        num_classes = 1000, # Number of output classes for classification.
        stem_hidden_dim = 32,   # Hidden dimension size for the initial stem layer.
        embed_dims = [64, 128, 320, 448],   # List of embedding dimensions for each stage of the model.
        num_heads = [2, 4, 10, 14], # List of attention heads for each stage. 
        mlp_ratios = [8, 8, 4, 3],  # Expansion ratios for the Multi-Layer Perceptron (MLP) layers.
        drop_path_rate=0., # Dropout rate for stochastic depth.
        norm_layer=nn.LayerNorm,    # Normalization layer
        depths=[3, 4, 6, 3], # Number of layers (blocks) in each stage.
        num_stages=4, # Total number of stages in the network.
        token_label=True,   # Whether token labeling is enabled for dense prediction.
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        self.sep_stage = 2

        # This generates an array of dropout probabilities for each layer. The probabilities are linearly spaced from 0 to drop_path_rate.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        
        """
        The model is divided into num_stages (e.g., 4 stages), each with:

        Patch Embedding: Converts image patches to feature embeddings.
            Stage 0 uses a Stem module.
            Other stages use DownSamples for downsampling the feature map size and increasing embedding dimensions.
        Proxy Semantics: Captures semantic features at stage 0 using:
            self.q: A learnable query embedding for attention.
            self.pool: A pooling layer to downsample feature maps.
            self.kv: A linear layer to create key and value embeddings for attention.
            self.scale: Scaling factor for attention computation.
            self.se: A squeeze-and-excitation block to modulate feature importance.
        Blocks:
            DualBlock is used in earlier stages.
            MergeBlock is used in later stages to integrate features from different streams.
        """
        for i in range(num_stages):
            if i == 0:
                patch_embed = Stem(in_chans, stem_hidden_dim, embed_dims[i])
            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i])

            if i == 0:
                self.q = nn.Parameter(torch.empty((64, embed_dims[0])), requires_grad=True)
                self.q_embed = nn.Sequential(
                    nn.LayerNorm(embed_dims[0]),
                    nn.Linear(embed_dims[0], embed_dims[0])
                )
                self.pool = nn.AvgPool2d((7,7), stride=7)
                self.kv = nn.Linear(embed_dims[0], 2*embed_dims[0])
                self.scale = embed_dims[0] ** -0.5
                self.proxy_ln = nn.LayerNorm(embed_dims[0])
                self.se = nn.Sequential(
                    nn.Linear(embed_dims[0], embed_dims[0]),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dims[0], 2*embed_dims[0])
                )
                trunc_normal_(self.q, std=.02)
            else:
                semantic_embed = SemanticEmbed(
                    embed_dims[i - 1], embed_dims[i]
                )
                setattr(self, f"proxy_embed{i + 1}", semantic_embed)

            if i >= self.sep_stage:
                block = nn.ModuleList([
                    MergeBlock(
                        dim=embed_dims[i], 
                        num_heads=num_heads[i], 
                        mlp_ratio=mlp_ratios[i]-1 if (j%2!=0 and i==2) else mlp_ratios[i],  
                        drop_path=dpr[cur + j], 
                        norm_layer=norm_layer)
                for j in range(depths[i])])
            else:
                block = nn.ModuleList([
                    DualBlock(
                        dim=embed_dims[i], 
                        num_heads=num_heads[i], 
                        mlp_ratio=mlp_ratios[i],
                        drop_path=dpr[cur + j], 
                        norm_layer=norm_layer)
                for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            norm_proxy = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

            if i != num_stages - 1:
                setattr(self, f"norm_proxy{i + 1}", norm_proxy)

        post_layers = ['ca']

        # this Processes the combined feature representations after the main stages.
        self.post_network = nn.ModuleList([
            ClassBlock(
                dim = embed_dims[-1], 
                num_heads = num_heads[-1], 
                mlp_ratio = mlp_ratios[-1],
                norm_layer=norm_layer
            )
            for i in range(len(post_layers))
        ])

        # classification head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()    # mapsthe final embedding to num_classes.

        ##################################### token_label #####################################
        """
        This is an optional feature for dense prediction tasks:

        self.aux_head: An auxiliary classification head for token-level outputs.
        self.mix_token: Implements token mixing for augmenting input features during training.
        self.beta: Parameter for the Beta distribution used in token mixing.
        """
        self.return_dense = token_label
        self.mix_token = token_label
        self.beta = 1.0
        self.pooling_scale = 8
        if self.return_dense:
            self.aux_head = nn.Linear(
                embed_dims[-1],
                num_classes) if num_classes > 0 else nn.Identity()
        ##################################### token_label #####################################

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_cls(self, x):
        B, N, C = x.shape
        cls_tokens = x.mean(dim=1, keepdim=True)  #self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.post_network:
            x = block(x)
        return x

    def forward_last(self, x, tokenlabeling=False):
        if tokenlabeling:
            x = self.forward_cls(x)
        else:
            x = self.forward_cls(x)[:, 0]
        norm = getattr(self, f"norm{self.num_stages}")
        x = norm(x)
        return x

    def forward_sep(self, x, tokenlabeling=False, H=0, W=0):
        B = x.shape[0]
        for i in range(self.sep_stage):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")

            if tokenlabeling == False or i != 0:
                x, H, W = patch_embed(x)
            else:
                x = x.view(B, H*W, -1)
            C = x.shape[-1]
            if i == 0:
                x_down = self.pool(x.reshape(B, H, W, C).permute(0, 3, 1, 2))
                x_down_H, x_down_W = x_down.shape[2:]
                x_down = x_down.view(B, C, -1).permute(0, 2, 1)
                kv = self.kv(x_down).view(B, -1,  2, C).permute(2, 0, 1, 3)
                k, v = kv[0], kv[1]  # B, N, C
                
                if x_down.shape[1] == self.q.shape[0]:
                    self_q = self.q
                else:
                    self_q = self.q.reshape(8, 8, -1).permute(2, 0, 1)
                    self_q = F.interpolate(self_q.unsqueeze(0), size=(x_down_H, x_down_W), mode='bicubic').squeeze(0).permute(1, 2, 0)
                    self_q = self_q.reshape(-1, self_q.shape[-1])
                
                attn = (self.q_embed(self_q) @ k.transpose(-1, -2)) * self.scale   # q: 1, M, C,   k: B, N, C -> B, M, N
                attn = attn.softmax(-1)  # B, M, N
                semantics = attn @ v   # B, M, C
                semantics = semantics.view(B, -1, C)

                semantics = torch.cat([semantics.unsqueeze(2), x_down.unsqueeze(2)], dim=2)
                se = self.se(semantics.sum(2).mean(1))
                se = se.view(B, 2, C).softmax(1)
                semantics = (semantics * se.unsqueeze(1)).sum(2)
                semantics = self.proxy_ln(semantics)
            else:
                semantics_embed = getattr(self, f"proxy_embed{i + 1}")
                semantics = semantics_embed(semantics)

            for blk in block:
                x, semantics = blk(x, H, W, semantics)

            norm = getattr(self, f"norm{i + 1}")
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            norm_semantics = getattr(self, f"norm_proxy{i + 1}")
            semantics = norm_semantics(semantics)
                
        return x, semantics

    def forward_merge(self, x, semantics, tokenlabeling=False):
        B = x.shape[0]
        for i in range(self.sep_stage, self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, H, W = patch_embed(x)

            semantics_embed = getattr(self, f"proxy_embed{i + 1}")
            semantics = semantics_embed(semantics)
            
            x = torch.cat([x, semantics], dim=1)
            for blk in block:
                x = blk(x, H, W)

            semantics = x[:, H*W:]
            x = x[:, 0:H*W]
            
            if i != self.num_stages - 1:
                norm = getattr(self, f"norm{i + 1}")
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

                norm_semantics = getattr(self, f"norm_proxy{i + 1}")
                semantics = norm_semantics(semantics)
  
        if tokenlabeling:
            return torch.cat([x, semantics], dim=1), H, W
        else:
            return torch.cat([x, semantics], dim=1)

    def forward(self, x):
        if not self.return_dense:
            x, semantics = self.forward_sep(x)  # Separates the input into hierarchical features using patch embedding and proxy semantics.
            x = self.forward_merge(x, semantics)    # Merges features across stages while combining information from earlier stages.
            x = self.forward_last(x)    # Processes the final stage's features for classification.
            x = self.head(x)
            return x
        else:
            """
            When token labeling is enabled:

            forward_embeddings: Extracts patch embeddings.
            mix_token: Augments tokens using random mixing.
            forward_cls: Computes class tokens for dense prediction.
            """
            x, H, W = self.forward_embeddings(x)
            # mix token, see token labeling for details.
            if self.mix_token and self.training:
                lam = np.random.beta(self.beta, self.beta)
                patch_h, patch_w = x.shape[1] // self.pooling_scale, x.shape[
                    2] // self.pooling_scale
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam, scale=self.pooling_scale)
                temp_x = x.clone()
                sbbx1,sbby1,sbbx2,sbby2=self.pooling_scale*bbx1,self.pooling_scale*bby1,\
                                        self.pooling_scale*bbx2,self.pooling_scale*bby2
                temp_x[:, sbbx1:sbbx2, sbby1:sbby2, :] = x.flip(0)[:, sbbx1:sbbx2, sbby1:sbby2, :]
                x = temp_x
            else:
                bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0
            
            x, semantics = self.forward_sep(x, True, H, W)
            x, H, W = self.forward_merge(x, semantics, True)
            x = self.forward_last(x, tokenlabeling=True)

            x_cls = self.head(x[:, 0])
            x_aux = self.aux_head(
                x[:, 1:1+H*W]
            )  # generate classes in all feature tokens, see token labeling

            if not self.training:
                return x_cls + 0.5 * x_aux.max(1)[0]

            if self.mix_token and self.training:  # reverse "mix token", see token labeling for details.
                x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])

                temp_x = x_aux.clone()
                temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
                x_aux = temp_x

                x_aux = x_aux.reshape(x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1])

            return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)

    def forward_embeddings(self, x):
        patch_embed = getattr(self, f"patch_embed{0 + 1}")
        x, H, W = patch_embed(x)
        x = x.view(x.size(0), H, W, -1)
        return x, H, W
# This is a depthwise convolution layer used for spatial feature refinement:
#    Takes feature maps of shape (B, N, C) and applies 2D convolutions.
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

@register_model
def dualvit_s(pretrained=False, **kwargs):
    model = DualViT(
        stem_hidden_dim = 32,
        embed_dims = [64, 128, 320, 448], 
        num_heads = [2, 4, 10, 14], 
        mlp_ratios = [8, 8, 4, 3, 2],
        norm_layer = partial(nn.LayerNorm, eps=1e-6), 
        depths = [3, 4, 6, 3], 
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def dualvit_b(pretrained=False, **kwargs):
    model = DualViT(
        stem_hidden_dim = 64,
        embed_dims = [64, 128, 320, 512], 
        num_heads = [2, 4, 10, 16], 
        mlp_ratios = [8, 8, 4, 3, 2],
        norm_layer = partial(nn.LayerNorm, eps=1e-6), 
        depths = [3, 4, 15, 3], 
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def dualvit_l(pretrained=False, **kwargs):
    model = DualViT(
        stem_hidden_dim = 64,
        embed_dims = [96, 192, 384, 512], 
        num_heads = [3, 6, 12, 16], 
        mlp_ratios = [8, 8, 4, 3, 2],
        norm_layer = partial(nn.LayerNorm, eps=1e-6), 
        depths = [3, 6, 21, 3], 
        **kwargs)
    model.default_cfg = _cfg()
    return model

"""
Key Components for DualViT:

    Patch Embedding: Converts images into embeddings.
    Proxy Semantics: Captures semantic representations for tokens.
    Hierarchical Blocks: Processes features through Dual and Merge blocks.
    Auxiliary Head: Enables dense prediction for token labeling tasks.
    Classification Head: Final layer for output logits.
"""