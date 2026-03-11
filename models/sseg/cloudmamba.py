import math
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import einsum, rearrange, repeat
from natten.functional import na2d

try:
    from fvcore.nn import flop_count, parameter_count
except ImportError:
    flop_count = parameter_count = None

try:
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
except ImportError:
    IMAGENET_DEFAULT_MEAN = IMAGENET_DEFAULT_STD = None

try:
    from timm.models.pvt_v2 import DropPath, to_2tuple, register_model
except ImportError:
    from timm.models.layers import DropPath, to_2tuple
    register_model = lambda fn: fn



def rotate_every_two(x):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


def toodd(size):
    size = list(to_2tuple(size))
    if size[0] % 2 == 0:
        size[0] = size[0] + 1
    if size[1] % 2 == 0:
        size[1] = size[1] + 1
    return tuple(size)

# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_C=True, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    if with_C:
        flops = 9 * B * L * D * N
    else:
        flops = 7 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops


def selective_scan_state_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_C=False, with_D=False, with_Z=False)
    return flops


def selective_scan_state_fn(u, delta, A, B, D=None, z=None, delta_bias=None,
                            delta_softplus=False, return_last_state=False):
    """Pure-PyTorch selective scan returning the state trajectory h of shape [B, D, N, L].

    Uses a vectorised log-cumsum trick (O(L) ops, fully parallel on GPU).
    Works for any d_state N since the SSM diagonal-A assumption decouples each state dim.
    """
    B_batch, d_model, L = u.shape
    N = A.shape[1]

    u_f     = u.float()
    delta_f = delta.float()
    A_f     = A.float()
    B_f     = B.float()   # [B, N, L]

    if delta_bias is not None:
        delta_f = delta_f + delta_bias.float().unsqueeze(-1)
    if delta_softplus:
        delta_f = F.softplus(delta_f)

    # ZOH discretisation
    # dA: [B, D, N, L]   dA_t = exp(delta_t * A)
    dA    = torch.exp(delta_f.unsqueeze(2) * A_f.unsqueeze(0).unsqueeze(-1))
    # dB_u: [B, D, N, L]   dB_u_t = delta_t * B_t * u_t
    dB_u  = (delta_f.unsqueeze(2)
             * B_f.unsqueeze(1)
             * u_f.unsqueeze(2))

    # Vectorised scan via log-cumsum:  h_t = P_t * cumsum(dB_u / P, dim=-1)
    #   where P_t = prod_{i=1}^{t} dA_i  (log-space for stability)
    EPS   = 1e-9
    log_P = torch.cumsum(torch.log(dA.clamp(min=EPS)), dim=-1)   # [B, D, N, L]
    P     = torch.exp(log_P)
    hs    = P * torch.cumsum(dB_u / P.clamp(min=EPS), dim=-1)    # [B, D, N, L]

    hs = hs.to(u.dtype)
    if return_last_state:
        return hs, hs[..., -1]
    return hs



class RoPE(nn.Module):

    def __init__(self, embed_dim, num_heads):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    
    def forward(self, slen):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        # index = torch.arange(slen[0]*slen[1]).to(self.angle)
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        # sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
        # sin = sin.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        sin = torch.cat([sin_h, sin_w], -1) #(h w d1)
        # cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
        # cos = cos.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        cos = torch.cat([cos_h, cos_w], -1) #(h w d1)

        return (sin, cos)


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5, enable_bias=True):
        super().__init__()
        
        self.dim = dim
        self.init_value = init_value
        self.enable_bias = enable_bias
          
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value, requires_grad=True)
        if enable_bias:
            self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x
    
    def extra_repr(self) -> str:
        return '{dim}, init_value={init_value}, bias={enable_bias}'.format(**self.__dict__)
    

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels):
        super().__init__(num_groups=1, num_channels=num_channels, eps=1e-6)


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()


class CloudSSM(nn.Module):
    '''
    Attention-augmented SSM layer
    '''
    def __init__(
        self,
        d_model,
        d_state=4,
        d_conv=5,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0,
        device=None,
        dtype=None,
        num_heads=1,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv//2, groups=self.d_inner)
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self.D_init(self.d_inner, dt_init)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, bias=True,**factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError

        return dt_proj


    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init=="random" or "constant":
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init=="simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init=="zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log


    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init=="random" or "constant":
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D) 
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D


    def forward_ssm(self, x, state_params=None):
        
        B, C, H, W = x.shape
        L = H * W

        xs = x.view(B, -1, L)
        
        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, C), xs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.matmul(self.dt_projs_weight.view(1, C, -1), dts)
        
        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias

        # state: [B, D, N, L]
        state = selective_scan_state_fn(xs, dts, As, Bs, None, z=None,
                                        delta_bias=dt_projs_bias,
                                        delta_softplus=True,
                                        return_last_state=False)

        with torch.amp.autocast('cuda', enabled=False):
            state = state.to(torch.float32)
            Cs    = Cs.to(torch.float32)
            xs_f  = xs.to(torch.float32)
            # [B,D,N,L] x [B,N,L] -> [B,D,L]  (correct contraction over state dim)
            x = einsum(state, Cs, 'b d n l, b n l -> b d l')
            x = x + xs_f * Ds.view(-1, 1)

        return x
    

    def forward(self, x, state_params=None):
        
        B, C, H, W = x.shape

        x = self.act(self.conv2d(x))
        x = self.forward_ssm(x, state_params=state_params)

        x = rearrange(x, 'b d (h w) -> b d h w', h=H, w=W)
        if self.dropout is not None:
            x = self.dropout(x)

        return x



class Attention(nn.Module):
    def __init__(self, 
                 embed_dim=64, 
                 num_heads=2, 
                 window_size=7, 
                 window_dilation=1, 
                 global_mode=False, 
                 image_size=None, 
                 use_rpb=False, 
                 sr_ratio=1):
        
        super().__init__()
        window_size = to_2tuple(window_size)
        window_dilation = to_2tuple(window_dilation)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.window_dilation = window_dilation
        self.global_mode = global_mode
        self.sr_ratio = sr_ratio
        self.image_size = image_size
        
        self.qkv = nn.Conv2d(embed_dim, embed_dim*3, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )
        
        self.lepe = nn.Conv2d(embed_dim, embed_dim, kernel_size=5, padding=2, groups=embed_dim)
        self.norm = LayerNorm2d(embed_dim)
        self.proj = nn.Sequential(
            LayerNorm2d(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
        
        if not global_mode:
            self.ssm = CloudSSM(d_model=embed_dim, num_heads=num_heads)
            
        if use_rpb:
            rpb_list = [nn.Parameter(torch.empty(num_heads//2, (2 * window_size[0] - 1), (2 * window_size[1] - 1)), requires_grad=True)] * 2
            if global_mode: 
                rpb_list.append(nn.Parameter(torch.empty(1, num_heads, image_size[0]*image_size[1], image_size[0]*image_size[1]), requires_grad=True))
            self.rpb = nn.ParameterList(rpb_list)
            
        self.reset_parameters()
        

    def reset_parameters(self):
        if hasattr(self, 'rpb'):
            for item in self.rpb:
                nn.init.zeros_(item)
    

    def forward(self, x, pos_enc):
        
        B, C, H, W = x.shape
        dilation = [int(H/self.window_size[0]), int(W/self.window_size[1])]
        dilation = [min(dilation[0], self.window_size[0]), min(dilation[1], self.window_size[1])]
        
        gate = self.gate(x)
        qkv = self.qkv(x)
        lepe = self.lepe(qkv[:, -C:, ...])

        # natten 0.21+ expects (B, H, W, heads, head_dim)
        q, k, v = rearrange(qkv, 'b (m n c) h w -> m b h w n c', m=3, n=self.num_heads)

        sin, cos = pos_enc
        # theta_shift needs (b n h w c) format → permute, shift, permute back
        q_bnhwc = q.permute(0, 3, 1, 2, 4)
        k_bnhwc = k.permute(0, 3, 1, 2, 4)
        q_bnhwc = theta_shift(q_bnhwc, sin, cos) * self.scale
        k_bnhwc = theta_shift(k_bnhwc, sin, cos)
        q = q_bnhwc.permute(0, 2, 3, 1, 4)   # (B, H, W, n, c)
        k = k_bnhwc.permute(0, 2, 3, 1, 4)

        q1, q2 = torch.chunk(q, chunks=2, dim=3)
        k1, k2 = torch.chunk(k, chunks=2, dim=3)
        v1, v2 = torch.chunk(v, chunks=2, dim=3)

        # na2d takes (B, H, W, heads, head_dim), dilation as (h, w) tuple
        dil_tuple = tuple(self.window_dilation) if not isinstance(self.window_dilation, tuple) else self.window_dilation
        dil2_tuple = tuple(dilation)

        v1 = na2d(q1, k1, v1,
                  kernel_size=toodd(self.window_size),
                  dilation=dil_tuple)
        v2 = na2d(q2, k2, v1 + v2,
                  kernel_size=toodd(self.window_size),
                  dilation=dil2_tuple)

        x = torch.cat([v1, v2], dim=3)                      # (B, H, W, n, c)
        x = rearrange(x, 'b h w n c -> b (n c) h w')

        if not self.global_mode:
            x = self.ssm(x, state_params=None)
        else:
            
            q = rearrange(q, 'b n h w c -> b n (h w) c')
            k = rearrange(k, 'b n h w c -> b n (h w) c')
            v = rearrange(x, 'b (n c) h w -> b n (h w) c', n=self.num_heads, h=H, w=W)
            attn = einsum(q, k, 'b n l c, b n m c -> b n l m')
            
            if hasattr(self, 'rpb'):
                if self.rpb[-1].shape[2:] != attn.shape[2:]:
                    rpb = F.interpolate(self.rpb[-1], size=attn.shape[2:], mode='bicubic', align_corners=False)
                    attn = attn + rpb
                else:
                    attn = attn + self.rpb[-1]
                
            attn = torch.softmax(attn, dim=-1)  
            x = einsum(attn, v, 'b n l m, b n m c -> b n c l').reshape(B, -1, H, W)

        x = self.norm(x) + lepe

        with torch.amp.autocast('cuda', enabled=False):
            # disable amp for better training stability
            x = x.to(torch.float32)
            gate = gate.to(torch.float32)
            x = x * gate
            x = self.proj(x)
        
        return x


class FFN(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        act_layer=nn.GELU,
        dropout=0): 
        super().__init__()

        self.fc1 = nn.Conv2d(embed_dim, ffn_dim, kernel_size=1)
        self.act_layer = act_layer()
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, kernel_size=3, padding=1, groups=ffn_dim)
        self.fc2 = nn.Conv2d(ffn_dim, embed_dim, kernel_size=1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.act_layer(x)
        x = x + self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x



class Block(nn.Module):
    def __init__(self,
                 image_size=None,
                 embed_dim=64,
                 num_heads=2, 
                 window_size=7,
                 window_dilation=1,
                 global_mode=False,
                 use_rpb=False,
                 sr_ratio=1,
                 ffn_dim=256, 
                 drop_path=0, 
                 layerscale=False,
                 resscale=False,
                 layer_init_values=1e-6,
                 token_mixer=Attention,
                 channel_mixer=FFN,
                 norm_layer=LayerNorm2d):
        super().__init__()
        
        self.layerscale = layerscale
        self.resscale = resscale

        self.cpe1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.norm1 = norm_layer(embed_dim)
        self.token_mixer = token_mixer(embed_dim, num_heads, window_size, window_dilation, global_mode, image_size, use_rpb, sr_ratio)
        self.cpe2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = channel_mixer(embed_dim, ffn_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        if layerscale or resscale:
            self.layer_scale1 = LayerScale(embed_dim, init_value=layer_init_values)
            self.layer_scale2 = LayerScale(embed_dim, init_value=layer_init_values)
        else:
            self.layer_scale1 = nn.Identity()
            self.layer_scale2 = nn.Identity()

    def forward(self, x, pos_enc):
        if self.resscale:
            x = x + self.cpe1(x)
            x = self.layer_scale1(x) + self.drop_path(self.token_mixer(self.norm1(x), pos_enc))
            x = x + self.cpe2(x)
            x = self.layer_scale2(x) + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.cpe1(x)
            x = x + self.drop_path(self.layer_scale1(self.token_mixer(self.norm1(x), pos_enc)))
            x = x + self.cpe2(x)
            x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))
        return x
    

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self,
                 image_size=None,
                 embed_dim=64, 
                 depth=4, 
                 num_heads=4,
                 window_size=7,
                 window_dilation=1,
                 global_mode=False,
                 use_rpb=False,
                 sr_ratio=1,
                 ffn_dim=96, 
                 drop_path=0,
                 layerscale=False,
                 resscale=False,
                 layer_init_values=1e-6,
                 norm_layer=LayerNorm2d,
                 use_checkpoint=0,
            ):

        super().__init__()
        
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.rope = RoPE(embed_dim, num_heads)
        
        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(embed_dim=embed_dim,
                          num_heads=num_heads,
                          window_size=window_size,
                          window_dilation=window_dilation,
                          global_mode=global_mode,
                          ffn_dim=ffn_dim,
                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                          layerscale=layerscale,
                          resscale=resscale,
                          layer_init_values=layer_init_values,
                          norm_layer=norm_layer,
                          image_size=image_size,
                          use_rpb=use_rpb,
                          sr_ratio=sr_ratio,
            )
            self.blocks.append(block)

    def forward(self, x):
        pos_enc = self.rope((x.shape[2:]))
        for i, blk in enumerate(self.blocks):
            if i < self.use_checkpoint and x.requires_grad:
                x = checkpoint.checkpoint(blk, x, pos_enc, use_reentrant=False)
            else:
                x = blk(x, pos_enc)
        return x


def stem(in_chans=3, embed_dim=96):
    return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim)
        )


class CloudMamba(nn.Module):
    def __init__(self,
                 image_size=224,
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dims=[64, 128, 256, 512],
                 depths=[2, 2, 6, 2],
                 num_heads=[2, 4, 8, 16],
                 window_size=[7, 7, 7, 7],
                 window_dilation=[1, 1, 1, 1],
                 use_rpb=False,
                 sr_ratio=[8, 4, 2, 1],
                 mlp_ratios=[4, 4, 4, 4], 
                 drop_rate=0,
                 drop_path_rate=0,
                 projection=1024,
                 layerscales=[False, False, False, False],
                 resscales=[False, False, False, False],
                 layer_init_values=[1, 1, 1, 1],
                 norm_layer=LayerNorm2d,
                 return_features=False,
                 use_checkpoint=[0, 0, 0, 0]):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios
        self.return_features = return_features

        # split image into non-overlapping patches
        self.patch_embed = stem(in_chans=in_chans, embed_dim=embed_dims[0])

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # input resolution
        image_size = to_2tuple(image_size)
        image_size = [(image_size[0]//2**(i+2), image_size[1]//2**(i+2)) for i in range(4)]
        
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                embed_dim=embed_dims[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                window_dilation=window_dilation[i_layer],
                global_mode=(i_layer==3),
                use_rpb=use_rpb,
                sr_ratio=sr_ratio[i_layer],
                ffn_dim=int(mlp_ratios[i_layer]*embed_dims[i_layer]),
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                layerscale=layerscales[i_layer],
                resscale=resscales[i_layer],
                layer_init_values=layer_init_values[i_layer],
                norm_layer=norm_layer,
                image_size=image_size[i_layer],
                use_checkpoint=use_checkpoint[i_layer],
            )
                       
            downsample = nn.Sequential(
                nn.Conv2d(embed_dims[i_layer], embed_dims[i_layer+1], kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dims[i_layer+1])
            ) if (i_layer < self.num_layers - 1) else nn.Identity()
            
            self.layers.append(layer)
            self.layers.append(downsample)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(self.num_features, projection, kernel_size=1),
            nn.BatchNorm2d(projection),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(projection, num_classes, kernel_size=1) if num_classes > 0 else nn.Identity(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        
        x = self.patch_embed(x)
        out = []
        idx = 0
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i%2 == 0:
                out.append(x)
                idx += 1

        return tuple(out)

    def forward(self, x):
        if self.return_features:
            return self.forward_features(x)
        else:
            x = self.patch_embed(x)
            for layer in self.layers:
                x = layer(x)
            x = self.classifier(x).flatten(1)
        return x
    
    def flops(self, shape=(3, 224, 224)):
        
        supported_ops={
            "prim::PythonOp.SelectiveScanStateFn": selective_scan_state_flop_jit,
        }

        model = copy.deepcopy(self)
        
        if torch.cuda.is_available:
            model.cuda()
        model.eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return (sum(Gflops.values()) * 1e9, params)



def _cfg(url=None, **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'classifier': 'classifier',
        **kwargs,
    }


# ─── Segmentation Decoder ─────────────────────────────────────────────────────

class _ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel recalibration."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.GELU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        return x * self.fc(w).view(b, c, 1, 1)


class _UpBlock(nn.Module):
    """Bilinear 2× upsample + skip concat + double conv + SE channel attention."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBNAct(in_ch + skip_ch, out_ch),
            _ConvBNAct(out_ch, out_ch),
        )
        self.se = _SEBlock(out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return self.se(self.conv(torch.cat([x, skip], dim=1)))


class CloudMambaUnet(nn.Module):
    """CloudMamba backbone + UNet decoder for dense cloud-detection segmentation.

    Encoder feature scales (for 512×512 input, patch_stride=4):
        f0 : [B, e0, H/4,  W/4 ]   128×128
        f1 : [B, e1, H/8,  W/8 ]    64×64
        f2 : [B, e2, H/16, W/16]    32×32
        f3 : [B, e3, H/32, W/32]    16×16

    Decoder upsamples back to H×W through 4 UpBlocks + a final ×4 upsample.

    Args:
        in_channels  : input spectral channels (default 3, RGB)
        num_classes  : number of segmentation classes (default 2)
        out_channels : alias for num_classes (for model_zoo compatibility)
        image_size   : training image resolution (default 512)
    """

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 out_channels: int = None,
                 image_size: int = 512,
                 embed_dims=None,
                 depths=None,
                 num_heads=None,
                 window_size=None,
                 layerscales=None,
                 layer_init_values=None,
                 drop_path_rate: float = 0.15):
        super().__init__()
        if out_channels is not None:
            num_classes = out_channels

        # ~30M params: embed_dims up, depths [2,2,7,2]
        embed_dims        = embed_dims        or [96, 192, 384, 512]
        depths            = depths            or [2, 2, 7, 2]
        num_heads         = num_heads         or [6, 6, 12, 16]
        window_size       = window_size       or [7, 7, 7, 7]
        layerscales       = layerscales       or [False, False, True, True]
        layer_init_values = layer_init_values or [1e-6, 1e-6, 1e-6, 1e-6]

        self.encoder = CloudMamba(
            image_size=image_size,
            in_chans=in_channels,
            num_classes=0,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            drop_path_rate=drop_path_rate,
            layerscales=layerscales,
            layer_init_values=layer_init_values,
            return_features=True,
        )

        e0, e1, e2, e3 = embed_dims

        # Decoder: 3 × 2× skip-fusion steps
        self.up3 = _UpBlock(e3, e2, e2)   # 16 → 32
        self.up2 = _UpBlock(e2, e1, e1)   # 32 → 64
        self.up1 = _UpBlock(e1, e0, e0)   # 64 → 128
        # 128 → 512: two separate ×2 steps with intermediate feature refinement
        self.up0 = nn.Sequential(
            _ConvBNAct(e0, e0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            _ConvBNAct(e0, e0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            _ConvBNAct(e0, e0 // 2),
        )
        self.seg_head = nn.Conv2d(e0 // 2, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0, f1, f2, f3 = self.encoder(x)   # multi-scale features
        x = self.up3(f3, f2)
        x = self.up2(x,  f1)
        x = self.up1(x,  f0)
        x = self.up0(x)
        return self.seg_head(x)             # [B, num_classes, H, W]



