""" 
PhysMamba: State Space Duality Model for Remote Physiological Measurement
"""
import sys
import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
from functools import partial
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.layers import DropPath, to_2tuple
import math
from einops import rearrange
from mamba_ssm.modules.SA_Mamba2_simple import Mamba2Simple as SA_Mamba
from mamba_ssm.modules.CA_Mamba2_simple import Mamba2Simple as CA_Mamba
from tensorboardX import SummaryWriter
import numpy as np

class CAFusion_Stem(nn.Module):
    def __init__(self,apha=0.5,belta=0.5,dim=24):

        super(CAFusion_Stem, self).__init__()

        self.stem11 = nn.Sequential(nn.Conv2d(3, dim//2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim//2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
            )
        
        self.stem12 = nn.Sequential(nn.Conv2d(12, dim//2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
            )

        self.stem21 =nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.stem22 =nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.apha = apha
        self.belta = belta

    def forward(self, x):
        """Definition of Fusion_Stem.
        Args:
          x [N,D,C,H,W]
        Returns:
          fusion_x [N*D,C,H/8,W/8]
        """
        N, D, C, H, W = x.shape 
        x1 = torch.cat([x[:,:1,:,:,:],x[:,:1,:,:,:],x[:,:D-2,:,:,:]],1)
        x2 = torch.cat([x[:,:1,:,:,:],x[:,:D-1,:,:,:]],1)
        x3 = x
        x4 = torch.cat([x[:,1:,:,:,:],x[:,D-1:,:,:,:]],1)
        x5 = torch.cat([x[:,2:,:,:,:],x[:,D-1:,:,:,:],x[:,D-1:,:,:,:]],1)

        x_diff = self.stem12(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],2).view(N * D, 12, H, W))

        x3 = x3.contiguous().view(N * D, C, H, W) 
        x = self.stem11(x3) 

        #fusion layer1
        x_path1 = self.apha*x + self.belta*x_diff 
        x_path1 = self.stem21(x_path1) 
        #fusion layer2
        x_path2 = self.stem22(x_diff) 

        x = self.apha*x_path1 + self.belta*x_path2

        return x
 

class SAFusion_Stem(nn.Module):
    def __init__(self,apha=0.5,belta=0.5,dim=24):
        super(SAFusion_Stem, self).__init__()


        self.stem11 = nn.Sequential(nn.Conv2d(3, dim//2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim//2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
            )
        
        self.stem12 = nn.Sequential(nn.Conv2d(12, dim//2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
            )

        self.stem21 =nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.stem22 =nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.apha = apha
        self.belta = belta

    def forward(self, x):
        """Definition of Fusion_Stem.
        Args:
          x [N,D,C,H,W]
        Returns:
          fusion_x [N*D,C,H/8,W/8]
        """
        N, D, C, H, W = x.shape
        x1 = torch.cat([x[:,:1,:,:,:],x[:,:1,:,:,:],x[:,:D-2,:,:,:]],1)
        x2 = torch.cat([x[:,:1,:,:,:],x[:,:D-1,:,:,:]],1)
        x3 = x
        x4 = torch.cat([x[:,1:,:,:,:],x[:,D-1:,:,:,:]],1)
        x5 = torch.cat([x[:,2:,:,:,:],x[:,D-1:,:,:,:],x[:,D-1:,:,:,:]],1)
        # x_diff = self.stem12(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],2).view(N * D, 12, H, W))
        # import pdb; pdb.set_trace()
        x_diff = self.stem12(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],2).view(N * D, 12, H, W))
        x3 = x3.contiguous().view(N * D, C, H, W)
        x = self.stem11(x3)

        #fusion layer1
        x_path1 = self.apha*x + self.belta*x_diff
        x_path1 = self.stem21(x_path1)
        #fusion layer2
        x_path2 = self.stem22(x_diff)

        x = self.apha*x_path1 + self.belta*x_path2

        return x
    


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=3, keepdim=True)
        xsum = torch.sum(xsum, dim=4, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[3] * xshape[4] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class Frequencydomain_FFN(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()

        self.scale = 0.02
        self.dim = dim * mlp_ratio

        self.r = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.dim))

        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, dim * mlp_ratio, 1, 1, 0, bias=False),  
            nn.BatchNorm1d(dim * mlp_ratio),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(dim * mlp_ratio, dim, 1, 1, 0, bias=False),  
            nn.BatchNorm1d(dim),
        )


    def forward(self, x):
        B, N, C = x.shape
  
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)

        x_fre = torch.fft.fft(x, dim=1, norm='ortho') # FFT on N dimension

        x_real = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.real, self.r) - \
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.i) + \
            self.rb
        )
        x_imag = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.r) + \
            torch.einsum('bnc,cc->bnc', x_fre.real, self.i) + \
            self.ib
        )

        x_fre = torch.stack([x_real, x_imag], dim=-1).float()
        x_fre = torch.view_as_complex(x_fre)
        x = torch.fft.ifft(x_fre, dim=1, norm="ortho")
        x = x.to(torch.float32)

        x = self.fc2(x.transpose(1, 2)).transpose(1, 2)
        return x


class CA_MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = CA_Mamba(
            d_model=dim,  
            d_state=d_state,  
            d_conv=d_conv, 
            expand=expand  
        )
    def forward(self, x,C_SA):
        B, N, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm,C_SA)    
        return x_mamba

class SA_MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = SA_Mamba(
            d_model=dim,  
            d_state=d_state,  
            d_conv=d_conv, 
            expand=expand  
        )
    def forward(self, x):
        B, N, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)    
        return x_mamba

class SAMambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = SAMamba(
            d_model=dim,  
            d_state=d_state,  
            d_conv=d_conv, 
            expand=expand  
        )
    def forward(self, x):
        B, N, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)    
        return x_mamba

class CABlock_CrossMamba(nn.Module):
    def __init__(self, 
        dim, 
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = CA_MambaLayer(dim)
        self.mlp = Frequencydomain_FFN(dim,mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)
        self.downsampleseq=nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3,stride=2, padding=1),
            nn.BatchNorm1d(64)
            # norm_layer(dim)
        )
        

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

    def forward(self, x,C_SA_list):
        B, D, C = x.size()  
        x_path = torch.zeros(x.size()).to("cuda:0") 
        C1= C_SA_list[0].permute(0,2,1)
        C1=C1.permute(0,2,1)
        x_o = self.drop_path(self.attn(x,C1)) 
        tt = D // 2
        for j in range(D//tt):
            C2=C_SA_list[1][j].permute(0,2,1)
            C2=C2.permute(0,2,1)
            x_div = self.attn(x[:,j * tt : (j + 1)* tt,:],C2) 
            x_path[:,j * tt : (j + 1)* tt,:] = x_div 
        x_o = x_o + self.drop_path(x_path) 
        tt = D // 4
        for j in range(D//tt):
            C3=C_SA_list[2][j].permute(0,2,1)
            C3=C3.permute(0,2,1)
            x_div = self.attn(x[:,j * tt : (j + 1)* tt,:],C3)
            x_path[:,j * tt : (j + 1)* tt,:] = x_div 
        x_o = x_o + self.drop_path(x_path) 
        tt = D // 8
        for j in range(D//tt):
            C4=C_SA_list[3][j].permute(0,2,1)
            C4=C4.permute(0,2,1)
            x_div = self.attn(x[:,j * tt : (j + 1)* tt,:],C4)
            x_path[:,j * tt : (j + 1)* tt,:] = x_div
        x_o = x_o + self.drop_path(x_path)
        x = x + self.drop_path(self.norm1(x_o)) 
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        return x

class SABlock_CrossMamba(nn.Module):
    def __init__(self, 
        dim, 
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = SA_MambaLayer(dim)
        self.mlp = Frequencydomain_FFN(dim,mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
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
        B, D, C = x.size()  
        x_path= torch.zeros(x.size()).to("cuda:0") 
        C_SA=[]
        C_SA_0=[]
        C_SA_2=[]
        C_SA_4=[]
        C_SA_8=[]
        x_o,C_SA_0 =self.attn(x)
        x_o = self.drop_path(x_o) 
        tt = D // 2
        for j in range(D//tt):
            x_div,C_SA_2_j = self.attn(x[:,j * tt : (j + 1)* tt,:]) 
            C_SA_2.append(C_SA_2_j)
            x_path[:,j * tt : (j + 1)* tt,:] = x_div 
        x_o = x_o + self.drop_path(x_path) 
        tt = D // 4
        for j in range(D//tt):
            x_div,C_SA_4_j = self.attn(x[:,j * tt : (j + 1)* tt,:]) 
            C_SA_4.append(C_SA_4_j)
            x_path[:,j * tt : (j + 1)* tt,:] = x_div 
        x_o = self.drop_path(x_path) 
        tt = D // 8
        for j in range(D//tt):
            x_div,C_SA_8_j = self.attn(x[:,j * tt : (j + 1)* tt,:])
            C_SA_8.append(C_SA_8_j)
            x_path[:,j * tt : (j + 1)* tt,:] = x_div
        x_o = x_o + self.drop_path(x_path)

        x = x + self.drop_path(self.norm1(x_o)) 
        x = x + self.drop_path(self.mlp(self.norm2(x))) 

        C_SA=[C_SA_0,C_SA_2,C_SA_4,C_SA_8]

        return x,C_SA


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class PhysMamba(nn.Module):
    def __init__(self, 
                 depth=24, 
                 embed_dim_CA=64, 
                 embed_dim_SA=64, 
                 mlp_ratio=2,
                 drop_rate=0.,
                 drop_path_rate_CA=0.1,
                 drop_path_rate_SA=0.1,
                 initializer_cfg=None,
                 device=None,
                 dtype=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()

        ## CA Layers
        self.embed_dim_CA = embed_dim_CA

        self.Fusion_Stem_CA = CAFusion_Stem(dim=embed_dim_CA//4)
        self.attn_mask_CA = Attention_mask()

        self.stem3_CA = nn.Sequential(
            nn.Conv3d(embed_dim_CA//4, embed_dim_CA, kernel_size=(2, 5, 5), stride=(2, 1, 1),padding=(0,2,2)),
            nn.BatchNorm3d(embed_dim_CA),
        )

        dpr_CA = [x.item() for x in torch.linspace(0, drop_path_rate_CA, depth)]  # stochastic depth decay rule
        inter_dpr_CA = [0.0] + dpr_CA
        self.blocks_CrossCA = nn.ModuleList([CABlock_CrossMamba(
            dim = embed_dim_CA, 
            mlp_ratio = mlp_ratio,
            drop_path=inter_dpr_CA[i], 
            norm_layer=nn.LayerNorm,)
        for i in range(depth)])

        ## SA Layers
        self.embed_dim_SA = embed_dim_SA

        self.Fusion_Stem_SA = SAFusion_Stem(dim=embed_dim_SA//4)
        self.attn_mask_SA = Attention_mask()
        self.stem3_SA = nn.Sequential(
            nn.Conv3d(embed_dim_SA//4, embed_dim_SA, kernel_size=(2, 5, 5), stride=(2, 1, 1),padding=(0,2,2)),
            nn.BatchNorm3d(embed_dim_SA),
        )

        dpr_SA = [x.item() for x in torch.linspace(0, drop_path_rate_SA, depth)]  # stochastic depth decay rule
        inter_dpr_SA = [0.0] + dpr_SA
        self.blocks_CrossSA = nn.ModuleList([SABlock_CrossMamba(
            dim = embed_dim_SA, 
            mlp_ratio = mlp_ratio,
            drop_path=inter_dpr_SA[i], 
            norm_layer=nn.LayerNorm,)
        for i in range(depth)])

        self.ConvBlockLast= nn.Conv1d(128, 1, kernel_size=1,stride=1, padding=0)

        self.upsample = nn.Upsample(scale_factor=2)
        

        # init
        self.apply(segm_init_weights)
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.upchannels=nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1, stride=1, padding=0).to('cuda')


    def forward(self, x):
        B, D, C, H, W = x.shape 
        
        x_SA = x[:, :, :, :, :]  
        x_CA = x[:, :, :, :, :]  

        Bf, Df, Cf, Hf, Wf = x_SA.shape
        Bs, Ds, Cs, Hs, Ws = x_CA.shape

        #### SA Pathway
        x_SA = self.Fusion_Stem_SA(x_SA)    #[N*D C H/8 W/8] 
        x_SA = x_SA.view(Bf,Df,self.embed_dim_SA//4,Hf//8,Wf//8).permute(0,2,1,3,4) 
        x_SA = self.stem3_SA(x_SA) 

        mask_f = torch.sigmoid(x_SA) 
        mask_f = self.attn_mask_SA(mask_f) 
        x_SA = x_SA * mask_f 

        x_SA = torch.mean(x_SA,4) 
        x_SA = torch.mean(x_SA,3) 
        x_SA = rearrange(x_SA, 'b c t -> b t c') 

        #### CA Pathway

        x_CA = self.Fusion_Stem_CA(x_CA)    #[N*D C H/8 W/8] 
        x_CA = x_CA.view(Bs,Ds,self.embed_dim_CA//4,Hs//8,Ws//8).permute(0,2,1,3,4) 
        x_CA = self.stem3_CA(x_CA) 

        mask_s = torch.sigmoid(x_CA) 
        mask_s = self.attn_mask_CA(mask_s) 
        x_CA = x_CA * mask_s 

        x_CA = torch.mean(x_CA,4) 
        x_CA = torch.mean(x_CA,3) 
        x_CA = rearrange(x_CA, 'b c t -> b t c') 

        C_SA_list=[]
        for blk in self.blocks_CrossSA:
            x_SA,C_SA_list_j= blk(x_SA) 
            C_SA_list.append(C_SA_list_j)
        C_len=len(C_SA_list)
        i=0
        for blk in self.blocks_CrossCA:
            if i<C_len:
                x_CA = blk(x_CA,C_SA_list[i]) 

        # Fusion
        x_concat=torch.cat((x_SA,x_CA),dim=2)

        rPPG = x_concat.permute(0,2,1)  
        rPPG = self.upsample(rPPG) 
        rPPG = self.ConvBlockLast(rPPG)    #[N, 1, D] 
        rPPG = rPPG.squeeze(1) 
 
        return rPPG
