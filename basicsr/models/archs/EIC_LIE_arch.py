import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings

from . import MISCKernel_cuda as misckernel

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class BasicConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            stride,
            bias=False,
            relu=True):
        super(BasicConv, self).__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=padding,
                stride=stride,
                bias=bias)
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):

        max_c = img.max(dim=1).values.unsqueeze(1)

        input = torch.cat([img,max_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map

class MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
    def forward(self, x_in):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp))

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)

class Transforme_Block(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
            permute=True
    ):
        super().__init__()
        self.permute = permute
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        if self.permute:
            x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        if self.permute:
            x =  x.permute(0, 3, 1, 2)
        return x

class Decoder(nn.Module):
    def __init__(self, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Decoder, self).__init__()
        self.level = level

        dim_level = dim * 2**level
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([

                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),

                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                Transforme_Block(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fea_encoder):

        fea = x
        for i, (FeaUpSample, Fusion, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fusion(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            fea = LeWinBlcok(fea)

        return fea

class Cross_MHBC_cl(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
    def forward(self, x_in, y_in):
        """
        x_in: [b,h,w,c]  T      # input_feature
        y_in: [b,h,w,c]  X
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        y = y_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(y)
        v_inp = self.to_v(y)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp))

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out, attn

class Cross_MHBC_lc(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
    def forward(self, x_in, y_in, attn):
        """
        x_in: [b,h,w,c]   X      # input_feature
        y_in: [b,h,w,c]   T
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        y = y_in.reshape(b, h * w, c)
        v_inp = self.to_v(y)

        v = rearrange(v_inp, 'b n (h d) -> b h n d', h=self.num_heads)

        y =  v @ attn.transpose(-2, -1)

        y = y.permute(0, 3, 1, 2)

        y = y.reshape(b, h * w, self.num_heads * self.dim_head)
        x = x + y
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class EL_Block(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.c_MHBC_illu_img = Cross_MHBC_cl(dim, dim_head = dim_head, heads = heads)
        self.c_MHBC_evs_img = Cross_MHBC_cl(dim, dim_head = dim_head, heads = heads)
        self.c_MHBC_img_illu = Cross_MHBC_lc(dim, dim_head = dim_head, heads = heads)
        self.c_MHBC_img_evs = Cross_MHBC_lc(dim, dim_head = dim_head, heads = heads)
        self.TB = Transforme_Block(dim, dim_head = dim_head, heads = heads, permute=False)

        self.ffn_illu = PreNorm(dim, FeedForward(dim=dim))
        self.ffn_evs = PreNorm(dim, FeedForward(dim=dim))

        self.filter_evs = IAEF(dim, dim, 5, dim)

    def forward(self, x):

        illu, img, evs = x

        out, attn1 = self.c_MHBC_illu_img(img, illu)

        img = img + out

        out, attn2 = self.c_MHBC_evs_img(img, evs)

        img = img + out

        img = img + self.TB(img)

        evs = self.c_MHBC_img_evs(evs, img, attn2)

        evs = evs + self.ffn_evs(evs)

        evs = self.filter_evs(evs, illu) + evs

        illu = self.c_MHBC_img_illu(illu, img, attn1)

        illu = illu + self.ffn_illu(illu)

        return (illu, img, evs)

class EL_Block_Group(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                EL_Block(dim=dim, dim_head=dim_head, heads=heads)
            )

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        illu, img, evs = x
        evs = evs.permute(0, 2, 3, 1)
        img = img.permute(0, 2, 3, 1)
        illu = illu.permute(0, 2, 3, 1)
        x = (illu, img, evs)
        for Block in self.blocks:
            illu, img, evs = x
            _x = Block(x)
            _illu, _img, _evs = _x
            x = (illu+_illu, img+_img, evs+_evs)
        illu, img, evs = x
        evs = evs.permute(0, 3, 1, 2)
        img = img.permute(0, 3, 1, 2)
        illu = illu.permute(0, 3, 1, 2)
        out = (illu, img, evs)
        return out

class Encoder(nn.Module):
    def __init__(self, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Encoder, self).__init__()
        self.level = level

        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                EL_Block_Group(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)

            ]))
            dim_level *= 2
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        fea = x
        fea_encoder = []

        for (EL_Block, FeaDownSample, EvsDownsample, IlluEvsDownsample) in self.encoder_layers:
            fea = EL_Block(fea)
            illu, img, evs = fea
            fea_encoder.append(img)
            img = FeaDownSample(img)
            evs = EvsDownsample(evs)
            illu = IlluEvsDownsample(illu)
            fea = (illu, img, evs)
        return fea, fea_encoder

class IAEF(nn.Module):
    """Illumination-aware event filtering with the paper's branch roles.

    Illumination features predict the separable vertical/horizontal kernels;
    event features predict the sampling weights and two spatial offsets.
    """

    def __init__(self, base_channel=40, out_channels=40, kernel_size=5, illu_channel=40):
        super().__init__()
        self.softmax = nn.Softmax(1)
        self.kernel_size = kernel_size
        self.kernel_pad = int((self.kernel_size - 1) / 2.0)
        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])
        self.moduleKernel = misckernel.FunctionKernel.apply

        self.KernelOutBias = BasicConv(base_channel, out_channels, kernel_size=3, relu=False, stride=1)
        self.KernelOutWeight = BasicConv(illu_channel, kernel_size ** 2, kernel_size=3, relu=False, stride=1)
        self.KernelOutkernelx = BasicConv(base_channel, kernel_size, kernel_size=3, relu=False, stride=1)
        self.KernelOutkernely = BasicConv(base_channel, kernel_size, kernel_size=3, relu=False, stride=1)
        self.KernelOutAlpha = BasicConv(base_channel , kernel_size ** 2, kernel_size=3, relu=False, stride=1)
        self.KernelOutBeta = BasicConv(base_channel , kernel_size ** 2, kernel_size=3, relu=False, stride=1)
    def forward(self, evs, illu):

        evs = evs.permute(0, 3, 1, 2).contiguous()
        illu = illu.permute(0, 3, 1, 2).contiguous()
        s_kernal_bias = self.KernelOutBias(evs)
        s_kernal_weight = self.KernelOutWeight(evs)
        s_kernal_weight = self.softmax(s_kernal_weight)
        s_kernal_alpha = self.KernelOutAlpha(evs)
        s_kernal_beta = self.KernelOutBeta(evs)
        s_kernal_posx = self.KernelOutkernelx(illu)
        s_kernal_posy = self.KernelOutkernely(illu)

        out = self.moduleKernel(self.modulePad(torch.cat([evs,evs.new_ones(evs.size(0),1,evs.size(2),evs.size(3))], 1)  ),s_kernal_posx,s_kernal_posy, s_kernal_alpha, s_kernal_beta,s_kernal_weight)
        out_norm = out[:,-1:,:,:]
        out_norm[out_norm.abs()<0.01] = 1.0
        out = out[:,:-1,:,:] / out_norm
        out += s_kernal_bias

        out = out.permute(0, 2, 3, 1).contiguous()
        return out

class EICLIE(nn.Module):
    def __init__(self, in_channels=9, out_channels=3, n_feat=40, level=2, num_blocks=[1, 2, 2], use_fsas=False):
        super(EICLIE, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)

        self.embedding = nn.Conv2d(in_channels, n_feat, 3, 1, 1, bias=False)
        self.encoder = Encoder(dim=n_feat, level=level, num_blocks=num_blocks)
        self.decoder = Decoder(dim=n_feat, level=level, num_blocks=num_blocks)
        self.mapping = nn.Conv2d(n_feat, out_channels, 3, 1, 1, bias=False)
        self.conv_ev = nn.Conv2d(6, n_feat, kernel_size=1, bias=True)

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
             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             if m.bias is not None:
                 nn.init.constant_(m.bias, 0)

    def forward(self, x):

        img = x
        del x

        evs = img[:,3:9,:,:]
        img = img[:,0:3,:,:]

        f_ev = self.conv_ev(evs)

        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img

        out = self.embedding(input_img)
        out = (illu_fea, out, f_ev)
        out, fea_skip = self.encoder(out)

        _, out, _ = out
        out = self.decoder(out, fea_skip)
        out = self.mapping(out) + input_img

        return out
