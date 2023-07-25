import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class Embeddings(nn.Module):
    def __init__(self,dim=64,dim1=128,bias=True):
        super(Embeddings, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.en_layer1_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            self.activation,
        )
        self.en_layer1_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.en_layer1_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.en_layer1_4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1))

        self.en_layer2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            self.activation,
        )
        self.en_layer2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.en_layer2_3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.en_layer2_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1))


        self.en_layer3_1 = nn.Sequential(
            nn.Conv2d(128, 320, kernel_size=3, stride=2, padding=1),
            self.activation,
        )
        ######
        self.layer_fussion = LAM_Module_v2(in_dim=int(dim * 3))
        self.conv_fuss = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)
        ######
        self.layer_fussion1 = LAM_Module_v2(in_dim=int(dim1 * 3))
        self.conv_fuss1 = nn.Conv2d(int(dim1 * 3), int(dim1), kernel_size=1, bias=bias)

    def forward(self, x):

        hx = self.en_layer1_1(x)
        hx1 = self.activation(self.en_layer1_2(hx) + hx)
        hx2 = self.activation(self.en_layer1_3(hx1) + hx1)
        hx3 = self.activation(self.en_layer1_4(hx2) + hx2)
        inp_fusion_123 = torch.cat([hx1.unsqueeze(1), hx2.unsqueeze(1), hx3.unsqueeze(1)], dim=1)
        #
        out_fusion_123 = self.layer_fussion(inp_fusion_123)
        out_fusion_123 = self.conv_fuss(out_fusion_123)
        #
        residual_1 = out_fusion_123



        hx4 = self.en_layer2_1(residual_1)
        hx5 = self.activation(self.en_layer2_2(hx4) + hx4)
        hx6 = self.activation(self.en_layer2_3(hx5) + hx5)
        hx7 = self.activation(self.en_layer2_4(hx6) + hx6)
        inp_fusion_567 = torch.cat([hx5.unsqueeze(1), hx6.unsqueeze(1), hx7.unsqueeze(1)], dim=1)
        #
        out_fusion_567 = self.layer_fussion1(inp_fusion_567)
        out_fusion_567 = self.conv_fuss1(out_fusion_567)
        #
        residual_2 = out_fusion_567

        hx8 = self.en_layer3_1(residual_2)



        return hx8, residual_1, residual_2


class Embeddings_output(nn.Module):
    def __init__(self):
        super(Embeddings_output, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.de_layer3_1 = nn.Sequential(
            nn.ConvTranspose2d(320, 192, kernel_size=4, stride=2, padding=1),
            self.activation,
        )
        head_num = 3
        dim = 192

        self.de_layer2_2 = nn.Sequential(
            nn.Conv2d(192+128, 192, kernel_size=1, padding=0),
            self.activation,
        )

        self.de_block_1 = Intra_SA(dim, head_num)
        self.de_block_2 = Inter_SA(dim, head_num)
        self.de_block_3 = Intra_SA(dim, head_num)
        self.de_block_4 = Inter_SA(dim, head_num)
        self.de_block_5 = Intra_SA(dim, head_num)
        self.de_block_6 = Inter_SA(dim, head_num)


        self.de_layer2_1 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=4, stride=2, padding=1),
            self.activation,
        )

        self.de_layer1_3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, padding=0),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.de_layer1_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.de_layer1_1 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            self.activation
        )

    def forward(self, x, residual_1, residual_2):


        hx = self.de_layer3_1(x)

        hx = self.de_layer2_2(torch.cat((hx, residual_2), dim = 1))
        hx = self.de_block_1(hx)
        hx = self.de_block_2(hx)
        hx = self.de_block_3(hx)
        hx = self.de_block_4(hx)
        hx = self.de_block_5(hx)
        hx = self.de_block_6(hx)
        hx = self.de_layer2_1(hx)

        hx = self.activation(self.de_layer1_3(torch.cat((hx, residual_1), dim = 1)) + hx)
        hx = self.activation(self.de_layer1_2(hx) + hx)
        hx = self.de_layer1_1(hx)

        return hx

class Attention(nn.Module):
    def __init__(self, head_num):
        super(Attention, self).__init__()
        self.num_attention_heads = head_num
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        B, N, C = x.size()
        attention_head_size = int(C / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, query_layer, key_layer, value_layer):
        B, N, C = query_layer.size()
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        _, _, _, d = query_layer.size()
        attention_scores = attention_scores / math.sqrt(d)
        attention_probs = self.softmax(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (C,)
        attention_out = context_layer.view(*new_context_layer_shape)

        return attention_out





# CPE (Conditional Positional Embedding)
class PEG(nn.Module):
    def __init__(self, hidden_size):
        super(PEG, self).__init__()
        self.PEG = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)

    def forward(self, x):
        x = self.PEG(x) + x
        return x

###############################################################################
class FeedForward(nn.Module):
    def __init__(self, hidden_size, ffn_expansion_factor=2.66, bias=True):
        super(FeedForward, self).__init__()

        hidden_features = int(hidden_size * ffn_expansion_factor)

        self.project_in = nn.Linear(hidden_size, hidden_features * 2, bias=bias)

        self.dwconv = nn.Linear(hidden_features * 2, hidden_features * 2, bias=bias)

        self.project_out = nn.Linear(hidden_features, hidden_size, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=-1)
        x = F.gelu(x2) * x1 + F.gelu(x1) * x2
        x = self.project_out(x)
        return x

###############################################################################
#### Cross-layer Attention Fusion Block
class LAM_Module_v2(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim,bias=True):
        super(LAM_Module_v2, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d( self.chanel_in ,  self.chanel_in *3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in*3, self.chanel_in*3, kernel_size=3, stride=1, padding=1, groups=self.chanel_in*3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize,N*C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1+x
        out = out.view(m_batchsize, -1, height, width)
        return out

class Intra_SA(nn.Module):
    def __init__(self, dim, head_num, ffn_expansion_factor=2.66, bias=True):
        super(Intra_SA, self).__init__()
        self.hidden_size = dim // 2
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(dim)
        self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_h = nn.Linear(self.hidden_size, self.hidden_size * 3)  # qkv_h
        self.qkv_local_v = nn.Linear(self.hidden_size, self.hidden_size * 3)  # qkv_v
        self.fuse_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = Mlp(dim)
        self.attn = Attention(head_num=self.head_num)
        self.PEG = PEG(dim)
        ###
        # self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        h = x
        B, C, H, W = x.size()

        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x_input = torch.chunk(self.conv_input(x), 2, dim=1)
        feature_h = (x_input[0]).permute(0, 2, 3, 1).contiguous()
        feature_h = feature_h.view(B * H, W, C//2)
        feature_v = (x_input[1]).permute(0, 3, 2, 1).contiguous()
        feature_v = feature_v.view(B * W, H, C//2)
        qkv_h = torch.chunk(self.qkv_local_h(feature_h), 3, dim=2)
        qkv_v = torch.chunk(self.qkv_local_v(feature_v), 3, dim=2)
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
        q_v, k_v, v_v = qkv_v[0], qkv_v[1], qkv_v[2]

        if H == W:
            query = torch.cat((q_h, q_v), dim=0)
            key = torch.cat((k_h, k_v), dim=0)
            value = torch.cat((v_h, v_v), dim=0)
            attention_output = self.attn(query, key, value)
            attention_output = torch.chunk(attention_output, 2, dim=0)
            attention_output_h = attention_output[0]
            attention_output_v = attention_output[1]
            attention_output_h = attention_output_h.view(B, H, W, C//2).permute(0, 3, 1, 2).contiguous()
            attention_output_v = attention_output_v.view(B, W, H, C//2).permute(0, 3, 2, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
        else:
            attention_output_h = self.attn(q_h, k_h, v_h)
            attention_output_v = self.attn(q_v, k_v, v_v)
            attention_output_h = attention_output_h.view(B, H, W, C//2).permute(0, 3, 1, 2).contiguous()
            attention_output_v = attention_output_v.view(B, W, H, C//2).permute(0, 3, 2, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))

        x = attn_out + h
        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x = self.PEG(x)

        return x

class Inter_SA(nn.Module):
    def __init__(self,dim, head_num, ffn_expansion_factor=2.66, bias=True):
        super(Inter_SA, self).__init__()
        self.hidden_size = dim
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.conv_input = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.conv_h = nn.Conv2d(self.hidden_size//2, 3 * (self.hidden_size//2), kernel_size=1, padding=0)  # qkv_h
        self.conv_v = nn.Conv2d(self.hidden_size//2, 3 * (self.hidden_size//2), kernel_size=1, padding=0)  # qkv_v
        self.ffn_norm = nn.LayerNorm(self.hidden_size)
        # self.ffn = Mlp(self.hidden_size)
        self.fuse_out = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.attn = Attention(head_num=self.head_num)
        self.PEG = PEG(dim)
        ###
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        h = x
        B, C, H, W = x.size()

        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x_input = torch.chunk(self.conv_input(x), 2, dim=1)
        feature_h = torch.chunk(self.conv_h(x_input[0]), 3, dim=1)
        feature_v = torch.chunk(self.conv_v(x_input[1]), 3, dim=1)
        query_h, key_h, value_h = feature_h[0], feature_h[1], feature_h[2]
        query_v, key_v, value_v = feature_v[0], feature_v[1], feature_v[2]

        horizontal_groups = torch.cat((query_h, key_h, value_h), dim=0)
        horizontal_groups = horizontal_groups.permute(0, 2, 1, 3).contiguous()
        horizontal_groups = horizontal_groups.view(3*B, H, -1)
        horizontal_groups = torch.chunk(horizontal_groups, 3, dim=0)
        query_h, key_h, value_h = horizontal_groups[0], horizontal_groups[1], horizontal_groups[2]

        vertical_groups = torch.cat((query_v, key_v, value_v), dim=0)
        vertical_groups = vertical_groups.permute(0, 3, 1, 2).contiguous()
        vertical_groups = vertical_groups.view(3*B, W, -1)
        vertical_groups = torch.chunk(vertical_groups, 3, dim=0)
        query_v, key_v, value_v = vertical_groups[0], vertical_groups[1], vertical_groups[2]


        if H == W:
            query = torch.cat((query_h, query_v), dim=0)
            key = torch.cat((key_h, key_v), dim=0)
            value = torch.cat((value_h, value_v), dim=0)
            attention_output = self.attn(query, key, value)
            attention_output = torch.chunk(attention_output, 2, dim=0)
            attention_output_h = attention_output[0]
            attention_output_v = attention_output[1]
            attention_output_h = attention_output_h.view(B, H, C//2, W).permute(0, 2, 1, 3).contiguous()
            attention_output_v = attention_output_v.view(B, W, C//2, H).permute(0, 2, 3, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
        else:
            attention_output_h = self.attn(query_h, key_h, value_h)
            attention_output_v = self.attn(query_v, key_v, value_v)
            attention_output_h = attention_output_h.view(B, H, C//2, W).permute(0, 2, 1, 3).contiguous()
            attention_output_v = attention_output_v.view(B, W, C//2, H).permute(0, 2, 3, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))

        x = attn_out + h
        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x = self.PEG(x)

        return x

class ESIDformer(nn.Module):
    def __init__(self):
        super(ESIDformer, self).__init__()

        self.encoder = Embeddings()
        head_num = 5
        dim = 320
        self.Trans_block_1 = Intra_SA(dim, head_num)
        self.Trans_block_2 = Inter_SA(dim, head_num)
        self.Trans_block_3 = Intra_SA(dim, head_num)
        self.Trans_block_4 = Inter_SA(dim, head_num)
        self.Trans_block_5 = Intra_SA(dim, head_num)
        self.Trans_block_6 = Inter_SA(dim, head_num)
        self.Trans_block_7 = Intra_SA(dim, head_num)
        self.Trans_block_8 = Inter_SA(dim, head_num)
        self.Trans_block_9 = Intra_SA(dim, head_num)
        self.Trans_block_10 = Inter_SA(dim, head_num)
        self.Trans_block_11 = Intra_SA(dim, head_num)
        self.Trans_block_12 = Inter_SA(dim, head_num)
        self.decoder = Embeddings_output()


    def forward(self, x):

        hx, residual_1, residual_2 = self.encoder(x)
        hx = self.Trans_block_1(hx)
        hx = self.Trans_block_2(hx)+hx
        hx = self.Trans_block_3(hx)
        hx = self.Trans_block_4(hx)+hx
        hx = self.Trans_block_5(hx)
        hx = self.Trans_block_6(hx)+hx
        hx = self.Trans_block_7(hx)
        hx = self.Trans_block_8(hx)+hx
        hx = self.Trans_block_9(hx)
        hx = self.Trans_block_10(hx)+hx
        hx = self.Trans_block_11(hx)
        hx = self.Trans_block_12(hx)+hx
        hx = self.decoder(hx, residual_1, residual_2)

        return hx + x