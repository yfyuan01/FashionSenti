import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from Model.base import ImageEncoderTextEncoderBase
from Preprocess.loss import NormalizationLayer
# Code adapted from the fairseq repo.

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # assert key.size() == value.size()
        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
class MultimodalTransformer(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=2, attn_dropout=0.1, relu_dropout=0.1,
                 res_dropout=0.1, attn_mask = False):
        super(MultimodalTransformer, self).__init__()
        # these parameters need to be revised later
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True
        self.fc1 = nn.Linear(self.embed_dim,2*self.embed_dim)
        self.fc2 = nn.Linear(2*self.embed_dim,self.embed_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None ,x_v =None):
        residual = x
        x = self.maybe_layer_norm(0,x,before=True)
        mask = self.buffered_future_mask(x,x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x,key=x,value=x,attn_mask =mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x,key=x_k,value=x_v,attn_mask = mask)
        x = F.dropout(x,p=self.res_dropout,training=self.training)
        x = residual+x
        x = self.maybe_layer_norm(0, x, after=True)
        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout,training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout,training=self.training)
        x = residual+x
        x = self.maybe_layer_norm(1, x, after=True)
        return x
    def maybe_layer_norm(self, i, x, before=False, after=False):
        # different: true same: false
        assert before^after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x
    def fill_with_neg_inf(self,t):
        return t.float().fill_(float('-inf')).type_as(t)
    def buffered_future_mask(self, tensor, tensor2=None):
        dim1 = dim2 = tensor.size()[0]
        if tensor2 is not None:
            dim2 = tensor2.size(0)
        future_mask = torch.triu(self.fill_with_neg_inf(torch.ones(dim1,dim2)),1+abs(dim2-dim1))
        if tensor.is_cuda():
            future_mask = future_mask.cuda()
        return future_mask[:dim1,dim2]

class M3HATT(ImageEncoderTextEncoderBase):
    def __init__(self, args, **kwargs):
        super(M3HATT, self).__init__(**kwargs)
        self.args = args
        self.texts = kwargs.get('texts')
        normalize_scale = args.normalize_scale
        self.model['norm'] = NormalizationLayer(learn_scale=True,
                                                normalize_scale=normalize_scale)
        self.model['txt-attr'] = MultimodalTransformer()
        self.model['img-txt'] = MultimodalTransformer()
        self.model['img-txt-attr'] = MultimodalTransformer()
        self.orig_d_l = 900
        self.orig_d_i = args.fdims
        self.d_l = args.fdims/2
        self.d_i = args.fdims/2
        self.proj_attr = nn.Conv1d(self.orig_d_l,self.d_l,kernel_size=1,padding=0,bias=False)
        self.proj_img = nn.Conv1d(self.orig_d_i, self.d_i,kernel_size=1,padding=0,bias=False)
        self.classnum = 4262
        self.model['linear'] = nn.Linear(self.out_feature_image/2, self.classnum)
        self.model = nn.ModuleDict(self.model)
        self.opt = torch.optim.AdamW(
            self.get_config_optim(args.lr),
            lr=args.lr,
            betas=(0.55, 0.999)
        )

        self.proj1 = nn.Linear(3*self.out_feature_image/2, self.out_feature_image/2)
        self.proj2 = nn.Linear(self.out_feature_image/2,self.out_feature_image/2)
    def get_config_optim(self, lr):
        params = []
        for k, v in self.model.items():
            if k == 'backbone':
                params.append({'params': v.parameters(), 'lr': lr, 'lrp': float(self.args.lrp)})
            else:
                params.append({'params': v.parameters(), 'lr': lr, 'lrp': 1.0})
        return params
    def adjust_lr(self, lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr * param_group['lrp']

    def save(self, path, state={}):
        state['state_dict'] = dict()
        for k, v in self.model.items():
            state['state_dict'][k] = v.state_dict()
        state['texts'] = self.texts
        torch.save(state, path)

    def load(self, path):
        state_dict = torch.load(path)['state_dict']
        for k, v in state_dict.items():
            self.model[k].load_state_dict(v)
    def get_original_image_feature(self, x):
        x = self.extract_image_feature(x)
        return self.model['norm'](x)
    def get_original_text_feature(self, x):
        x = self.extract_text_feature(x)
        return self.model['norm'](x)
    def update(self, predict, tag):
        '''
        input = (
            (img, gid, data['c_id']),
            (we_key, text),
            (name,tag)
        )
        '''
        # loss
        final_loss = nn.CrossEntropyLoss()
        loss = final_loss(predict,tag)
        # backward
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # return log
        log_data = dict()
        log_data['loss'] = float(loss.data)
        return log_data
    def forward(self, x):
        x_img = self.extract_image_feature(x[0][0])
        #x_face = self.extract_image_feature(x[0][3])
        #x_item = self.extract_image_feature(x[0][4])
        #x_img = torch.stack((x_img,x_face, x_item),dim=1).transpose(1,2)
        x_img = x_img.unsqueeze(1).transpose(1,2)
        x_img = self.proj_img(x_img)
        x_txt = self.extract_tag_text_feature(x[1][1])
        x_txt = x_txt.transpose(1,2)
        x_txt = self.proj_attr(x_txt)
        x_txt = x_txt.permute(2,0,1)
        x_tag = self.extract_tag_attribute_feature(x[2][2])
        x_tag = x_tag.transpose(1,2)
        x_tag = self.proj_attr(x_tag)
        x_img = x_img.permute(2,0,1)
        # time * batch * channel
        x_tag = x_tag.permute(2,0,1)
        x_combine1 = self.model['txt-attr'](x_tag,x_txt,x_txt)[-1]
        x_combine2 = self.model['img-txt'](x_img, x_txt, x_txt)[-1]
        x_combine3 = self.model['img-txt-attr'](x_txt, x_tag, x_tag)[-1]
        x_combine4 = self.model['img-txt-attr'](x_txt, x_img, x_img)[-1]
        x_combine5 = (x_combine3+x_combine4)/2
        x_f = self.proj2(F.dropout(F.relu(self.proj1(torch.cat((x_combine1,x_combine2,x_combine5),dim=1))),p=0.1,training=self.training))
        x_f = self.model['linear'](x_f)
        x_t = x[2][1]
        return (x_f, x_t)



