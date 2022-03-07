# output graft for CLIP-CVAE 
import torch 
import torch.nn as nn
from transformers import GPT2Model, GPT2PreTrainedModel

from .Clip_CVAE import MLP 
from .module import EncoderLayer 


class OG_Layer(nn.Module): 
    def __init__(self, N=1, d_model=768, d_k=64, d_v=64, dropout=.1, h=12, d_ff=3072, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(OG_Layer, self).__init__() 
        self.d_model = d_model 
        self.dropout = dropout 
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)]) 
        
    def forward(self, input): 
        # input(bsz, latent_var + seq_len, d_in) 
        out = input 
        for l in self.layers: 
            out = l(out, out, out) 
        return out 



class OG_CVAE(GPT2PreTrainedModel): 
    def __init__(self, config, latent_num=64, clip_d=512, learn_prior=True):
        super(OG_CVAE, self).__init__(config) 
        self.learn_prior = learn_prior 

        self.priornet = MLP((clip_d, 2*latent_num, latent_num))
        self.posternet = MLP((clip_d*2, 2*latent_num, latent_num))

        self.gpt = GPT2Model.from_pretrained('ckpt/gpt2') 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.d_model = config.n_embd
        self.latent_num = latent_num 
        self.pre_len = 5 

        self.project = nn.Linear(latent_num, self.pre_len*config.n_embd) 
        self.graft = OG_Layer()
        self.tie_weights()
    
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.gpt.wte)
    
    def reparameterize(self, mean, logvar, z=None):
        std = logvar.mul(0.5).exp()
        if z is None:
            z = torch.randn(std.size(), device=mean.device, dtype=mean.dtype)
        return z.mul(std) + mean 
    

    def kl_loss(self, mean1, logvar1, mean2, logvar2):
        exponential = logvar1 - logvar2 - torch.pow(mean1 - mean2, 2) / logvar2.exp() - torch.exp(logvar1 - logvar2) + 1
        result = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))
        return result.mean()

    def forward(self, tokens, img_features, txt_features, mask=None, from_prior=False, from_mean=False): 
        # latent representation 
        # image_features, txt_features (bsz, 512) 

        all_txt_hidden_states = self.gpt(input_ids=tokens, output_hidden_states=True)[0]
        # (bsz, seq_len, d_model)
        prefix_features = torch.cat((img_features, txt_features), dim=1)
        posterior_mean, posterior_logvar = self.posternet(prefix_features) 

        if self.learn_prior: 
            prior_mean, prior_logvar = self.priornet(img_features)

        if from_prior: 
            latent_mean, latent_logvar = prior_mean, prior_logvar 
        else: 
            latent_mean, latent_logvar = posterior_mean, posterior_logvar 
        
        latent_mean = latent_mean.view(-1, self.latent_num)
        latent_logvar = latent_logvar.view(-1, self.latent_num)
        if from_mean: 
            z = latent_mean 
        else: 
            z = self.reparameterize(latent_mean, latent_logvar)  # (bsz, latent_num)
        assert not torch.isnan(z).any(), 'training get nan z'  
        
        z_emb = self.project(z).view(-1, self.pre_len, self.d_model) # (bsz, pre_len, d_model)
        
        embedding_cat = torch.cat((z_emb, all_txt_hidden_states), dim=1) 

        embedding_cat = self.graft(embedding_cat) 
        lm_logits = self.lm_head(embedding_cat)


        num = self.latent_num
        kl_loss = self.kl_loss(posterior_mean.view(-1, num), posterior_logvar.view(-1, num), \
                                prior_mean.view(-1, num), prior_logvar.view(-1, num)).unsqueeze(0) 
        outputs = (lm_logits,) + (kl_loss,) 
        return outputs  



