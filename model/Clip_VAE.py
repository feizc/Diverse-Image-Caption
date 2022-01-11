import torch 
import torch.nn as nn
from transformers import GPT2LMHeadModel 
from typing import Tuple, Optional, Union 



class PriorNetwork(nn.Module): 
    def __init__(self, d_in, d_out):
        super(PriorNetwork, self).__init__() 
        self.mean = nn.Linear(d_in, d_out) 
        self.logvar = nn.Linear(d_in, d_out) 
    
    def forward(self, input_emb): 
        mean = self.mean(input_emb) 
        logvar = self.logvar(input_emb) 

        outputs = (mean, logvar,) 
        return outputs 



class RecognitionNetwork(nn.Module): 
    def __init__(self, d_in, d_out):
        super(RecognitionNetwork, self).__init__() 
        self.mean = nn.Linear(d_in, d_out) 
        self.logvar = nn.Linear(d_in, d_out) 
    
    def forward(self, input_emb): 
        mean = self.mean(input_emb) 
        logvar = self.logvar(input_emb) 

        outputs = (mean, logvar,) 
        return outputs 


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_model(x), self.logvar_model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.mean_model = nn.Sequential(*layers)
        
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act()) 
        self.logvar_model = nn.Sequential(*layers) 



class ClipVAE(nn.Module): 
    def __init__(self, prefix=5, learn_prior=True):
        super(ClipVAE, self).__init__() 
        self.learn_prior = learn_prior 

        self.priornet = MLP((512, 1024, 768*5))
        self.posternet = MLP((512*2, 1024, 768*5))
        self.gpt = GPT2LMHeadModel.from_pretrained('ckpt/gpt2') 
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]  
        self.d_model = 768 
        self.prefix = prefix
    
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
        prefix_features = torch.cat((img_features, txt_features), dim=1)
        posterior_mean, posterior_logvar = self.posternet(prefix_features) 

        if self.learn_prior: 
            prior_mean, prior_logvar = self.priornet(img_features)

        if from_prior: 
            latent_mean, latent_logvar = prior_mean, prior_logvar 
        else: 
            latent_mean, latent_logvar = posterior_mean, posterior_logvar 
        
        latent_mean = latent_mean.view(-1, self.prefix, self.d_model)
        latent_logvar = latent_logvar.view(-1, self.prefix, self.d_model)

        if from_mean: 
            z = latent_mean 
        else: 
            z = self.reparameterize(latent_mean, latent_logvar) 
        assert not torch.isnan(z).any(), 'training get nan z'  
        
        emb_txt = self.gpt.transformer.wte(tokens)  # (bsz, seq_len, d_model)
        
        embedding_cat = torch.cat((z, emb_txt), dim=1) 
        out = self.gpt(inputs_embeds=embedding_cat, attention_mask=mask) 
        
        num = self.prefix * self.d_model
        kl_loss = self.kl_loss(posterior_mean.view(-1, num), posterior_logvar.view(-1, num), \
                                prior_mean.view(-1, num), prior_logvar.view(-1, num)).unsqueeze(0) 
        outputs = (out,) + (kl_loss,) 
        return outputs  




