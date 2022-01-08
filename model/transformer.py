import torch 
import utils 
from torch import nn 
from .module import Module, BeamSearch
import copy 

class CaptioningModel(Module):
    def __init__(self):
        super(CaptioningModel, self).__init__()

    def init_weights(self):
        raise NotImplementedError

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        raise NotImplementedError

    def forward(self, images, seq, *args):
        device = images.device
        b_s = images.size(0)
        seq_len = seq.size(1)
        state = self.init_state(b_s, device)
        out = None

        outputs = []
        for t in range(seq_len):
            out, state = self.step(t, state, out, images, seq, *args, mode='teacher_forcing')
            outputs.append(out)

        outputs = torch.cat([o.unsqueeze(1) for o in outputs], 1)
        return outputs

    def test(self, visual: utils.TensorOrSequence, max_len: int, eos_idx: int, **kwargs) -> utils.Tuple[torch.Tensor, torch.Tensor]:
        b_s = utils.get_batch_size(visual)
        device = utils.get_device(visual)
        outputs = []
        log_probs = []

        mask = torch.ones((b_s,), device=device)
        with self.statefulness(b_s):
            out = None
            for t in range(max_len):
                log_probs_t = self.step(t, out, visual, None, mode='feedback', **kwargs)
                out = torch.max(log_probs_t, -1)[1]
                mask = mask * (out.squeeze(-1) != eos_idx).float()
                log_probs.append(log_probs_t * mask.unsqueeze(-1).unsqueeze(-1))
                outputs.append(out)

        return torch.cat(outputs, 1), torch.cat(log_probs, 1)

    def beam_search(self, visual: utils.TensorOrSequence, max_len: int, eos_idx: int, beam_size: int, out_size=1,
                    return_probs=False, **kwargs):
        # self -> Transformer model
        bs = BeamSearch(self, max_len, eos_idx, beam_size) 
        return bs.apply(visual, out_size, return_probs, **kwargs)


class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq, *args):
        enc_output, mask_enc = self.encoder(images)
        dec_output = self.decoder(seq, enc_output, mask_enc)
        return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.enc_output, self.mask_enc = self.encoder(visual)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc)




