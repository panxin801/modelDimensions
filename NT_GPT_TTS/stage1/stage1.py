
import torch
import torch.nn as nn
import torch.nn.functional as F
import k2

import ice_conformer.conformer as conformer
from ice_conformer.scaling import (ScaledLinear, ScaledEmbedding)
from ref_enc import ECAPA_TDNN as ReferenceEncoder


class TextEncoder(nn.Module):
    def __init__(self,
                 in_dim: 384,
                 n_text=512,
                 layer_nums: int = 6):
        super().__init__()

        self.embedding = ScaledEmbedding(n_text, in_dim)  # 512,384
        self.conformer = conformer.Conformer(
            num_features=in_dim,
            d_model=in_dim,
            cnn_module_kernel=5,
            dim_feedforward=int(in_dim * 4),
            num_encoder_layers=layer_nums)

    def forward(self, x, xLens, warmup=1.0):
        """ x:
            xLens: [B,phone_len]
            warmup: [B]
        Return:
            out: [B，phone_len，in_dim]
        """

        x = self.embedding(x)  # [B,phone_len, in_dim]=[b,t,384]
        layerResults, xLens = self.conformer(
            x, xLens, warmup=warmup)  # list,len=1, [B，phone_len，in_dim]. [B]
        out = layerResults[-1]  # [B，phone_len，in_dim]
        out = F.relu(out)  # [B，phone_len，in_dim]
        return out


class TokenEncoder(nn.Module):
    def __init__(self,
                 in_dim=256,
                 hid_dim=512,
                 n_token=512,
                 num_layers=2,):
        super().__init__()

        self.num_layers = num_layers
        self.hid_dim = hid_dim

        self.embedding = ScaledEmbedding(n_token, in_dim)  # 513, 256
        self.lstm = nn.LSTM(batch_first=True,
                            input_size=in_dim,  # 256
                            hidden_size=hid_dim,  # 512
                            num_layers=num_layers,
                            bidirectional=False)
        self.fc = ScaledLinear(hid_dim, hid_dim)

    def forward(self, x):
        """ x: [B,token_len]
        Return:
            x: [B,token_len, hid_dim]
            y: y is tuple, y[0].size()=[2,B,hid_dim]
        """

        x = self.embedding(x)  # [B,token_len, in_dim]
        h0 = torch.zeros(
            self.num_layers, x.shape[0], self.hid_dim).to(x.device)
        c0 = torch.zeros_like(h0).to(x.device)
        x, y = self.lstm(x, (h0, c0))
        # x=[B,token_len, hid_dim], y is tuple, y[0].size()=[2,B,hid_dim]
        x = self.fc(x)
        x = F.relu(x)

        return x, y

    def inference(self, x, hidden=None):
        """ x:[B,current_inference_step]
            hidden:
        Return:
            x: # [B,current_inference_step,512]
            hidden: tuple
        """

        x = self.embedding(x)
        if hidden is None:
            h0 = torch.zeros(
                self.num_layers, x.shape[0], self.hid_dim).to(x.device)
            c0 = torch.zeros(
                self.num_layers, x.shape[0], self.hid_dim).to(x.device)
            x, hidden = self.lstm(x, (h0, c0))
        else:
            x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        x = F.relu(x)
        return x, hidden


class AffineLinear(nn.Module):
    def __init__(self, in_dim,  # 512
                 out_dim):  # 513*2
        super().__init__()

        self.affine = ScaledLinear(in_dim, out_dim)

    def forward(self, input):
        """ input: [B,in_dim]
        Return:
            x:[B, out_dim]
        """

        x = self.affine(input)
        return x


class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self,
                 in_channels,  # 513
                 style_dim):  # 512
        super().__init__()

        self.in_channels = in_channels  # 513
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=False)  # 513

        self.style = AffineLinear(style_dim, in_channels * 2)  # 512,1026
        self.style.affine.bias.data[:in_channels] = 1
        self.style.affine.bias.data[in_channels:] = 0

    def forward(self, input, style_code):
        """ input: [B,phone_len,ranges,num_vocabs]
            style_code: [B,in_channels]
        Return:
            out: [B,phone_len,ranges,num_vocabs]
        """

        style = self.style(style_code)  # [B,in_channels*2]
        if input.dim() == 4:
            style = style.unsqueeze(1).unsqueeze(1)  # [B,1,1,in_channels*2]
        elif input.dim() == 3:
            style = style.unsqueeze(1)

        # [B,1,1,in_channels],[B,1,1,in_channels]
        gamma, beta = style.chunk(2, dim=-1)
        out = self.norm(input)  # [B,phone_len,ranges,num_vocabs]
        out = gamma * out + beta
        return out


class JointStyleBlock(nn.Module):
    def __init__(self,
                 ref_size: int = 512,
                 audio_size: int = 512):
        super().__init__()

        self.fc = ScaledLinear(ref_size, int(ref_size) * 2)  # 513,1026
        self.fc2 = ScaledLinear(int(ref_size) * 2, ref_size)  # 1026,513
        self.ln = StyleAdaptiveLayerNorm(ref_size, audio_size)  # 513,512

    def forward(self, x, ref_audio):
        """ x: [B,phone_len,ranges,num_vocabs]
            ref_audio: [B,512]
        Return:
            x:# [B,phone_len,ranges,num_vocabs]
        """

        x1 = self.ln(x, ref_audio)  # [B,phone_len,ranges,num_vocabs]
        x1 = self.fc(x1)  # [B,phone_len,ranges,num_vocabs]
        x1 = F.relu(x1)  # [B,phone_len,ranges,num_vocabs]
        x1 = self.fc2(x1)  # [B,phone_len,ranges,num_vocabs]

        return x + x1


class JointStyleNet(nn.Module):
    def __init__(self,
                 ref_size: int = 512,
                 audio_size: int = 512,
                 num_layers: int = 3):
        super().__init__()

        self.layers = nn.ModuleList(
            [JointStyleBlock(ref_size, audio_size) for _ in range(num_layers)])

    def forward(self, x, ref_audio):
        """ x: [B,phone_len,ranges,num_vocabs]
            ref_audio: # [B,audio_size]
        Return:
            x: [B,phone_len,ranges,num_vocabs]
        """

        for layer in self.layers:
            x = layer(x, ref_audio)
        return x


class JointNet(nn.Module):
    def __init__(self,
                 encoder_dim: int = 384,
                 decoder_dim: int = 512,
                 reference_dim: int = 512,
                 joint_dim: int = 512,
                 vocab_size: int = 513
                 ):
        super().__init__()

        self.encoder_proj = ScaledLinear(encoder_dim, vocab_size)  # 384, 513
        self.decoder_proj = ScaledLinear(decoder_dim, vocab_size)  # 512, 513
        self.output_linear = ScaledLinear(vocab_size, vocab_size)  # 512, 513

        self.jointStyleNet = JointStyleNet(
            vocab_size, reference_dim)  # 513,512
        self.refEncoder = ReferenceEncoder(hid_C=int(reference_dim * 2),  # 1024
                                           out_C=reference_dim)  # 512

    def forward(self,
                encoder,
                decoder,
                reference,
                if_pro=True):
        """ encoder: [B, phone_len, ranges, num_vocabs]
            decoder: [B, phone_len, ranges, num_vocabs]
            reference: [B, 80, segmentSize]
            if_pro: False
        Return:
            logits: [B,phone_len,ranges,num_vocabs]
        """

        if if_pro:
            encoderOut = self.encoder_proj(encoder)
            decoderOut = self.decoder_proj(decoder)
        else:
            encoderOut = encoder  # [B, phone_len, ranges, num_vocabs]
            decoderOut = decoder  # [B, phone_len, ranges, num_vocabs]

        referenceOut = self.refEncoder(reference)  # [B,reference_dim]

        if encoderOut.dim() == 3 and decoderOut.dim() == 4:
            seq_lens = encoderOut.size(1)
            tar_lens = decoderOut.size(1)

            encoderOut = encoderOut.unsqueeze(1)
            decoderOut = decoderOut.unsqueeze(1)

            encoderOut = encoderOut.repeat(1, tar_lens, 1, 1)
            decoderOut = decoderOut.repeat(1, 1, seq_lens, 1)

        logits = encoderOut + decoderOut  # [B,phone_len,ranges,num_vocabs]
        # [B,phone_len,ranges,num_vocabs]
        logits = self.jointStyleNet(logits, referenceOut)
        # [B,phone_len,ranges,num_vocabs]
        logits = self.output_linear(F.relu(logits))
        # [B,phone_len,ranges,num_vocabs]
        logits = F.log_softmax(logits, dim=-1)

        return logits


class Stage1Net(nn.Module):
    def __init__(self,
                 text_dim: int = 384,
                 num_vocabs: int = 513,
                 num_phonemes: int = 512,
                 token_dim: int = 512,  # 256
                 hid_token_dim: int = 512,
                 inner_dim: int = 512,  # 513
                 ref_dim: int = 512,  # 513
                 layer_nums=6,
                 use_fp16=False):
        super().__init__()

        self.use_fp16 = use_fp16

        self.textEncoder = TextEncoder(text_dim,  # 384
                                       num_phonemes,  # 512
                                       layer_nums=layer_nums)  # 6
        self.tokenEncoder = TokenEncoder(token_dim,  # 256
                                         hid_token_dim,  # 512
                                         num_vocabs)  # 513
        self.simple_token_proj = ScaledLinear(
            hid_token_dim, num_vocabs)  # 512, 513
        self.simple_phone_proj = ScaledLinear(text_dim, num_vocabs)  # 384, 513
        self.jointNet = JointNet()

    def forward(self,
                text_seq: torch.Tensor,
                token_seq: torch.Tensor,
                text_lens: torch.Tensor,
                token_lens: torch.Tensor,
                reference_audio: torch.Tensor,
                true_seq: torch.Tensor,
                lm_scale: float = 0.25,
                am_scale: float = 0.25,
                prune_range: int = 50,  # 50
                warmup: float = 1.0,
                reduction: str = "none",):
        """
            text_seq: [B,max_phone_len], phone_ids from text
            token_seq: [B,max_token_len], token_ids from wav
            text_lens: [B]: length of each phone sequence
            token_lens: [B]
            reference_audio: [B,80,segmentSize]
            true_seq: [B,max_token_len-1]-> token_seq[:,1:],除去BOS的真实序列
            lm_scale: 0.25,
            am_scale: 0.25,
            prune_range: 50,
            warmup:1.0,
            reduction: "none",
        Return:
            simple_loss: [B]
            pruned_loss: [B]
        """

        text_seq = self.textEncoder(
            text_seq, text_lens, warmup)  # [B，phone_len，text_dim]
        token_seq, _ = self.tokenEncoder(token_seq)
        # [B,token_len, hid_token_dim]

        boundary = torch.zeros(text_seq.size(
            0), 4, dtype=torch.int64, device=text_seq.device)
        # boundary[:, 2] = token_lens
        # boundary[:, 3] = text_lens

        boundary[:, 2] = text_lens
        boundary[:, 3] = token_lens
        # [symbl_start, frame_start, symbl_end, frame_end]

        am = self.simple_phone_proj(text_seq)  # [B,phone_len,num_vocabs]
        lm = self.simple_token_proj(token_seq)  # [B,token_len, num_vocabs]

        with torch.amp.autocast(enabled=self.use_fp16, dtype=torch.bfloat16, device_type="cuda"):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),  # [B,token_len, num_vocabs]
                am=am.float(),  # [B,phone_len, num_vocabs]
                symbols=true_seq,  # [B, token_len-1]
                termination_symbol=0,
                lm_only_scale=lm_scale,  # 0.25
                am_only_scale=am_scale,  # 0.25
                return_grad=True,
                reduction=reduction  # "None"
            )  # simple_loss:[B], px_grad:[B,token_len-1, phone_len], py_grad:[B,token_len, phone_len-1]

        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,  # [B,token_len-1, phone_len]
            py_grad=py_grad,  # [B, token_len, phone_len-1]
            boundary=boundary,  # [B,4]
            s_range=prune_range,  # 50
        )  # [B,phone_len-1, prune_range may auto adjust to other value]

        # am=[B,phone_len,num_vocabs]
        # lm=[B, token_len, num_vocabs]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.jointNet.encoder_proj(text_seq),
            lm=self.jointNet.decoder_proj(token_seq),
            ranges=ranges
        )  # [B,phone_len, ranges, num_vocabs], [B,phone_len, ranges, num_vocabs]

        logits = self.jointNet(am_pruned,
                               lm_pruned,
                               reference_audio,
                               if_pro=False)  # [B,phone_len,ranges,num_vocabs]

        with torch.amp.autocast(enabled=self.use_fp16, dtype=torch.bfloat16, device_type="cuda"):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),  # [B,phone_len,ranges,num_vocabs]
                symbols=true_seq,  # [B,token_len-1]
                ranges=ranges,  # [B,phone_len,ranges,]
                termination_symbol=0,
                boundary=boundary,  # [B,4]
                reduction=reduction
            )

        return simple_loss, pruned_loss

    @torch.inference_mode()
    def decode(self,
               text_outputs: torch.Tensor,
               max_lens: int,
               reference_emb: torch.Tensor,
               max_token_lens: int = 2048):
        """ text_outputs:  [B, phone_len, text_dim]
            max_lens: int
            reference_emb: [B,80,segmentSize]
            max_token_lens: int
        Return:
            y_hats:
        """

        batch = text_outputs.size(0)
        y_hats = list()
        targets = torch.LongTensor([0] * batch).to(text_outputs.device)
        targets = targets.unsqueeze(-1)
        time_num = 0
        for i in range(int(max_lens)):
            pred = -1
            while (pred != 0):
                if time_num == 0:
                    label_output, hidden = self.tokenEncoder.inference(targets)
                else:
                    label_output, hidden = self.tokenEncoder.inference(
                        targets, hidden)

                text_outputs = text_outputs[:, i, :].unsqueeze(1)
                output = self.jointNet(
                    text_outputs, label_output, reference_emb)
                output = F.log_softmax(output, dim=-1)
                output = output.squeeze(1).squeeze(1)

                top_k_output_values, top_k_output_indices = torch.topk(
                    output, k=5, dim=-1)
                normed_top_k_output_values = F.log_softmax(
                    top_k_output_values, dim=-1)
                choosed_indices = torch.multinomial(
                    normed_top_k_output_values, num_samples=1)
                targets = top_k_output_indices[0, choosed_indices]
                pred = targets

                time_num += 1
                if pred == 0:
                    break
                else:
                    y_hats.append(targets[0, :])
                if time_num > max_token_lens:
                    break

            if time_num > max_token_lens:
                break
        y_hats = torch.stack(y_hats, dim=1)
        return y_hats

    @torch.inference_mode()
    def recognize(self,
                  inputs,
                  input_lens,
                  reference_audio):
        """ inputs: [B, phone_len]
            input_lens: [B]
            reference_audio: # [B,80,segmentSize]
        Return:
            return_value:

        """
        text_outputs = self.textEncoder(
            inputs, input_lens)  # [B, phone_len, text_dim]
        max_lens, _ = torch.max(input_lens, dim=-1)

        return self.decode(text_outputs, max_lens, reference_audio)


if __name__ == "__main__":
    model = Stage1Net(text_dim=384,
                      num_vocabs=513,
                      num_phonemes=512,
                      token_dim=256,
                      hid_token_dim=512,
                      inner_dim=513,
                      ref_dim=513,)

    model = model.cuda()
