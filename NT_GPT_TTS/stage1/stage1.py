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

        self.embedding = ScaledEmbedding(n_text, in_dim)
        self.conformer = conformer.Conformer(
            num_features=in_dim,
            d_model=in_dim,
            cnn_module_kernel=5,
            dim_feedforward=int(in_dim * 4),
            num_encoder_layers=layer_nums)

    def forward(self, x, xLens, warmup=1.0):
        """ x:
            xLens:
            warmup:
        Return:
            out:
        """

        x = self.embedding(x)
        layerResults, xLens = self.conformer(x, xLens, warmup=warmup)
        out = layerResults[-1]
        out = F.relu(out)
        return out


class TokenEncoder(nn.Module):
    def __init__(self,
                 in_dim=256,
                 hid_dim=512,
                 n_token=512):
        super().__init__()

        self.embedding = ScaledEmbedding(n_token, in_dim)
        self.lstm = nn.LSTM(batch_first=True,
                            input_size=in_dim,
                            hidden_size=hid_dim,
                            num_layers=2,
                            bidirectional=False)
        self.fc = ScaledLinear(hid_dim, in_dim)

    def forward(self, x):
        """ x:
        Return:
            x:
            y:
        """

        x = self.embedding(x)
        h0 = torch.zeros(
            self.num_layers, x.shape[0], self.hid_dim).to(x.device)
        c0 = torch.zeros_like(h0).to(x.device)
        x, y = self.lstm(x, (h0, c0))
        x = self.fc(x)
        x = F.relu(x)

        return x, y

    def inference(self, x, hidden=None):
        """ x:
            hidden:
        Return:
            x:
            hidden:
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
    def __init__(self, in_dim,
                 out_dim):
        super().__init__()

        self.affine = ScaledLinear(in_dim, out_dim)

    def forward(self, input):
        """ input:
        Return:
            x:
        """

        x = self.affine(input)
        return x


class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self,
                 in_channels,
                 style_dim):
        super().__init__()

        self.in_channels = in_channels
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=False)

        self.style = AffineLinear(style_dim, in_channels * 2)
        self.style.affine.bias.data[:in_channels] = 1
        self.style.affine.bias.data[in_channels:] = 0

    def forward(self, input, style_code):
        """ input:
            style_code:
        Return:
            out: 
        """

        style = self.style(style_code)
        if input.dim() == 4:
            style = style.unsqueeze(1).unsqueeze(1)
        elif input.dim() == 3:
            style = style.unsqueeze(1)

        gamma, beta = style.chunk(2, dim=-1)
        out = self.norm(input)
        out = gamma * out + beta
        return out


class JointStyleBlock(nn.Module):
    def __init__(self,
                 ref_size: int = 512,
                 audio_size: int = 512):
        super().__init__()

        self.fc = ScaledLinear(ref_size, int(ref_size) * 2)
        self.fc2 = ScaledLinear(int(ref_size) * 2, ref_size)
        self.ln = StyleAdaptiveLayerNorm(ref_size, audio_size)

    def forward(self, x, ref_audio):
        """ x:
            ref_audio:
        Return:
            x:
        """

        x1 = self.ln(x, ref_audio)
        x1 = self.fc(x1)
        x1 = F.relu(x1)
        x1 = self.fc2(x1)

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
        """ x:
            ref_audio:
        Return:
            x:
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
                 hidden_dim: int = 2048,
                 vocab_size: int = 513
                 ):
        super().__init__()

        self.encoder_proj = ScaledLinear(encoder_dim, vocab_size)
        self.decoder_proj = ScaledLinear(decoder_dim, vocab_size)
        self.output_linear = ScaledLinear(joint_dim, vocab_size)

        self.jointStyleNet = JointStyleNet(vocab_size, reference_dim)
        self.refEncoder = ReferenceEncoder(hid_C=int(reference_dim * 2),
                                           out_C=reference_dim)

    def forward(self,
                encoder,
                decoder,
                reference,
                if_pro=True):
        """ encoder:
            decoder:
            reference:
            if_pro:
        Return:
            logits:
        """

        if if_pro:
            encoderOut = self.encoder_proj(encoder)
            decoderOut = self.decoder_proj(decoder)
        else:
            encoderOut = encoder
            decoderOut = decoder

        referenceOut = self.refEncoder(reference)

        if encoderOut.dim() == 3 and decoderOut.dim() == 3:
            seq_lens = encoderOut.size(1)
            tar_lens = decoderOut.size(1)

            encoderOut = encoderOut.unsqueeze(1)
            decoderOut = decoderOut.unsqueeze(1)

            encoderOut = encoderOut.repeat(1, tar_lens, 1, 1)
            decoderOut = decoderOut.repeat(1, 1, seq_lens, 1)

        logits = encoderOut + decoderOut
        logits = self.jointStyleNet(logits, referenceOut)
        logits = self.output_linear(logits)
        logits = F.log_softmax(logits, dim=-1)

        return logits


class Stage1Net(nn.Module):
    def __init__(self,
                 text_dim: int = 384,
                 num_vocabs: int = 513,
                 num_phonemes: int = 512,
                 token_dim: int = 512,
                 hid_token_dim: int = 512,
                 inner_dim: int = 512,
                 ref_dim: int = 512,
                 layer_nums=6,
                 use_fp16=False):
        super().__init__()

        self.use_fp16 = use_fp16

        self.textEncoder = TextEncoder(text_dim,
                                       num_phonemes,
                                       layer_nums=layer_nums)
        self.tokenEncoder = TokenEncoder(token_dim,
                                         hid_token_dim,
                                         num_vocabs)
        self.simple_token_proj = ScaledLinear(hid_token_dim, num_vocabs)
        self.simple_phone_proj = ScaledLinear(text_dim, num_vocabs)
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
                prune_range: int = 50,
                warmup: float = 1.0,
                reduction: str = "none",):
        """
            text_seq: torch.Tensor,
            token_seq: torch.Tensor,
            text_lens: torch.Tensor,
            token_lens: torch.Tensor,
            reference_audio: torch.Tensor,
            true_seq: torch.Tensor,
            lm_scale: float = 0.25,
            am_scale: float = 0.25,
            prune_range: int = 50,
            warmup: float = 1.0,
            reduction: str = "none",
        Return:
            simple_loss
            pruned_loss
        """

        text_seq = self.textEncoder(text_seq, text_lens, warmup)
        token_seq, _ = self.tokenEncoder(token_seq)

        boundary = torch.zeros(text_seq.size(
            0), 4, dtype=torch.int64, device=text_seq.device)
        boundary[:, 2] = token_lens
        boundary[:, 3] = text_lens

        am = self.simple_phone_proj(text_seq)
        lm = self.simple_token_proj(token_seq)

        with torch.amp.autocast(enabled=self.use_fp16, dtype=torch.bfloat16):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=true_seq,
                termination_symbol=0,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                return_grad=True,
                reduction=reduction
            )

        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.jointNet.encoder_proj(text_seq),
            lm=self.jointNet.decoder_proj(token_seq),
            ranges=ranges
        )

        logits = self.jointNet(am_pruned, lm_pruned,
                               reference_audio, if_pro=False)

        with torch.amp.autocast(enabled=self.use_fp16, dtype=torch.bfloat16):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=true_seq,
                ranges=ranges,
                termination_symbol=0,
                boundary=boundary,
                reduction=reduction
            )

        return simple_loss, pruned_loss

    @torch.inference_mode()
    def decode(self,
               text_outputs: torch.Tensor,
               max_lens: int,
               reference_emb: torch.Tensor,
               max_token_lens: int = 2048):
        """ text_outputs: 
            max_lens:
            reference_emb:
            max_token_lens:
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
        """ inputs:
            input_lens:
            reference_audio:
        Return:
            return_value:

        """
        text_outputs = self.textencoder(inputs)
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
