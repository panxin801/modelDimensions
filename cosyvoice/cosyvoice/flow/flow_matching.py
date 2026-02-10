# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from matcha.models.components.flow_matching import BASECFM
from cosyvoice.utils.common import set_all_random_seed


class ConditionalCFM(BASECFM):
    def __init__(self,
                 in_channels,  # 240
                 cfm_params,
                 n_spks=1,  # 1
                 spk_emb_dim=64,  # 80
                 estimator: nn.Module = None):  # ConditionalDecoder
        super().__init__(n_feats=in_channels,
                         cfm_params=cfm_params,
                         n_spks=n_spks,
                         spk_emb_dim=spk_emb_dim,)
        self.t_scheduler = cfm_params.t_scheduler  # "cosine"
        self.training_cfg_rate = cfm_params.training_cfg_rate  # 0.2
        self.inference_cfg_rate = cfm_params.inference_cfg_rate  # 0.7
        in_channels = in_channels + \
            (spk_emb_dim if n_spks > 0 else 0)  # 240+80=320
        # Just change the architecture of the estimator here
        self.estimator = estimator  # ConditionalDecoder

    @torch.inference_mode()
    def forward(self,
                mu,
                mask,
                n_timesteps,
                temperature=1.0,
                spks=None,
                cond=None,
                prompt_len=0,
                cache=torch.zeros(1, 80, 0, 2)
                ):
        """Forward diffusion
        Args:
            mu (torch.Tensor): output of encoder, concate prompt_token and llm generated token
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask, in mel_spec shape
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)mel_len1: 是0
            cond: Not used but kept for future purposes, [B,D,T_mel],
        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        # torch.randn_like 和torch.rand_like 结果差别大了, [B,D,T_mel]
        # mu的构成相当于<prompt_token|上个chunk的mel_overklap|当前mel|下个mel_overlap>相当于有这几个部分构成，其中有的部分在例如第一或者最后chunk是没有的。
        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        # cache.size=[B,D,T,2], 2=z_cache and mu_cache
        cache_size = cache.size(2)
        # fix prompt and overlap part mu and z
        # 如果 cache 的大小不为零，则从 cache 中读取之前存储的 z 和 mu 的cache，并替换当前生成的噪声 z 和 mu 的相应部分。固定随机数，稳定复现。
        if cache_size != 0:
            z[:, :, :cache_size] = cache[:, :, :, 0]
            mu[:, :, :cache_size] = cache[:, :, :, 1]
        # 将当前的 z 和 mu 的prompt(提示)部分和最后 34 个时间步提取出来，并拼接成新的缓存 cache。34帧mel_spec对应20 overlap tokens.
        # 简单说就是prompt部分和mel_spec_overlap部分做了缓存。-34: 存下来用于下个chunk推理使用。这个chunk的-34帧，在下个chunk正好是:34（最开始的34帧），
        # 下个chunk的缓存恢复操作就在上边的if 中完成了，相当于<prompt_token|上个chunk的mel_overklap>部分都恢复了。<当前mel|下个mel_overlap>都是当前新产生的，然后被用于下边的新cache的存储。
        z_cache = torch.concat(
            [z[:, :, :prompt_len], z[:, :, -34:]], dim=2)  # [1,80, 34]
        mu_cache = torch.concat(
            [mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)  # [1,80, 34]
        # [B,D,T,2]=[1, 80, 34,2]
        cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1,
                                device=mu.device, dtype=mu.dtype)  # construct flow matching time steps, [n_timesteps+1]
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):
        """
        Fixed euler solver for ODEs. 使用欧拉方法求解偏微分方程（ODE）
        Args:
            x (torch.Tensor): random noise, [batch_size, n_feats, mel_timesteps]
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder, deemed as semantic token
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes, [batch_size, n_feats, mel_timesteps], mel_spec, mel_len1: 是0
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []  # 存储每个时间步的求解结果

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        # NOTE when flow run in amp mode, x.dtype is float32, which cause nan in trt fp16 inference, so set dtype=spks.dtype
        # 这里第0个维度（batch）是2。相当于2个副本，因为要进行CFG推理（无条件推理），和有条件推理
        x_in = torch.zeros([2, 80, x.size(2)],
                           device=x.device, dtype=spks.dtype)  # [B, 80, mel_timesteps]
        mask_in = torch.zeros(
            [2, 1, x.size(2)], device=x.device, dtype=spks.dtype)  # [2,1,mel_timesteps]
        mu_in = torch.zeros([2, 80, x.size(2)],
                            device=x.device, dtype=spks.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=spks.dtype)  # [2]
        spks_in = torch.zeros([2, 80], device=x.device,
                              dtype=spks.dtype)  # [2,80]
        cond_in = torch.zeros([2, 80, x.size(2)],
                              device=x.device, dtype=spks.dtype)  # [2,80,mel_timesteps]
        for step in range(1, len(t_span)):  # CFG inference
            # Classifier-Free Guidance inference introduced in VoiceBox
            # 通过遍历 t_span 的每个时间步进行迭代，模拟从初始噪声到目标mel-spectrogram的时间演化过程。
            # x_in=[2,80,206],x=[1,80,206]，x_in[:]=x相当于把x复制到x_in[0]和x_in[1]中。
            x_in[:] = x  # [2,80,206]
            mask_in[:] = mask  # [1,1,206]
            t_in[:] = t.unsqueeze(0)
            # following lines are total conditions
            mu_in[0] = mu  # [2,80,206], [0]有值
            spks_in[0] = spks  # [0]有值
            cond_in[0] = cond  # [0]有值
            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in,
                streaming
            )  # [2, 80, 206]，计算当前时间步微分dphi_dt
            dphi_dt, cfg_dphi_dt = torch.split(
                dphi_dt, [x.size(0), x.size(0)], dim=0)  # [1,80,206],[1,80,206]，分为两个部分对应于有条件和无条件的情况。提高生成的质量
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt -
                       self.inference_cfg_rate * cfg_dphi_dt)  # 公式14
            x = x + dt * dphi_dt  # [1,80,206], update x
            t = t + dt  # update t
            sol.append(x)
            if step < len(t_span) - 1:  # 当前不是最后一个
                dt = t_span[step + 1] - t  # 更新dt为下一步长

        return sol[-1].float()  # sol[-1].size()=[1,80,206]

    def forward_estimator(self, x, mask, mu, t, spks, cond, streaming=False):
        if isinstance(self.estimator, nn.Module):
            # Fig1.C
            return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        else:
            [estimator, stream], trt_engine = self.estimator.acquire_estimator()
            # NOTE need to synchronize when switching stream
            torch.cuda.current_stream().synchronize()
            with stream:
                estimator.set_input_shape('x', (2, 80, x.size(2)))
                estimator.set_input_shape('mask', (2, 1, x.size(2)))
                estimator.set_input_shape('mu', (2, 80, x.size(2)))
                estimator.set_input_shape('t', (2,))
                estimator.set_input_shape('spks', (2, 80))
                estimator.set_input_shape('cond', (2, 80, x.size(2)))
                data_ptrs = [x.contiguous().data_ptr(),
                             mask.contiguous().data_ptr(),
                             mu.contiguous().data_ptr(),
                             t.contiguous().data_ptr(),
                             spks.contiguous().data_ptr(),
                             cond.contiguous().data_ptr(),
                             x.data_ptr()]
                for i, j in enumerate(data_ptrs):
                    estimator.set_tensor_address(
                        trt_engine.get_tensor_name(i), j)
                # run trt engine
                assert estimator.execute_async_v3(
                    torch.cuda.current_stream().cuda_stream) is True
                torch.cuda.current_stream().synchronize()
            self.estimator.release_estimator(estimator, stream)
            return x

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, streaming=False):
        """Computes diffusion loss
        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)
        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.size()

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.squeeze(),
                              spks, cond, streaming=streaming)
        loss = F.mse_loss(pred * mask, u * mask,
                          reduction="sum") / (torch.sum(mask) * u.size(1))

        return loss, y
