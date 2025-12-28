import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models import UNet1DModel

class SigLIPDiffusionPolicy(nn.Module):
    def __init__(self, action_dim=7, obs_dim=7, horizon=16):
        super().__init__()
        
        # 1. 视觉特征提取 (SigLIP)
        # 选用 so400m 版本，兼顾速度与理解力
        self.visual_backbone = AutoModel.from_pretrained("google/siglip-so400m-patch14-224").vision_model
        self.vis_feature_dim = 1152 
        
        # 2. 本体感知投影 (机械臂当前状态，如关节角)
        self.proprio_proj = nn.Linear(obs_dim, 256)
        
        # 3. 扩散模型核心：1D-UNet
        # 输入是 [batch, horizon, action_dim]，条件是视觉+状态特征
        self.noise_pred_net = UNet1DModel(
            in_channels=action_dim,
            global_cond_channels=self.vis_feature_dim + 256, # 融合后的特征维度
            down_block_types=('DownResnetBlock1D', 'DownResnetBlock1D', 'DownResnetBlock1D'),
            up_block_types=('UpResnetBlock1D', 'UpResnetBlock1D', 'UpResnetBlock1D'),
            block_out_channels=(256, 512, 1024),
        )
        
        # 4. 噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2', # 这种调度对机器人动作更平滑
            prediction_type='epsilon' 
        )
        
        self.horizon = horizon
        self.action_dim = action_dim

    def forward(self, image, state, noisy_action, timesteps):
        """
        训练时调用：输入带噪声的动作，预测噪声
        """
        # 提取视觉特征 [B, vis_feature_dim]
        vis_feat = self.visual_backbone(image).last_hidden_state[:, 0]
        
        # 提取状态特征 [B, 256]
        state_feat = self.proprio_proj(state)
        
        # 拼接条件特征
        obs_cond = torch.cat([vis_feat, state_feat], dim=-1) # [B, 1408]
        
        # 预测噪声
        noise_pred = self.noise_pred_net(
            sample=noisy_action,      # [B, horizon, action_dim]
            timestep=timesteps,       # [B]
            global_cond=obs_cond      # [B, 1408]
        ).sample
        
        return noise_pred

    @torch.no_grad()
    def sample_action(self, image, state):
        """
        推理时调用：从纯噪声生成动作序列
        """
        device = next(self.parameters()).device
        bs = image.shape[0]
        
        # 初始化纯噪声动作
        action_seq = torch.randn((bs, self.horizon, self.action_dim), device=device)
        
        # 预先计算条件特征
        vis_feat = self.visual_backbone(image).last_hidden_state[:, 0]
        state_feat = self.proprio_proj(state)
        obs_cond = torch.cat([vis_feat, state_feat], dim=-1)

        # 迭代去噪
        self.noise_scheduler.set_timesteps(50) # 推理时可以用更少的步数提速
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                sample=action_seq,
                timestep=k,
                global_cond=obs_cond
            ).sample
            
            action_seq = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=action_seq
            ).prev_sample
            
        return action_seq