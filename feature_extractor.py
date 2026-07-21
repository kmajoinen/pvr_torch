"""
Frozen-PVR fast path for explicit training loops (train_sac.py): encodes
each frame ONCE, at env-step time, and hands the training loop a flat
feature vector. The replay buffer stores features instead of pixels, and
gradient steps never touch the encoder -- with a frozen encoder the
features are a fixed function of the frame, so re-encoding replayed pixels
on every gradient step would be pure waste.

Backed by src/embeddings.py's EmbeddingNet (same as train_bc.py): a new PVR
checkpoint needs a new configs/embedding/<name>.yaml, not new code here.

For the finetuned case (encoder trains jointly with the RL head, SB3 only)
see feature_extractor_sb3.py's PVRFeaturesExtractor.
"""
import numpy as np
import torch as th
import gymnasium as gym


class FrozenEncoderWrapper(gym.ObservationWrapper):
    """
    Encodes each uint8 HWC frame into a (out_size,) float32 feature vector,
    once, at env-step time -- so the replay buffer stores features and
    gradient updates never touch the encoder.
    """

    def __init__(
        self,
        env: gym.Env,
        embedding_net,            # src.embeddings.EmbeddingNet, already built
        amp_bf16: bool = True,
    ):
        super().__init__(env)
        self.embedding_net = embedding_net
        self.embedding_net.eval()
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (int(embedding_net.out_size),), np.float32
        )
        self.amp_bf16 = amp_bf16 and th.cuda.is_available()

    @th.inference_mode()
    def observation(self, obs: np.ndarray) -> np.ndarray:
        # single frame (H, W, C) uint8 -> (1, C, H, W) uint8, the same
        # layout/dtype EmbeddingNet.encode expects
        x = th.as_tensor(np.ascontiguousarray(obs)).permute(2, 0, 1).unsqueeze(0)
        with th.autocast("cuda", dtype=th.bfloat16, enabled=self.amp_bf16):
            feats = self.embedding_net.encode(x)
        feats = th.as_tensor(feats).float()
        return feats.squeeze(0).cpu().numpy()
