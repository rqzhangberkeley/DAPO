# recipe/dapo/src/actor_mixedk.py
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import torch

# Try common locations for the base actor
try:
    from verl.trainer.ppo.actor import PPOActor as _BaseActor  # type: ignore
except Exception:
    from verl.trainer.ppo.actor_worker import PPOActor as _BaseActor  # type: ignore


class PPOActorMixedK(_BaseActor):
    """
    PPO actor that:
      - uses batch["advantages"] if provided
      - else computes RLOO inside the minibatch by grouping rows by UID
    This avoids any fixed-N assumptions during loss computation.
    """

    @staticmethod
    def _compute_adv_minibatch(non_tensor_batch: Dict[str, Any]) -> torch.Tensor:
        uids = np.asarray(non_tensor_batch["uid"], dtype=object)
        if "acc" in non_tensor_batch:
            rewards = np.asarray(non_tensor_batch["acc"], dtype=np.float32)
        else:
            rewards = np.asarray(non_tensor_batch["score"], dtype=np.float32)

        s, c = {}, {}
        for u, r in zip(uids, rewards):
            s[u] = s.get(u, 0.0) + float(r)
            c[u] = c.get(u, 0) + 1

        adv = np.empty_like(rewards)
        for i, (u, r) in enumerate(zip(uids, rewards)):
            k, sr = c[u], s[u]
            adv[i] = 0.0 if k <= 1 else (k * float(r) - sr) / max(1, k - 1)
        return torch.from_numpy(adv.astype(np.float32))

    def _get_advantages(self, batch) -> torch.Tensor:
        if isinstance(batch, dict) and "advantages" in batch:
            adv = batch["advantages"]
            return adv if torch.is_tensor(adv) else torch.as_tensor(adv, dtype=torch.float32, device=self.device)

        if hasattr(batch, "batch") and hasattr(batch, "non_tensor_batch"):  # DataProto
            adv = self._compute_adv_minibatch(batch.non_tensor_batch)
        else:
            # Fallback if your trainer passes a dict with non-tensors packed separately
            non = batch.get("non_tensors", None)
            assert non is not None, "non_tensors missing for advantage computation"
            adv = self._compute_adv_minibatch(non)
        return adv.to(device=self.device, dtype=torch.float32)

    # If your base actor calls a specific policy-loss hook, override it:
    def compute_policy_loss(self, batch, new_logp, old_logp):
        advantages = self._get_advantages(batch).detach()
        ratio = torch.exp(new_logp - old_logp)
        clip_low, clip_high = 1.0 - self.clip_ratio_low, 1.0 + self.clip_ratio_high
        pg1 = -advantages * ratio
        pg2 = -advantages * torch.clamp(ratio, clip_low, clip_high)
        return torch.mean(torch.max(pg1, pg2))
