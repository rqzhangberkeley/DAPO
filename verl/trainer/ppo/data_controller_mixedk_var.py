# verl/trainer/ppo/data_controller_mixedk_var.py
from __future__ import annotations
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
import numpy as np
import json, os

from verl import DataProto


class MixedKVarDataController:
    """
    Variable‑k controller for SPEED:
      • First phase (phase==1):
          - compute per‑UID std(acc); kept = std>0 (has 0 and 1), filtered = std==0
          - push ALL phase‑1 rows to training buffer
          - queue ONLY kept prompts for second phase
      • Second phase (phase==2): push rows directly to training buffer
      • Training: pick B UIDs with k>=2 rows in buffer and return ALL rows per UID (variable k)
    """

    def __init__(
        self,
        curriculum_enable: bool,
        gen_batch_size: int,
        train_batch_size: int,
        initial_n: int,
        n_continue: int,
        max_num_gen_batches: int,
        n_gpus: int,
        max_prompt_length: int,
        max_buffer_size: int,
        tokenizer=None,
        kept_file=None,
        filtered_file=None,
    ):
        self.curriculum_enable = curriculum_enable
        self.gen_batch_size = gen_batch_size
        self.train_batch_size = train_batch_size
        self.initial_n = initial_n
        self.n_continue = n_continue
        self.total_n_generations = initial_n + n_continue
        self.max_num_gen_batches = max_num_gen_batches
        self.n_gpus = n_gpus
        self.max_prompt_length = max_prompt_length

        self.prompts_for_first_generation_phase: Optional[DataProto] = None
        self.prompts_for_second_generation_phase: Optional[DataProto] = None
        self.prompts_for_training: Optional[DataProto] = None

        self.num_gen_batches = 0
        self.max_buffer_size = max_buffer_size

        assert self.n_continue % self.initial_n == 0, (
            f"n_continue must be divisible by initial_n: {self.n_continue=} % {self.initial_n=}"
        )
        self.multiplier = self.n_continue // self.initial_n

        self.tokenizer = tokenizer
        self.kept_file = kept_file
        self.filtered_file = filtered_file
        self.uid_to_prompt_text: Dict[str, str] = {}

    # ---------- utilities ----------
    @staticmethod
    def _uid_key(u) -> str:
        return u if isinstance(u, str) else str(u)

    def _attach_phase(self, dp: DataProto, phase_id: int) -> DataProto:
        non = dict(dp.non_tensor_batch)
        non["phase"] = np.full(dp.batch.batch_size[0], int(phase_id), dtype=np.int32)
        return DataProto(batch=dp.batch, non_tensor_batch=non, meta_info=dp.meta_info)

    def _selectable_uids(self, min_k: int = 2) -> List[str]:
        """UIDs with at least min_k rows currently in the training buffer."""
        if self.prompts_for_training is None:
            return []
        uids = [self._uid_key(u) for u in self.prompts_for_training.non_tensor_batch["uid"]]
        cnt = defaultdict(int)
        for u in uids:
            cnt[u] += 1
        return [u for u, k in cnt.items() if k >= min_k]

    # ---------- API ----------
    def is_ready_for_training(self) -> bool:
        return len(self._selectable_uids(min_k=2)) >= self.train_batch_size

    def add_new_prompts_first_phase(self, batch: DataProto) -> None:
        assert self.prompts_for_first_generation_phase is None, "phase‑1 batch already present"
        self.prompts_for_first_generation_phase = batch

        # cache uid -> prompt text for stable logging
        uids = batch.non_tensor_batch["uid"].tolist()
        attn = batch.batch["attention_mask"]
        inp  = batch.batch["input_ids"]
        for i, uid in enumerate(uids):
            uid = self._uid_key(uid)
            plen = int(attn[i].sum().item())
            pids = inp[i][-plen:]
            self.uid_to_prompt_text[uid] = self.tokenizer.decode(pids.tolist(), skip_special_tokens=True)

    def get_generation_inputs(self) -> DataProto:
        assert self.prompts_for_first_generation_phase is not None or self.prompts_for_second_generation_phase is not None, \
            "No prompts for generation."
        self.num_gen_batches += 1

        if self.prompts_for_second_generation_phase is None:
            batch = self.prompts_for_first_generation_phase
            self.prompts_for_first_generation_phase = None
            return self._attach_phase(batch, phase_id=1)

        # schedule second‑phase prompts (aligned to #GPUs); may mix with phase‑1 leftovers
        num_second = self.prompts_for_second_generation_phase.batch.batch_size[0]
        num_second_for_gen = num_second - (num_second % self.n_gpus)
        if num_second_for_gen == 0:
            batch = self.prompts_for_first_generation_phase
            self.prompts_for_first_generation_phase = None
            return self._attach_phase(batch, phase_id=1)

        rep_idx = np.repeat(np.arange(num_second_for_gen), self.multiplier).tolist()
        repeated_second = self.prompts_for_second_generation_phase[rep_idx]

        if self.prompts_for_first_generation_phase is not None:
            first = self._attach_phase(self.prompts_for_first_generation_phase, phase_id=1)
            second = self._attach_phase(repeated_second, phase_id=2)
            out = DataProto.concat([first, second])
            self.prompts_for_first_generation_phase = None
        else:
            out = self._attach_phase(repeated_second, phase_id=2)

        # keep leftovers
        if num_second % self.n_gpus == 0:
            self.prompts_for_second_generation_phase = None
        else:
            self.prompts_for_second_generation_phase = self.prompts_for_second_generation_phase[-(num_second % self.n_gpus):]
        return out

    def update_prompts(self, batch: DataProto, metric_name: str = "acc", global_step: int = 0) -> Tuple[List[str], List[int]]:
        uids  = batch.non_tensor_batch["uid"]
        phase = batch.non_tensor_batch["phase"]
        metric = batch.non_tensor_batch[metric_name]

        # split metrics by phase
        uid2m_first, uid2m_second = defaultdict(list), defaultdict(list)
        for u, m, p in zip(uids, metric, phase):
            (uid2m_first if p == 1 else uid2m_second)[u].append(float(m))

        # first‑phase UIDs (stable order)
        first_uids, seen = [], set()
        for u, p in zip(uids, phase):
            if p == 1 and u not in seen:
                seen.add(u); first_uids.append(u)

        # kept vs filtered by std>0
        uid2std = {u: float(np.std(uid2m_first[u])) for u in first_uids}
        kept = [u for u, s in uid2std.items() if s > 0.0]
        filtered = [u for u, s in uid2std.items() if s == 0.0]

        # indices to push: ALL phase‑1 rows (kept + filtered) and ALL phase‑2 rows
        add_idxs = [i for i, p in enumerate(phase) if p == 1] + [i for i, p in enumerate(phase) if p == 2]
        if add_idxs:
            rows = batch[add_idxs]
            self.prompts_for_training = rows if self.prompts_for_training is None else DataProto.concat([self.prompts_for_training, rows])

        # queue only kept prompts for phase‑2
        kept_first_prompt_idx, seen = [], set()
        for i, (u, p) in enumerate(zip(uids, phase)):
            if p == 1 and u in kept and u not in seen:
                kept_first_prompt_idx.append(i); seen.add(u)
        if kept_first_prompt_idx:
            kept_prompts = batch[kept_first_prompt_idx].select(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=[k for k in batch.non_tensor_batch.keys() if k not in ["score","acc","reward","phase"]],
                meta_info_keys=None
            ).truncate(start=0, end=self.max_prompt_length)
            self.prompts_for_second_generation_phase = kept_prompts if self.prompts_for_second_generation_phase is None else DataProto.concat([self.prompts_for_second_generation_phase, kept_prompts])

        # filtered logging (phase‑1 rows only)
        if self.filtered_file is not None and filtered:
            for uid in filtered:
                kuid = self._uid_key(uid)
                idxs = [i for i, (u, p) in enumerate(zip(uids, phase)) if u == uid and p == 1]
                if not idxs: 
                    continue
                sub = batch.select_idxs(idxs)
                avg = float(np.mean(sub.non_tensor_batch["acc"].astype(np.float32)))
                gt = sub.non_tensor_batch["reward_model"][0]["ground_truth"]
                entry = {
                    "global_step": global_step,
                    "prompt": self.uid_to_prompt_text.get(kuid, "<MISSING_PROMPT_CACHE>"),
                    "average_acc": avg,
                    "ground_truth": gt,
                    "n_responses": len(idxs),
                }
                self.filtered_file.write(json.dumps(entry) + "\n")
                self.filtered_file.flush(); os.fsync(self.filtered_file.fileno())

        return [], []  # unused by the trainer

    def get_training_data(self, prompt_batch_size: int, global_step: int = 0, max_per_uid: Optional[int] = None) -> DataProto:
        assert self.prompts_for_training is not None, "Training buffer is empty."
        selectable = self._selectable_uids(min_k=2)
        if len(selectable) < prompt_batch_size:
            raise RuntimeError(f"Not enough selectable UIDs: have {len(selectable)}, need {prompt_batch_size}")

        # stable UID order by first appearance
        ordered, seen = [], set()
        for u in [self._uid_key(x) for x in self.prompts_for_training.non_tensor_batch["uid"]]:
            if u in selectable and u not in seen:
                seen.add(u); ordered.append(u)
        chosen = ordered[:prompt_batch_size]

        # gather indices for all rows of chosen UIDs
        uid_to_indices = defaultdict(list)
        for i, u in enumerate(self.prompts_for_training.non_tensor_batch["uid"]):
            uid_to_indices[self._uid_key(u)].append(i)
        training_indices = []
        for u in chosen:
            idxs = uid_to_indices[u]
            if max_per_uid is not None:
                idxs = idxs[:max_per_uid]
            training_indices.extend(idxs)

        training = self.prompts_for_training[training_indices]

        # kept logging (variable‑k)
        if self.kept_file is not None:
            seen = set()
            for i, u in enumerate(training.non_tensor_batch["uid"]):
                ku = self._uid_key(u)
                if ku in seen: 
                    continue
                seen.add(ku)
                idxs = [j for j, uu in enumerate(training.non_tensor_batch["uid"]) if self._uid_key(uu) == ku]
                sub = training.select_idxs(idxs)
                avg = float(np.mean(sub.non_tensor_batch["acc"].astype(np.float32)))
                gt = sub.non_tensor_batch["reward_model"][0]["ground_truth"]
                entry = {
                    "global_step": global_step,
                    "prompt": self.uid_to_prompt_text.get(ku, "<MISSING_PROMPT_CACHE>"),
                    "average_acc": avg,
                    "ground_truth": gt,
                    "n_responses": len(idxs),
                }
                self.kept_file.write(json.dumps(entry) + "\n")
                self.kept_file.flush(); os.fsync(self.kept_file.fileno())

        # remove consumed rows from buffer
        remain = [i for i in range(self.prompts_for_training.batch.batch_size[0]) if i not in training_indices]
        self.prompts_for_training = self.prompts_for_training[remain]
        return training

    def reset_num_gen_batches(self):
        self.num_gen_batches = 0

    def get_bsz_for_first_generation_phase(self) -> int:
        """Match original controller API so trainer can prefetch safely."""
        if self.prompts_for_first_generation_phase is None:
            return 0
        return self.prompts_for_first_generation_phase.batch.batch_size[0]

    def get_bsz_for_second_generation_phase(self) -> int:
        """Match original controller API so trainer can prefetch safely."""
        if self.prompts_for_second_generation_phase is None:
            return 0
        return self.prompts_for_second_generation_phase.batch.batch_size[0]

