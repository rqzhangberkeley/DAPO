# recipe/dapo/src/fast_dapo_ray_trainer_mixedk.py
from __future__ import annotations
import numpy as np
import torch

# Use the Ray-based trainer as base
from recipe.dapo.src.fast_dapo_ray_trainer import RayFastDAPOTrainer as _BaseTrainer
from verl.trainer.ppo.data_controller_mixedk_var import MixedKVarDataController


class FastDAPORayTrainerMixedK(_BaseTrainer):
    """
    Mixed‑k trainer that performs exactly ONE optimizer step per RL step:
      • prefetch first‑phase prompts when controller is empty
      • generate until the controller is ready (has B selectable UIDs)
      • build one variable‑k training batch (all rows for each of B UIDs)
      • precompute RLOO advantages across the FULL batch
      • call actor.update(...) ONCE with the FULL batch (no trainer-side mini/micro batching)
    """

    def __init__(
        self,
        config,
        tokenizer,
        processor,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls,
        reward_fn,
        val_reward_fn,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )

        dcfg = config.data
        self.data_controller = MixedKVarDataController(
            curriculum_enable=getattr(config.curriculum, "enable", False),
            gen_batch_size=dcfg.gen_batch_size,
            train_batch_size=dcfg.train_batch_size,
            initial_n=getattr(config.actor_rollout_ref.rollout, "n", 12),
            n_continue=getattr(config.actor_rollout_ref.rollout, "n_continue", 12),
            max_num_gen_batches=getattr(config.algorithm.filter_groups, "max_num_gen_batches", 50),
            n_gpus=getattr(config.trainer, "n_gpus_per_node", 1),
            max_prompt_length=dcfg.max_prompt_length,
            max_buffer_size=getattr(dcfg, "max_buffer_size", dcfg.train_batch_size * 64),
            tokenizer=self.tokenizer,
            kept_file=getattr(self, "kept_file", None),
            filtered_file=getattr(self, "filtered_file", None),
        )

    # ---------- Advantage precomputation across the FULL training batch (variable‑k RLOO) ----------
    @staticmethod
    def _attach_variable_k_rloo(training_data):
        uids = np.array([str(u) for u in training_data.non_tensor_batch["uid"]], dtype=object)
        if "acc" in training_data.non_tensor_batch:
            rewards = np.asarray(training_data.non_tensor_batch["acc"], dtype=np.float32)
        else:
            rewards = np.asarray(training_data.non_tensor_batch["score"], dtype=np.float32)

        uid2sum, uid2cnt = {}, {}
        for u, r in zip(uids, rewards):
            uid2sum[u] = uid2sum.get(u, 0.0) + float(r)
            uid2cnt[u] = uid2cnt.get(u, 0) + 1

        adv = np.empty_like(rewards)
        for i, (u, r) in enumerate(zip(uids, rewards)):
            k = uid2cnt[u]; sr = uid2sum[u]
            adv[i] = 0.0 if k <= 1 else (k * float(r) - sr) / max(1, k - 1)

        any_tensor = next(iter(training_data.batch.values()))
        training_data.batch["advantages"] = torch.from_numpy(adv.astype(np.float32)).to(any_tensor.device)
        return training_data

    # ---------- Helper to call your actor on a batch (one optimizer step) ----------
    def _actor_update_on_batch(self, batch):
        actor = (
            getattr(self, "actor", None)
            or getattr(self, "actor_worker", None)
            or getattr(self, "actor_rollout_ref", None)
        )
        if actor is None:
            raise RuntimeError("Actor object not found on trainer (tried: actor / actor_worker / actor_rollout_ref).")

        for name in ("update_on_batch", "update", "step", "train_on_batch"):
            if hasattr(actor, name):
                return getattr(actor, name)(batch)

        if hasattr(actor, "actor"):
            for name in ("update_on_batch", "update", "step", "train_on_batch"):
                if hasattr(actor.actor, name):
                    return getattr(actor.actor, name)(batch)

        raise RuntimeError("No suitable actor update method found. Expose e.g. `.update_on_batch(DataProto)`.")

    # ---------- NEW: prefetch first‑phase prompts when controller is empty ----------
    def _prefetch_if_needed(self):
        need_first = (self.data_controller.get_bsz_for_first_generation_phase() == 0)
        need_second = (self.data_controller.get_bsz_for_second_generation_phase() == 0)
        if need_first and need_second:
            # --- IMPORTANT ---
            # The base RayFastDAPOTrainer exposes a generator for prompt batches.
            # In most repos it is named `get_next_prompt_batch` and returns a DataProto.
            # If your local name differs, change the method name on the next line.
            prompt_batch = self.get_next_prompt_batch(self.config.data.gen_batch_size)
            self.data_controller.add_new_prompts_first_phase(prompt_batch)

    # ---------- Training loop: ONE optimizer step per RL step ----------
    def fit(self):
        total_epochs = self.config.trainer.total_epochs

        for self.global_steps in range(1, total_epochs + 1):
            # 1) Generate until we have B selectable UIDs
            while not self.data_controller.is_ready_for_training():
                # ensure the controller has a first‑phase batch to start generating
                self._prefetch_if_needed()

                gen_inputs = self.data_controller.get_generation_inputs()        # phase‑tagged DataProto
                gen_outputs = self.actor_rollout.generate_sequences(gen_inputs)  # rollout produces responses & metrics
                self.data_controller.update_prompts(gen_outputs, metric_name="acc", global_step=self.global_steps)

            # 2) Build one variable‑k training batch (all rows for each of B UIDs)
            training_data = self.data_controller.get_training_data(
                prompt_batch_size=self.config.data.train_batch_size,
                global_step=self.global_steps
            )

            # 3) Attach RLOO per row across THE FULL BATCH
            training_data = self._attach_variable_k_rloo(training_data)

            # 4) ONE optimizer step using the FULL batch
            self._actor_update_on_batch(training_data)

            # 5) Validation / checkpoint as in base trainer
            if self.global_steps % self.config.trainer.test_freq == 0:
                self.validate(self.global_steps)
            if self.global_steps % self.config.trainer.save_freq == 0:
                self.save_checkpoint(self.global_steps)
