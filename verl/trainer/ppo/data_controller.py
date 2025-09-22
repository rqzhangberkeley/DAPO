from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from verl import DataProto
import torch, json, os, time, random, math, logging
from collections import defaultdict

class DataController:
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
        tokenizer = None, # tokenizer.
        kept_file = None, # the file to save the kept prompts.
        filtered_file = None, # the file to save the filtered prompts.
    ):
        """Initialize the DataController.
        
        Args:
            curriculum_enable: Whether to use curriculum learning
            gen_batch_size: Number of prompts to generate in each batch
            train_batch_size: Number of qualified prompts needed for training
            initial_n: Number of initial responses per prompt (n)
            n_continue: Number of additional responses for qualified prompts (N-n)
            max_num_gen_batches: Maximum number of generation batches to try
        """
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
        self.num_prompts_for_training = 0
        self.num_gen_batches = 0
        self.max_buffer_size = max_buffer_size
        assert self.max_buffer_size >= self.train_batch_size * self.total_n_generations, f"The maximum buffer size should be greater than or equal to the training batch size times the total number of generations. Got {self.max_buffer_size=} and {self.train_batch_size=} and {self.total_n_generations=}"
        assert self.n_continue % self.initial_n == 0, f"The number of additional responses should be divisible by the number of initial responses. Got {self.n_continue=} and {self.initial_n=}"
        self.multiplier = self.n_continue // self.initial_n

        self.tokenizer = tokenizer
        self.kept_file = kept_file # the file to save the kept prompts
        self.filtered_file = filtered_file # the file to save the filtered prompts

    def is_ready_for_training(self) -> bool:
        if self.prompts_for_training is None:
            return False
        elif self.num_prompts_for_training >= self.train_batch_size:
            return True
        else:
            return False
    
    def get_num_prompts_for_training(self) -> int:
        return self.num_prompts_for_training

    def get_num_gen_batches(self) -> int:
        return self.num_gen_batches

    def get_bsz_for_first_generation_phase(self) -> int:
        if self.prompts_for_first_generation_phase is None:
            return 0
        else: 
            return self.prompts_for_first_generation_phase.batch.batch_size[0]
        
    def get_bsz_for_second_generation_phase(self) -> int:
        if self.prompts_for_second_generation_phase is None:
            return 0
        else:
            return self.prompts_for_second_generation_phase.batch.batch_size[0]

    def add_new_prompts_first_phase(self, batch: DataProto) -> None:
        """Add new prompts to be processed in the current batch."""
        assert self.prompts_for_first_generation_phase is None, "First generation phase is already done. Before we call add_new_prompts, the prompts_for_first_generation_phase should be None."
        self.prompts_for_first_generation_phase = batch

        # record the information of the prompts. Added by RZ.
        if not hasattr(self, "uid_to_prompt_text"):
            self.uid_to_prompt_text = {}
            self.uid_to_prompt_length = {} # we shall know this to store the responses.
        
        # for i, uid in enumerate(set(batch.non_tensor_batch["uid"])):
        #     prompt_len = int(batch.batch["attention_mask"][i].sum().item())
        #     prompt_ids = batch.batch["input_ids"][i][(-prompt_len):]
        #     prompt_text = self.tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=True)
        #     self.uid_to_prompt_text[uid] = prompt_text
        #     self.uid_to_prompt_length[uid] = prompt_len

        uids = batch.non_tensor_batch["uid"].tolist()
        attn = batch.batch["attention_mask"]
        inp  = batch.batch["input_ids"]

        for i, uid in enumerate(uids):
            prompt_len = int(attn[i].sum().item())
            prompt_ids = inp[i][-prompt_len:]
            prompt_text = self.tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=True)
            self.uid_to_prompt_text[uid] = prompt_text
            self.uid_to_prompt_length[uid] = prompt_len

    
    def get_generation_inputs(self) -> DataProto:
        """Get the prompts for the next generation step."""
        if self.prompts_for_first_generation_phase is None and self.prompts_for_second_generation_phase is None:
            raise ValueError("No prompts for generation.")

        self.num_gen_batches += 1
        if self.num_gen_batches >= self.max_num_gen_batches:
            raise ValueError(f"Generated too many batches. We have {self.num_gen_batches=} >= {self.max_num_gen_batches=}.")
        
        if self.prompts_for_second_generation_phase is None:
            assert self.prompts_for_first_generation_phase.batch.batch_size[0] % self.n_gpus == 0, "The number of prompts for the first generation phase should be divisible by the number of GPUs."
            
            batch = self.prompts_for_first_generation_phase
            self.prompts_for_first_generation_phase = None # remove the prompts.
            # return batch
            return self._attach_phase(batch, phase_id=1)
        else:
            num_second_phase_prompts = self.prompts_for_second_generation_phase.batch.batch_size[0]
            num_second_phase_prompts_for_generation = num_second_phase_prompts - (num_second_phase_prompts % self.n_gpus) # RZ: The number of prompts must be a multiple of the number of GPUs.
            print(f"Originally we have {num_second_phase_prompts} second phase prompts. After batchings, we have {num_second_phase_prompts_for_generation} second phase prompts for generation.")

            if num_second_phase_prompts_for_generation == 0:
                batch = self.prompts_for_first_generation_phase
                self.prompts_for_first_generation_phase = None # remove the prompts.
                # return batch
                return self._attach_phase(batch, phase_id=1)
            else:
                repeated_indices = []
                for i in range(num_second_phase_prompts_for_generation):
                    repeated_indices.extend([i] * self.multiplier)
                repeated_second_phase = self.prompts_for_second_generation_phase[repeated_indices] # multipy the second phase prompts.
                
                if self.prompts_for_first_generation_phase is not None:
                    # batch = DataProto.concat([self.prompts_for_first_generation_phase, repeated_second_phase])
                    first_batch = self._attach_phase(self.prompts_for_first_generation_phase, phase_id=1)
                    second_batch = self._attach_phase(repeated_second_phase, phase_id=2)
                    batch = DataProto.concat([first_batch, second_batch])
                else:
                    # batch = repeated_second_phase
                    batch = self._attach_phase(repeated_second_phase, phase_id=2)
                    
                self.prompts_for_first_generation_phase = None
                
                if num_second_phase_prompts % self.n_gpus == 0:
                    self.prompts_for_second_generation_phase = None
                else:
                    self.prompts_for_second_generation_phase = self.prompts_for_second_generation_phase[-(num_second_phase_prompts % self.n_gpus):]
                return batch

    def update_prompts(self, batch: DataProto, metric_name: str = "acc", global_step: int = 0) -> Tuple[List[str], List[int]]:
        """Update prompt statistics based on the generated responses and their rewards.
        
        There are two types of prompts in 'batch':
            one with self.initial_n responses, and the other with self.n_continue responses.
        For the first type, those are the prompt that finish the first generation phase.
        For the second type, those are the prompt that finish the second generation phase and we can use those for training.
        """

        # # Collect the sequence reward for each trajectory
        # prompt_uid2metric_vals = defaultdict(list)
        # for uid, metric_val in zip(
        #     batch.non_tensor_batch["uid"], batch.non_tensor_batch[metric_name]
        # ):
        #     prompt_uid2metric_vals[uid].append(metric_val)

        # # categorize the prompts into two types.
        # prompts_uids_after_first_phase = [
        #     uid
        #     for uid, _ in prompt_uid2metric_vals.items()
        #     if len(prompt_uid2metric_vals[uid]) == self.initial_n
        # ]
        # prompts_uids_after_second_phase = [
        #     uid
        #     for uid, _ in prompt_uid2metric_vals.items()
        #     if len(prompt_uid2metric_vals[uid]) == self.n_continue
        # ]

        ######### Updated #########
        # Collect per-phase metrics and classify by explicit phase tag
        uids   = batch.non_tensor_batch["uid"]
        metric = batch.non_tensor_batch[metric_name]
        phase  = batch.non_tensor_batch.get("phase", None)
        assert phase is not None, "Expected 'phase' in non_tensor_batch; ensure get_generation_inputs attaches it."

        uid2metrics_first  = defaultdict(list)
        uid2metrics_second = defaultdict(list)
        for u, m, p in zip(uids, metric, phase):
            if p == 1:
                uid2metrics_first[u].append(m)
            else:
                uid2metrics_second[u].append(m)

        # Stable, de-duplicated UIDs per phase (preserve batch order)
        prompts_uids_after_first_phase, seen = [], set()
        for u, p in zip(uids, phase):
            if p == 1 and u not in seen:
                seen.add(u); prompts_uids_after_first_phase.append(u)
        prompts_uids_after_second_phase, seen = [], set()
        for u, p in zip(uids, phase):
            if p == 2 and u not in seen:
                seen.add(u); prompts_uids_after_second_phase.append(u)
        ######### End ofUpdated #########

        # Deal with the first-class prompts.
        # prompt_uid2metric_std = {}
        # for prompt_uid in prompts_uids_after_first_phase:
        #     prompt_uid2metric_std[prompt_uid] = np.std(prompt_uid2metric_vals[prompt_uid])

        ######### Updated #########
        prompt_uid2metric_std = {u: np.std(uid2metrics_first[u]) for u in prompts_uids_after_first_phase}
        ######### End of Updated #########
         
        kept_prompt_uids = [
            uid
            for uid, std in prompt_uid2metric_std.items()
            if std > 0
        ] # RZ: The qualified prompts that have pass rate between 0 and 1.
        # kept_traj_idxs = []
        # kept_traj_idxs_unique = []
        # kept_prompt_uids_unique = []
        # for idx, traj_from_prompt_uid in enumerate(batch.non_tensor_batch["uid"]):
        #     if traj_from_prompt_uid in kept_prompt_uids:
        #         kept_traj_idxs.append(idx)
        #         if traj_from_prompt_uid not in kept_prompt_uids_unique:
        #             kept_prompt_uids_unique.append(traj_from_prompt_uid)
        #             kept_traj_idxs_unique.append(idx)

        ######### Updated #########
        kept_traj_idxs = []
        kept_traj_idxs_unique = []
        kept_prompt_uids_unique = []
        for idx, (u, p) in enumerate(zip(uids, phase)):
            if p == 1 and u in kept_prompt_uids:
                kept_traj_idxs.append(idx)
                if u not in kept_prompt_uids_unique:
                    kept_prompt_uids_unique.append(u)
                    kept_traj_idxs_unique.append(idx)
        ######### End of Updated #########
        qualified_prompts_responses_after_first_phase = batch[kept_traj_idxs] # DataProto
        qualified_prompts_after_first_phase = batch[kept_traj_idxs_unique].select(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[key for key in batch.non_tensor_batch.keys() if key not in ['score', 'acc', 'reward', 'phase']],
            meta_info_keys=None
        ).truncate(start=0, end=self.max_prompt_length) # DataProto. But only have the prompt part. No responses.

        self.prompts_for_second_generation_phase = qualified_prompts_after_first_phase if self.prompts_for_second_generation_phase is None else DataProto.concat([self.prompts_for_second_generation_phase, qualified_prompts_after_first_phase])

        # Add the responses from the first-class prompts to the training set.
        self.prompts_for_training = qualified_prompts_responses_after_first_phase if self.prompts_for_training is None else DataProto.concat([self.prompts_for_training, qualified_prompts_responses_after_first_phase])
        
        # Deal with the second-class prompts.
        # traj_idxs = []
        # if prompts_uids_after_second_phase:
        #     for idx, traj_from_prompt_uid in enumerate(batch.non_tensor_batch["uid"]):
        #         if traj_from_prompt_uid in prompts_uids_after_second_phase:
        #             traj_idxs.append(idx)
        # responses_after_second_phase = batch[traj_idxs]
        # self.prompts_for_training = responses_after_second_phase if self.prompts_for_training is None else DataProto.concat([self.prompts_for_training, responses_after_second_phase])
        # self.num_prompts_for_training += len(set(prompts_uids_after_second_phase))

        traj_idxs = [i for i, p in enumerate(phase) if p == 2]
        responses_after_second_phase = batch[traj_idxs]
        self.prompts_for_training = responses_after_second_phase if self.prompts_for_training is None else DataProto.concat([self.prompts_for_training, responses_after_second_phase])
        # recompute readiness from buffer
        if self.prompts_for_training is not None:
            uid_to_count = defaultdict(int)
            for u in self.prompts_for_training.non_tensor_batch["uid"]:
                uid_to_count[u] += 1
            self.num_prompts_for_training = sum(1 for c in uid_to_count.values() if c >= self.total_n_generations)
        else:
            self.num_prompts_for_training = 0
        

        # save the filtered prompts, and their responses, as well as their information.
        filtered_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items()  if std == 0]
        for uid in filtered_prompt_uids:
            traj_indices = [idx for idx, u in enumerate(batch.non_tensor_batch["uid"]) if u == uid]
            # print(batch.select_idxs(traj_indices).non_tensor_batch)
            average_acc = batch.select_idxs(traj_indices).non_tensor_batch["acc"].mean().item()
            ground_truth = batch.select_idxs(traj_indices).non_tensor_batch['reward_model'][0]["ground_truth"]
            log_entry = {
                "global_step": global_step,
                "prompt": self.uid_to_prompt_text[uid], 
                "average_acc": average_acc,
                "ground_truth": ground_truth,
                "n_responses": len(traj_indices)
            }
            self.filtered_file.write(json.dumps(log_entry) + "\n")
            self.filtered_file.flush()
            os.fsync(self.filtered_file.fileno())

            # remove the information of the prompt to avoid this dictionary being too large.
            if uid in self.uid_to_prompt_text:
                del self.uid_to_prompt_text[uid]
            if uid in self.uid_to_prompt_length:
                del self.uid_to_prompt_length[uid]

    def get_training_data(self,
                          prompt_batch_size: int,
                          global_step: int = 0
                          ) -> DataProto:
        """Get qualified prompts for training."""

        assert prompt_batch_size == self.train_batch_size, f"The prompt batch size should be equal to the training batch size. Got {prompt_batch_size=} and {self.train_batch_size=}"

        # Get the indices of all qualified prompts (with self.total_n_generations responses).
        unique_uids = []
        for uid in self.prompts_for_training.non_tensor_batch["uid"]:
            if uid not in unique_uids:
                unique_uids.append(uid)
        uid_to_indices = defaultdict(list)
        for idx, uid in enumerate(self.prompts_for_training.non_tensor_batch["uid"]):
            uid_to_indices[uid].append(idx)
        training_indices = []
        for uid in unique_uids:
            if len(uid_to_indices[uid]) == self.total_n_generations:
                training_indices.extend(uid_to_indices[uid])

        # Check if the number of training indices is greater than or equal to the training response batch size.
        train_resp_bsz = prompt_batch_size * self.total_n_generations
        assert len(training_indices) >= train_resp_bsz, f"The number of training indices should be greater than or equal to the training response batch size. Got {len(training_indices)=} and {train_resp_bsz=}"

        # get the training data.
        training_indices = training_indices[:train_resp_bsz]
        training_data = self.prompts_for_training[training_indices]

        # save the training data to kept_file.
        # for idx, uid in enumerate(set(training_data.non_tensor_batch["uid"])):
        saved_uids = set()
        for idx, uid in enumerate(training_data.non_tensor_batch["uid"]):
            if uid in saved_uids: 
                continue
            saved_uids.add(uid)
            traj_indices = [idx for idx, u in enumerate(training_data.non_tensor_batch["uid"]) if u == uid]
            average_acc = training_data.select_idxs(traj_indices).non_tensor_batch["acc"].mean().item()
            ground_truth = training_data.select_idxs(traj_indices).non_tensor_batch['reward_model'][0]["ground_truth"]
            log_entry = {
                "global_step": global_step,
                "prompt": self.uid_to_prompt_text[uid], 
                "average_acc": average_acc,
                "ground_truth": ground_truth,
                "n_responses": len(traj_indices)
            }
            self.kept_file.write(json.dumps(log_entry) + "\n")
            self.kept_file.flush()
            os.fsync(self.kept_file.fileno())

            # remove the information of the prompt to avoid this dictionary being too large.
            if uid in self.uid_to_prompt_text:
                del self.uid_to_prompt_text[uid]
            if uid in self.uid_to_prompt_length:
                del self.uid_to_prompt_length[uid]

        # update the attributes.
        self.num_prompts_for_training -= self.train_batch_size
        remaining_indices = [i for i in range(self.prompts_for_training.batch.batch_size[0]) if i not in training_indices]
        self.prompts_for_training = self.prompts_for_training[remaining_indices]

        # Double Check if the training data has prompt_batch_size prompts and self.total_n_generations responses for each prompt.
        training_uids_to_indices = defaultdict(list)
        for idx, uid in enumerate(training_data.non_tensor_batch["uid"]):
            training_uids_to_indices[uid].append(idx)
        for uid in training_uids_to_indices.keys():
            assert len(training_uids_to_indices[uid]) == self.total_n_generations, f"The number of training indices for each prompt should be equal to the total number of generations. Got {len(training_uids_to_indices[uid])=} and {self.total_n_generations=}"
        assert len(training_uids_to_indices) == prompt_batch_size, f"The number of training uids should be equal to the prompt batch size. Got {len(training_uids_to_indices)=} and {prompt_batch_size=}"

        return training_data

    def reset_num_gen_batches(self):
        self.num_gen_batches = 0

    def _attach_phase(self, dp: DataProto, phase_id: int) -> DataProto:
        '''
        Attach a phase id to the data proto. A helper function.
        '''
        non = dict(dp.non_tensor_batch)
        non["phase"] = np.full(dp.batch.batch_size[0], int(phase_id), dtype=np.int32)
        return DataProto(batch=dp.batch, non_tensor_batch=non, meta_info=dp.meta_info)