# Experiment Log

#### Dataset Preparation
- DAPO-17k. The dataset in https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k is not equipped with the default prompt of Qwen2.5 series. Moreover, there are actually 17k prompts (that is why it is called DAPO-17k). Each prompt is repeated 100 times so the data size is 1.79M. So we do the following edits:
  * We use the prompts in ./examples/data-preprocess/math_dataset.py. 
  * For base models, we ignore the chat template. 
  * We deduplicate the dataset, and sample 1000 of them as the test set. The remaining 16398 is the training set.

#### Initial Test to Dataset
- Qwen2.5-1.5B
    * DAPO-17k-test-1000: pass@1 = 0.05.
    * pass@32: {'zeros': 0.54, 'ones': 0.0, '0.0-0.1': 0.299, '0.1-0.2': 0.091, '0.2-0.3': 0.042, '0.3-0.4': 0.013, '0.4-0.5': 0.01, '0.5-0.6': 0.004, '0.6-0.7': 0.001, '0.7-0.8': 0.0, '0.8-0.9': 0.0, '0.9-1.0': 0.0} 
- Qwen2.5-Math-1.5B (with chat template)
    * DAPO-17k-test-1000: pass@1 = 0.143
    * pass@32: {'zeros': 0.34, 'ones': 0.0, '0.0-0.1': 0.26, '0.1-0.2': 0.132, '0.2-0.3': 0.086, '0.3-0.4': 0.06, '0.4-0.5': 0.045, '0.5-0.6': 0.034, '0.6-0.7': 0.024, '0.7-0.8': 0.008, '0.8-0.9': 0.005, '0.9-1.0': 0.0}

#### Algorithm Design for RLOO
- For each RL step, we aim to have multiple prompts with N responses where there is at least one correct response and one incorrect response. When N is large this cost is huge. 
- So we use a smaller n and sample n responses for each prompt within a large batch. Ideally, we need n << N. If the pass rate of these n responseis noot 0 or 100, then we keep this prompt and sample n_continue = N-n responses. Otherwise, we discard this prompt.
- This is a very simple curriculum. Ideally, this specially works for two cases:
  * There is a huge dataset and doing RL on the whole dataset is infeasible based on the compute resource we have.
  * The distribution of the pass rate of all prompts is heavy-tailed. So most prompts have pass rates of 0 r 100 (or close to 0 or 100), and sampling n responses for these prompts is a waste.

#### Hyperparameters
```
curriculum.enable: whether we enable the curriculum.

data.gen_batch_size: how many prompts we try to generate n responses for each step (this is the generation batch size for the first generation stage).

data.train_batch_size: how many qualified prompts we need for each RL step. This is the batch size for the second generation stage. This is the number of actual  prompts  that we use to train the RL model in one RL step.

actor_rollout_ref.rollout.n: n.

actor_rollout_ref.rollout.n_continue: N-n.

max_num_gen_batches: the maximum number of generation batches we try to generate in the first stage.  If after this number of generation batches, we still don't have enough qualified prompts, we will raise an error.

```

#### Detailed Edits to the Codebase
- We add `data.use_chat_template` to denote whether we use the chat template for the base model. This needs to edit the `RLHFDataset.__getitem__` function and the truncation part.
- Since we do not use the DAPO's prompt, we need to change the reward function. This can be doneby editing the function in `verl/utils/reward_score/__init__.py`. We use the `math_verify.compute_score()` function to compute the score.
- Modify the `./recipe/dapo/dapo_ray_trainer.fit()` function. Pay attention to the `ProtoData.union()` function because it is used to merge the prompts and the responses. To write the main algorithm, you also need to add `DataProto.truncate()` and `DataProto.interleave_by_uid()` functions in `verl/protocol.py`. The interleave function is to make the response in both stages (for the same prompt) are in the contiguous space in the batch so that we can easily compute the advantage and do the RL training.
- If you are using an old version of tensordict (say, 0.1.2), you should remove the multi-modal part in the `verl/workers/actor/dp_actor.py` file.
