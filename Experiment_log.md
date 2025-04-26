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
    * DAPO-17k-test-1000: pass@1 =
    * pass@32: 

#### Algorithm Design


#### Detailed Edit to the Codebase
