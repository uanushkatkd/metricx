# Steps to run code: 
## To Fine-tune Metricx-23 using LoRA: 
- Run `fine_tune.sh`, it'll take datasets path,ckpt save path, batch size, and grad accumulation steps. Other hyperparameters are set to default. 
- To change other hyperparameters, go through `fine_tune_from_scratch.py`
- Once final ckpt is saved then run `merge_adapter.py`. It takes `base_model`, `checkpoint`, and `output_dir` as arguments.
## To run inference :
- Run `score.sh`, it'll take datasets path,ckpt save path, batch size, and output_file as arguments.`output_file` is where the scores predicted will be saved, which can be used further to calculate correlations.
- `score.py` will calucate correlations, but code is not clean.
 


# MetricX-23

*This is not an officially supported Google product.*

This repository contains the code for running inference on MetricX-23 models,
a family of models for automatic evaluation of translations that were proposed
in the WMT'23 Metrics Shared Task submission
[MetricX-23: The Google Submission to the WMT 2023 Metrics Shared Task](https://aclanthology.org/2023.wmt-1.63/).
The models were trained in [T5X](https://github.com/google-research/t5x) and
then converted for use in PyTorch.

## Available Models
There are 6 models available on HuggingFace that vary in the number of
parameters and whether or not the model is reference-based or reference-free
(also known as quality estimation, or QE):

* [MetricX-23-XXL](https://huggingface.co/google/metricx-23-xxl-v2p0)
* [MetricX-23-XL](https://huggingface.co/google/metricx-23-xl-v2p0)
* [MetricX-23-Large](https://huggingface.co/google/metricx-23-large-v2p0)
* [MetricX-23-QE-XXL](https://huggingface.co/google/metricx-23-qe-xxl-v2p0)
* [MetricX-23-QE-XL](https://huggingface.co/google/metricx-23-qe-xl-v2p0)
* [MetricX-23-QE-Large](https://huggingface.co/google/metricx-23-qe-large-v2p0)

We recommend using the XXL model versions for the best agreement with human
judgments of translation quality, the Large versions for best speed, and the
XL for an intermediate use case.


