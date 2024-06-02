---
license: apache-2.0
datasets:
- c4
language:
- en
inference: false
---

# MosaicBERT-Base model

MosaicBERT-Base is a custom BERT architecture and training recipe optimized for fast pretraining. 
MosaicBERT trains faster and achieves higher pretraining and finetuning accuracy when benchmarked against 
Hugging Face's [bert-base-uncased](https://huggingface.co/bert-base-uncased).

This study motivated many of the architecture choices around MosaicML's [MPT-7B](https://huggingface.co/mosaicml/mpt-7b) and [MPT-30B](https://huggingface.co/mosaicml/mpt-30b) models.

## Model Date

March 2023

## Documentation

* [Project Page (mosaicbert.github.io)](mosaicbert.github.io)
* [Github (mosaicml/examples/tree/main/examples/benchmarks/bert)](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert)
* [Paper (NeurIPS 2023)](https://openreview.net/forum?id=5zipcfLC2Z)
* Colab Tutorials:
  * [MosaicBERT Tutorial Part 1: Load Pretrained Weights and Experiment with Sequence Length Extrapolation Using ALiBi](https://colab.research.google.com/drive/1r0A3QEbu4Nzs2Jl6LaiNoW5EumIVqrGc?usp=sharing)
* [Blog Post (March 2023)](https://www.mosaicml.com/blog/mosaicbert)

## Community Adoption

* [DNABERT-2-117M](https://huggingface.co/zhihan1996/DNABERT-2-117M) for genome classification

## How to use

```python
import torch
import transformers
from transformers import AutoModelForMaskedLM, BertTokenizer, pipeline
from transformers import BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # MosaicBERT uses the standard BERT tokenizer

config = transformers.BertConfig.from_pretrained('mosaicml/mosaic-bert-base') # the config needs to be passed in
mosaicbert = AutoModelForMaskedLM.from_pretrained('mosaicml/mosaic-bert-base',config=config,trust_remote_code=True)

# To use this model directly for masked language modeling
mosaicbert_classifier = pipeline('fill-mask', model=mosaicbert, tokenizer=tokenizer,device="cpu")
mosaicbert_classifier("I [MASK] to the store yesterday.")
```

Note that the tokenizer for this model is simply the Hugging Face `bert-base-uncased` tokenizer.

In order to take advantage of ALiBi by extrapolating to longer sequence lengths, simply change the `alibi_starting_size` flag in the 
config file and reload the model. 

```python
config = transformers.BertConfig.from_pretrained('mosaicml/mosaic-bert-base')
config.alibi_starting_size = 1024 # maximum sequence length updated to 1024 from config default of 512

mosaicbert = AutoModelForMaskedLM.from_pretrained('mosaicml/mosaic-bert-base',config=config,trust_remote_code=True)
```

This simply presets the non-learned linear bias matrix in every attention block to 1024 tokens (note that this particular model was trained with a sequence length of 128 tokens).

**To continue MLM pretraining**, follow the [MLM pre-training section of the mosaicml/examples/benchmarks/bert repo](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert#pre-training).

**To fine-tune this model for classification**, follow the [Single-task fine-tuning section of the mosaicml/examples/benchmarks/bert repo](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert#fine-tuning).

### [Update 1/2/2024] Triton Flash Attention with ALiBi

Note that by default, triton Flash Attention is **not** enabled or required. In order to enable our custom implementation of triton Flash Attention with ALiBi from March 2023,
set `attention_probs_dropout_prob: 0.0`. We are currently working on supporting Flash Attention 2 (see [PR here](https://github.com/mosaicml/examples/pull/440)) and replacing the custom triton implementation.


### Remote Code

This model requires that `trust_remote_code=True` be passed to the `from_pretrained` method. This is because we train using [FlashAttention (Dao et al. 2022)](https://arxiv.org/pdf/2205.14135.pdf), which is not part of the `transformers` library and depends on [Triton](https://github.com/openai/triton) and some custom PyTorch code. Since this involves executing arbitrary code, you should consider passing a git `revision` argument that specifies the exact commit of the code, for example:

```python
mosaicbert = AutoModelForMaskedLM.from_pretrained(
   'mosaicml/mosaic-bert-base',
   trust_remote_code=True,
   revision='24512df',
)
```

However, if there are updates to this model or code and you specify a revision, you will need to manually check for them and update the commit hash accordingly.

## Model description

In order to build MosaicBERT, we adopted architectural choices from the recent transformer literature. 
These include [FlashAttention (Dao et al. 2022)](https://arxiv.org/pdf/2205.14135.pdf), [ALiBi (Press et al. 2021)](https://arxiv.org/abs/2108.12409),
and [Gated Linear Units (Shazeer 2020)](https://arxiv.org/abs/2002.05202). In addition, we remove padding inside the transformer block, 
and apply LayerNorm with low precision.

### Modifications to the Attention Mechanism
1. **FlashAttention**: Attention layers are core components of the transformer architecture. The recently proposed FlashAttention layer
reduces the number of read/write operations between the GPU HBM (high bandwidth memory, i.e. long-term memory) and the GPU SRAM
(i.e. short-term memory) [[Dao et al. 2022]](https://arxiv.org/pdf/2205.14135.pdf). We used the FlashAttention module built by
[hazy research](https://github.com/HazyResearch/flash-attention) with [OpenAI’s triton library](https://github.com/openai/triton).

2. **Attention with Linear Biases (ALiBi)**:  In most BERT models, the positions of tokens in a sequence are encoded with a position embedding layer;
   this embedding allows subsequent layers to keep track of the order of tokens in a sequence. ALiBi eliminates position embeddings and
   instead conveys this information using a bias matrix in the attention operation. It modifies the attention mechanism such that nearby
   tokens strongly attend to one another [[Press et al. 2021]](https://arxiv.org/abs/2108.12409). In addition to improving the performance of the final model, ALiBi helps the
   model to handle sequences longer than it saw during training. Details on our ALiBi implementation can be found [in the mosaicml/examples repo here](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert/src/bert_layers.py#L425).

3. **Unpadding**: Standard NLP practice is to combine text sequences of different lengths into a batch, and pad the sequences with empty
   tokens so that all sequence lengths are the same. During training, however, this can lead to many superfluous operations on those
   padding tokens. In MosaicBERT, we take a different approach: we concatenate all the examples in a minibatch into a single sequence
   of batch size 1. Results from NVIDIA and others have shown that this approach  leads to speed improvements during training, since
   operations are not performed on padding tokens (see for example [Zeng et al. 2022](https://arxiv.org/pdf/2208.08124.pdf)).
   Details on our “unpadding” implementation can be found [in the mosaicml/examples repo here](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert/src/bert_padding.py).

4. **Low Precision LayerNorm**: this small tweak forces LayerNorm modules to run in float16 or bfloat16 precision instead of float32, improving utilization.
   Our implementation can be found [in the mosaicml/examples repo here](https://docs.mosaicml.com/projects/composer/en/stable/method_cards/low_precision_layernorm.html).

### Modifications to the Feedforward Layers

5. **Gated Linear Units (GLU)**: We used Gated Linear Units for the feedforward sublayer of a transformer. GLUs were first proposed in 2016 [[Dauphin et al. 2016]](https://arxiv.org/abs/1612.08083),
   and incorporate an extra learnable matrix that “gates” the outputs of the feedforward layer. More recent work has shown that
   GLUs can improve performance quality in transformers [[Shazeer, 2020](https://arxiv.org/abs/2002.05202), [Narang et al. 2021](https://arxiv.org/pdf/2102.11972.pdf)]. We used the GeLU (Gaussian-error Linear Unit)
   activation function with GLU, which is sometimes referred to as GeGLU. The GeLU activation function is a smooth, fully differentiable
   approximation to ReLU; we found that this led to a nominal improvement over ReLU. More details on our implementation of GLU can be found here.
   The extra gating matrix in a GLU model potentially adds additional parameters to a model; we chose to augment our BERT-Base model with
   additional parameters due to GLU modules as it leads to a Pareto improvement across all timescales (which is not true of all larger
   models such as BERT-Large). While BERT-Base has 110 million parameters, MosaicBERT-Base has 137 million parameters. Note that
   MosaicBERT-Base trains faster than BERT-Base despite having more parameters. 




## Training data

MosaicBERT is pretrained using a standard Masked Language Modeling (MLM) objective: the model is given a sequence of 
text with some tokens hidden, and it has to predict these masked tokens. MosaicBERT is trained on 
the English [“Colossal, Cleaned, Common Crawl” C4 dataset](https://github.com/allenai/allennlp/discussions/5056), which contains roughly 365 million curated text documents scraped 
from the internet (equivalent to 156 billion tokens).  We used this more modern dataset in place of traditional BERT pretraining 
corpora like English Wikipedia and BooksCorpus.

## Pretraining Optimizations

Many of these pretraining optimizations below were informed by our [BERT results for the MLPerf v2.1 speed benchmark](https://www.mosaicml.com/blog/mlperf-nlp-nov2022).

1. **MosaicML Streaming Dataset**: As part of our efficiency pipeline, we converted the C4 dataset to [MosaicML’s StreamingDataset format](https://www.mosaicml.com/blog/mosaicml-streamingdataset) and used this
for both MosaicBERT-Base and the baseline BERT-Base. For all BERT-Base models, we chose the training duration to be 286,720,000 samples of sequence length 128; this covers 78.6% of C4. 


2. **Higher Masking Ratio for the Masked Language Modeling Objective**: We used the standard Masked Language Modeling (MLM) pretraining objective. 
While the original BERT paper also included a Next Sentence Prediction (NSP) task in the pretraining objective, 
subsequent papers have shown this to be unnecessary [Liu et al. 2019](https://arxiv.org/abs/1907.11692).
However, we found that a 30% masking ratio led to slight accuracy improvements in both pretraining MLM and downstream GLUE performance. 
We therefore included this simple change as part of our MosaicBERT training recipe. Recent studies have also found that this simple 
change can lead to downstream improvements [Wettig et al. 2022](https://arxiv.org/abs/2202.08005).

3. **Bfloat16 Precision**: We use [bf16 (bfloat16) mixed precision training](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) for all the models, where a matrix multiplication layer uses bf16
for the multiplication and 32-bit IEEE floating point for gradient accumulation. We found this to be more stable than using float16 mixed precision.

4. **Vocab Size as a Multiple of 64**: We increased the vocab size to be a multiple of 8 as well as 64 (i.e. from 30,522 to 30,528).
This small constraint is something of [a magic trick among ML practitioners](https://twitter.com/karpathy/status/1621578354024677377), and leads to a throughput speedup.

5. **Hyperparameters**: For all models, we use Decoupled AdamW with Beta_1=0.9 and Beta_2=0.98, and a weight decay value of 1.0e-5.
The learning rate schedule begins with a warmup to a maximum learning rate of 5.0e-4 followed by a linear decay to zero.
Warmup lasted for 6% of the full training duration. Global batch size was set to 4096, and microbatch size was 128; since global batch size was 4096, full pretraining consisted of 70,000 batches.
We set the maximum sequence length during pretraining to 128, and we used the standard embedding dimension of 768.
For MosaicBERT, we applied 0.1 dropout to the feedforward layers but no dropout to the FlashAttention module, as this was not possible with the OpenAI triton implementation.
Full configuration details for pretraining MosaicBERT-Base can be found in the configuration yamls [in the mosaicml/examples repo here](https://github.com/mosaicml/examples/blob/main/examples/benchmarks/bert/yamls/main/mosaic-bert-base-uncased.yaml).


## Evaluation results

When fine-tuned on downstream tasks (following the [finetuning details here](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert/yamls/finetuning/glue/mosaic-bert-base-uncased.yaml)), the MosaicBERT model achieves the following GLUE results:

| Task | MNLI-(m/mm) | QQP  | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE  | Average |
|:----:|:-----------:|:----:|:----:|:-----:|:----:|:-----:|:----:|:----:|:-------:|
|      | 0.8495 | 0.9029 | 0.9074|  0.9246 | 0.5511 | 0.8927  | 0.9003 | 0.8136 |  0.8428   |

Note that this is averaged over n=5 pretraining seeds.

## Collection of MosaicBERT-Base models trained using ALiBi on different sequence lengths

ALiBi allows a model trained with a sequence length n to easily extrapolate to sequence lengths >2n during finetuning. For more details, see [Train Short, Test Long: Attention with Linear
Biases Enables Input Length Extrapolation (Press et al. 2022)](https://arxiv.org/abs/2108.12409)

This model is part of the **family of MosaicBERT-Base models** trained using ALiBi on different sequence lengths:

* mosaic-bert-base (trained on a sequence length of 128 tokens)
* [mosaic-bert-base-seqlen-256](https://huggingface.co/mosaicml/mosaic-bert-base-seqlen-256)
* [mosaic-bert-base-seqlen-512](https://huggingface.co/mosaicml/mosaic-bert-base-seqlen-512)
* [mosaic-bert-base-seqlen-1024](https://huggingface.co/mosaicml/mosaic-bert-base-seqlen-1024) 
* [mosaic-bert-base-seqlen-2048](https://huggingface.co/mosaicml/mosaic-bert-base-seqlen-2048)

The primary use case of these models is for research on efficient pretraining and finetuning for long context embeddings.

## Intended uses & limitations

This model is intended to be finetuned on downstream tasks.

## Citation

Please cite this model using the following format:

```
@article{portes2023MosaicBERT,
  title={MosaicBERT: A Bidirectional Encoder Optimized for Fast Pretraining},
  author={Jacob Portes, Alexander R Trott, Sam Havens, Daniel King, Abhinav Venigalla,
  Moin Nadeem, Nikhil Sardana, Daya Khudia, Jonathan Frankle},
  journal={NeuRIPS https://openreview.net/pdf?id=5zipcfLC2Z},
  year={2023},
}
```