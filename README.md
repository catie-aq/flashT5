# FAT5 - A fast implementation of T5/UL2 with Flash Attention

❗  Ecrire une petite introduction qui liste tout ce que fait ce dépôt.</span>

**⚠ WARNING: This repository is not yet complete. Please refer to the roadmap part of this README. ⚠**


## General ideas that guided us

We worked on two main points - **optimizing memory bandwidth**, and **optimizing the use of Tensor Cores**:
- <ins>Optimizing the memory bandwidth optimization</ins>
  - We used the [Flash Attention (v2)](https://arxiv.org/abs/2307.08691) by Dao (2023).
  This key technique consists in developing a CUDA kernel that can merge several limiting operations into one.
  This can limit the need to copy large arrays into the GPU's global memory and then immediately reload them. This is now a common feature of transformer-decoder models.  
  **In our view, it's very important to maintain an encoder-decoder architecture, in order to compress the information to its *substantifique moelle* before extending it again if required**.  
  **Beyond NLP, we believe that such architectures are essential for audio, time series, or more generally in multimodality, where using a decoder seems sub-optimal to us compared to a sequence-by-sequence approach.**  
  We therefore chose to work with a [T5](http://jmlr.org/papers/v21/20-074.html) by Raffel et al. (2020) and in practice with the [nanoT5](https://arxiv.org/abs/2309.02373) version by Nawrot (2023).
  For pretext tasks during pre-training, we followed [UL2](https://arxiv.org/abs/2205.05131v3) by Tay et Dehghani (2022) with the following 7 tasks:
    ```py
    denoiser_list=[
    {"mu": 3.0, "r": 0.15, "max_spans": max_token_length, "prefix": "[R]"},
    {"mu": 8.0, "r": 0.15, "max_spans": max_token_length, "prefix": "[R]"},
    {"mu": 4.0, "r": 0.0, "max_spans": 1, "prefix": "[S]"},
    {"mu": 3.0, "r": 0.5, "max_spans": max_token_length, "prefix": "[X]"},
    {"mu": 8.0, "r": 0.15, "max_spans": max_token_length, "prefix": "[X]"},
    {"mu": 64.0, "r": 0.15, "max_spans": max_token_length, "prefix": "[X]"},
    {"mu": 64.0, "r": 0.5, "max_spans": max_token_length, "prefix": "[X]"}],
    denoiser_proportions=[0.165, 0.165, 0.34, 0.0825, 0.0825, 0.0825, 0.0825]
    ```
    with `mu`: the span size, `r`: the % of masking in the span and `prefix`: the type of the pretext task (the meaning of the letters `[R]`, `[S]` and `[X]` is described [here](https://huggingface.co/google/ul2#mixture-of-denoisers)).  
    As Flash Attention doesn't manage the (additive) attentional biases of the T5 model, we extended it to do so.
  
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="./assets/FAT5_dark.gif">
      <img alt="FAT5 animation" src="./assets/FAT5.gif">
    </picture>
  
    The code for this part is available in this [GitHub repository](https://github.com/catie-aq/flashT5/tree/main/src/utils/fa2_lib) or in the [following PR](https://github.com/Dao-AILab/flash-attention/pull/617) on the official GitHub repository of the Flash Attention.
    Note that we use absolute positional encoding.
  - A simpler approach is to compile the models with `torch.compile`.
  PyTorch then takes care of any possible merging, possibly by reordering operations.
  See [documentation](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) for more details.
  The aim is to eliminate "graph breaks", which are returns to an "eager" execution mode and have a negative impact on the operation's performance.
  This means rewriting operations to avoid such breaks in the compilation graph, as well as avoiding (for the time being) dynamic tensor sizes that are poorly supported.  
  These two methods are not easily compatible with PyTorch 2.1 (PyTorch 2.2 should make things simpler).
  We based ourselves on this interesting [document](https://docs.google.com/document/d/1W--T6wz8IY8fOI0Vm8BF44PdBgs283QvpelJZWieQWQ/edit) to achieve this.
  For more information, you can directly look at the [code](https://github.com/catie-aq/flash-NLP/blob/main/modeling/utils/fa2_lib/fa2_lib.py).

- <ins>Optimizing the use of Tensor Core</ins>
  - The first optimization is to use tensor sizes with certain multiples (of 8, 16, 32, 64, 128 in general).
  We invite the reader to refer to Nvidia's documentation, in particular this [article](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc) and this [one](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/).  
  In this logic, we trained a custom tokenizer of vocabulary size of 32,768 (2**15, following [this observation by Karpathy](https://twitter.com/karpathy/status/1621578354024677377)) trained on the first 1,000,000 rows of the French part of [OSCAR-2301](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301).
  - The second optimization consists of training the models in `bf16` or `fp16`.  
  Recent GPUs make full use of reduced precision (with a 2x throughput factor compared to `fp32`).
  `bf16` is only available on Ampere or higher architectures, but eliminates the need for [loss scaling](https://arxiv.org/abs/1710.03740) (Micikevicius et al. (2017)), which is generally required with fp16, thanks to its greater dynamic range (the exponent is coded on 8 bits, as with `fp32`).
  In this logic, we train our models in `bf16`.
  - The third optimization consists in limiting unnecessary calculations.  
  When working with sequences, there is a natural tendency to pad a set of sequences in order to build batches. 
  Padding tokens then generate unnecessary calculations. 
  The first thing to do is to limit padding to the maximum sequence size and not to a maximum value. 
  This is the [dynamic padding](https://huggingface.co/learn/nlp-course/chapter3/2?fw=pt#dynamic-padding) technique.
  Then, padding tokens are generally left over with the dynamic padding technique. For the rest, you have two choices:
    - either use a method of grouping data with similar sizes (for example, [this parameter](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.group_by_length) in Hugging Face or [retrieving the sampler from Hugging Face](https://discuss.huggingface.co/t/how-to-implement-trainers-group-by-length-in-pytorch/9232) for PyTorch)
    - or use the remaining tokens to concatenate different examples with a custom DataCollator.  
  We've opted for the second option. See, for example our DataCollator [a mixture of denoisers](https://github.com/catie-aq/flashT5/blob/main/src/data/data_collator_ul2.py) (UL2).
  - The fourth optimization is to use the right optimizer.  
    Changing the optimizer from the initial implementation of the model can be judicious to accelerate model convergence (but deviates from the basic implementation and can therefore prevent the results of an initial paper from being reproduced).
    Optimizers accelerate convergence by allowing large batch sizes, as in the case of [LAMB](https://arxiv.org/abs/1904.00962) by You et al. (2019), or by allowing the use of higher learning rates, as in [Sophia](https://arxiv.org/abs/2305.14342) by Liu et al. (2023).
    More efficient versions of the optimizers can also be used.  
    See the `fused` option in PyTorch's [Adam optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) or the optimizers available in [Apex](https://github.com/NVIDIA/apex).  
    We've used [AdamWScale](https://github.com/catie-aq/flashT5/blob/main/src/utils/adamw_scaled.py) optimizer (`lr = 2e-2`, `betas = (0.9, 0.999)`, `eps = 1e-6`, `weight_decay = 0.0`) for a first test. We look forward to testing others.
  - The final optimization consists in using less GPU memory.  
  Some techniques exist to limit the use of GPU memory by the model, such as [gradient checkpointing](https://pytorch.org/docs/stable/checkpoint.html) or ZeRO-type methods implemented in [DeepSpeed](https://github.com/microsoft/DeepSpeed). By limiting the amount of memory used, we can use larger batch sizes and thus speed up the model.


## Application to French
We've used the codes of this repository to train two FAT5-UL2 in French.
You can find the weights of the [base version](https://huggingface.co/CATIE-AQ/FAT5-base-UL2-fr) (305M parameters) and the [large version](https://huggingface.co/CATIE-AQ/FAT5-large-UL2-fr) (973M parameters) on [Hugging Face](https://huggingface.co/collections/CATIE-AQ/catie-french-fat5-65c0b4c12bc7789b319d8f72).  
Models are pre-trained on the French part of the [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) corpus by Nguyen et al. (2023), i.e. 1,258 GB of text.  
The base models were run on a single A 100; for 11 days for the base version and 30 days for the large version.  

❗ AJOUTER LES PLOTS DES LOSS  

❗ The models were then finetuned on BLABLA tasks.  
A TERMINER  

## Aplication to your own language

❗ Faire un petit descriptif des lignes de codes à lancer pour entraîner un modèle sur une autre langue que le français.

## Roadmap
We're not researchers or engineers working full-time on research and development projects. 
As CATIE is a non-profit association, we have to work on other projects in order to bring in money and pay our salaries.  
So think of this repository as a side project on which we'd like to make improvements, but for which we can't provide assiduous maintenance or development because of a lack of sanctuary hours. 

❗ LISTER TOUT CE QUI SERA FAIT DE MANIERE CERTAINE + TOUT CE QUI SERAIT A FAIRE MAIS SANS GARANTIES QUE CELA SOIT FAIT FAUTE DE TEMPS

- Positional Enconding
Currently in this GitHub repository we provide the absolute positional encoding.
We had also implemented the [ALIBI](https://arxiv.org/abs/2108.12409) one by Press et al. (2021) but it's now better to use the implementation since [added to the GitHub repository of the Flash Attention](https://github.com/Dao-AILab/flash-attention/pull/540).
This is not a priority compared to the other tests we would like to perform described above, but if we have some time, we would like to test the [ROPE](https://arxiv.org/abs/2104.09864) encoding of Su et al. (2023)

## License
❗ A AJOUTER

## Citation
❗ A AJOUTER
