---
layout: post
title: "LLM training overview"
categories: blog
permalink: /llm-training-overview/
---

## The Evolution of LLM Training: From Pre-Training to Alignment

In this long read, we'll explore the fundamental stages of Large Language Model (LLM) training that have been established since the early days of ChatGPT and refined over time:

1. **Pre-training** - the foundational stage where an LLM ingests huge amounts of texts, code etc and develops general language capabilities

2. **Instruction tuning** - the refinement stage that teaches the model to understand and follow complex instructions

3. **Alignment training** - the optimization stage focused on making the model helpful and harmless by implementing safeguards and ensuring that the model is an agreeable conversationalist

This discussion will be continued in the subsequent 

* [**Establishing long reasoning capabilities**](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic2/r.3_establishing_non_linear_reasoning.ipynb), the colab notebook telling the story of the emergence of DeepSeek R1 and other long reasoning LLMs
* The math of RHLF, DPO, and Reasoning training (will be available later)
* Multimodal LLMs architectures and training (will be available later)

In this long read, we'll leave the questions of LLM architecture under the hood, but we'll return to them in Topic 3.

Before reading, please check the [Topic 1 notebook about tokenization](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic1/1.2_tokenization.ipynb).

# Pre-training

Despite their formidable capabilites, LLMs do a very simple thing: given a sequence of tokens, they **predict the next token**. By iterating this procedure, the model completes the prompt, generating new tokens until the special `<EOS>` (End Of Sequence) token is produced or `max_tokens` is reached.

![]({{ site.baseurl }}/assets/images/llm-training-overview/LM-working-simple.gif){: .responsive-image style="--img-desktop:75%; --img-mobile:90%;"}

To train an LLM for next token prediction, you start with a large corpus of texts. The process works as follows.

For each text in the training data, the LLM processes the input by

* First converting each token into a vector **embedding**.
* These embeddings are then transformed through multiple neural network layers, ultimately producing **final vector representations** for each token position.
* From these final representations, we apply the **LM head** (Language Modeling Head), also known as the **unembedding layer**. It projects the vectors back into vocabulary space, making up **logits** $l_w$ for each token $w$ from the vocabulary.
* The **softmax function** is then applied to convert these logits into **next token probabilities** $\widehat{p}(w)$ for every token $w$ in the vocabulary.

For example, when processing the phrase `"London is famous for"`, the model produces 

* the predicted probabilities $\widehat{p}(w\vert \text{<BOS>})$ for all tokens as potential phrase starters
* the predicted probabilities $\widehat{p}(w\vert \text{``London''})$ for all tokens as potential continuations of `"London"`
* the predicted probabilities $\widehat{p}(w\vert \text{``London is''})$ for all tokens as potential continuations of `"London is"`
* etc,

![]({{ site.baseurl }}/assets/images/llm-training-overview/llm-pretraining-produce1.png){: .responsive-image style="--img-desktop:90%; --img-mobile:90%;"}

We'll discuss the LM head and the softmax function in the [LLM Inference Parameters notebook](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic1/1.6_llm_inference_parameters.ipynb).

The goal during training is to ensure that the right tokens will get the maximal probability. In the example below, we want

* $\widehat{p}(\text{``Luke''}|\text{<BOS>})$ be the maximal among all $\widehat{p}(w|\text{<BOS>})$,
* $\widehat{p}(\text{``,''}|\text{``Luke''})$ be the maximal among all $\widehat{p}(w|\text{``Luke''})$,
* $\widehat{p}(\text{``I''}|\text{``Luke,''})$ be the maximal among all $\widehat{p}(w|\text{``Luke,''})$,
* etc.

![]({{ site.baseurl }}/assets/images/llm-training-overview/LM-training-simple.png){: .responsive-image style="--img-desktop:90%; --img-mobile:90%;"}

For a token sequence $v_{1:L} = v_1v_2\ldots v_L$ from the training data, we may formulate this as an optimization problem:

$$
\begin{aligned}
\widehat{p}(v_2|v_1)&\to\max\\
\widehat{p}(v_3|v_{1:2})&\to\max\\
\widehat{p}(v_4|v_{1:3})&\to\max\\
&\ldots
\end{aligned}
$$

Now, we'll apply logarithm, which is monotonic, that is for $x > 0$, $x\to\max$ is the same as $\log{x}\to\max$. Luckily, the predicted probabilities are non-negative, and they are unlikely to be exactly zero thanks to floating-point computation issues. We thus get

$$
\begin{aligned}
\log\widehat{p}(v_2|v_1)&\to\max\\
\log\widehat{p}(v_3|v_{1:2})&\to\max\\
\log\widehat{p}(v_4|v_{1:3})&\to\max\\
&\ldots
\end{aligned}
$$

There are several potential reasons to this move; right now we'll state that it is beneficial for the optimization process. Indeed, logarithm "punishes" $\widehat{p}(w\vert p_{1:k})$ much more severely for being small (close to zero).

![]({{ site.baseurl }}/assets/images/llm-training-overview/log-vs-linear.png){: .responsive-image style="--img-desktop:25%; --img-mobile:50%;"}

Next, loss functions in Machine Learning are usually minimized, not maximized. So, we'll change the signs:

$$
\begin{aligned}
-\log\widehat{p}(v_2|v_1)&\to\min\\
-\log\widehat{p}(v_3|v_{1:2})&\to\min\\
-\log\widehat{p}(v_4|v_{1:3})&\to\min\\
&\ldots
\end{aligned}
$$

This is usually formulated in a more mathematically fancy way. Let's introduce the distribution 
$$p_{\mathrm{true}}(w|v_{1:k}) = \begin{cases}
1,\text{ if }w=v_{k+1},\\
0,\text{ otherwise}
\end{cases}$$

Then

$$\widehat{p}(v_{k+1}|v_{1:k}) = \sum_{w}p_{\mathrm{true}}(w|v_{1:k})\cdot \log\widehat{p}(w|v_{1:k})$$

Indeed, in the right hand side only one summand in nonzero - the one with $w = v_{k+1}$.

Finally, let's get all the things we want to minimize into one large sum - first along the sequence $v_{1:L}$, and then across all the sequences from the training dataset (or, rather, the current training batch). This way we'll get the final pre-training **loss function**:

$$\mathcal{L} = \sum_v\sum_{k=1}^{L-1}\sum_{w}p_{\mathrm{true}}(w|v_{1:k})\cdot \log\widehat{p}(w|v_{1:k})$$

Now you know the pre-training basics! Still, there are a few catches to note:

- The collection of texts should be really, really huge. In fact, the amount of data that today's LLMs consume for training is something like the entirety of the internet itself. And there are signs that, soon enough, the available web data will no longer be enough to satiate the growing appetites of LLMs. For more, see the following picture taken from the paper [Will we run out of data?](https://arxiv.org/pdf/2211.04325).

![]({{ site.baseurl }}/assets/images/llm-training-overview/running-out-of-data.png){: .responsive-image style="--img-desktop:90%; --img-mobile:90%;"}

- You probably already know the principle of "Garbage in, garbage out". For LLMs, it may probably be reformulated as "Treasure in, treasure out". And indeed, the best LLMs are known to be trained on high-quality, carefully curated data.

- The previous consideration is especially important, because **most of the LLM capabilities are established at the pre-training stage**, due to the amount of data used at this stage. (We'll show the comparison later in this long read.) Some of these capabilities might remain dormant, like [in the case of long reasoning](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic2/r.3_establishing_non_linear_reasoning.ipynb), but in any case on the later training stages they are rather avakened then established anew.

- Many of today's LLMs can work with large context lengths, and this is also established at the pre-training phase. But LLMs are not trained on 100k-long sequences from the very beginning, because that would be too taxing from the computations perspective. (Transformers' time and memory consumption grow as square of the input sequence length.) In most cases, **progressive length training** is used.

  At first, the LLM might be trained on up to 8k-token-long sequences, and this would be the most intensive training part, using more than half training tokens. After that, the LLM is trained for several more stages on gradually longer sequences, e.g. 16k $\to$ 32k $\to$ 128k.
