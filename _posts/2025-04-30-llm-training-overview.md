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

Before reading, please check the [Topic 1 notebook about tokenization](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic1/1.2_tokenization.ipynb).

# Pre-training

Despite their formidable capabilites, LLMs do a very simple thing: given a sequence of tokens, they **predict the next token**. By iterating this 

To train an LLM for the task of next token prediction, you only need a large selection of texts. Once you have that, you'll need to take each prefix (that is, the starting subsequences) within every text and train the LLM to predict the next token. And there's no need to label anything – just scrape the internet and be happy!

Well, there are a few catches to note:

- The collection of texts should be really, really huge. In fact, the amount of data that today's LLMs consume for training is something like the entirety of the internet itself. And there are signs that, soon enough, the available web data will no longer be enough to satiate the growing appetites of LLMs. For more, see the following picture taken from the paper [Will we run out of data?](https://arxiv.org/pdf/2211.04325).

![running-out-of-data.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/662b586e-86b7-4f44-9740-1dc06c7a67a4/09f89c11-43ed-47d0-9cb3-737d1bc3ebab/running-out-of-data.png)

- These things shouldn't really be used for training purposes, but allegedly, in some cases, they are anyway: copyrighted books, private chats from social networks, and many, many more. Some of them still leak somehow and contribute to the controversy around LLMs. See [this article](https://www.businessinsider.com/openais-latest-chatgpt-version-hides-training-on-copyrighted-material-2023-8?utm_source=reddit.com&r=US&IR=T) for an example. In any case, by and large, this means some portion of data is effectively excluded from training.
- You probably already know the principle of "Garbage in, garbage out". And indeed, training on low-quality data (noisy, irrelevant, and so on) may lead to poor quality results with the final model. Of course, with LLMs, the amount of training data needed is usually so huge that even raw internet dumps can contain enough useful information to produce a model of decent quality. With that being said, cleaner, high-quality data can allow LLMs to achieve more, and this is especially true for smaller LLMs. (We'll discuss this further in the section on Scaling Laws.)

## Tokenization

As we've already discussed, a text is split into **tokens** before being fed to an LLM. The splitting process is known as **tokenization**, and the tool that performs it is called a **tokenizer**. There are many tokenization strategies and the two simplest are **word-level tokenization** (each word is a token) and **character-level tokenization** (each character is a token).

Before we discuss the pros and cons of these two strategies, let's understand what we want out of an ideal tokenization; in short, we need the size of our vocabulary (that is, the list of all the individual tokens) to be balanced.

- The vocabulary shouldn't be too large. Indeed, with each token we need to store its embedding vector and too many embedding vectors will clog our GPU memory. Further, it's also not so good if the ratio of the LLM parameters concentrated in your embedding layer is too high. Ideally, a vocabulary should contain a prescribed number of tokens (like 10K or 50K).
- The items in the vocabulary should be meaningful, otherwise an LLM will struggle to make sense of them.

Incidentally, that second reason explains why character-level tokenization is not good. Moving on a bit, with word-level tokenization, the main problem is that you can't get an exhaustive list of all the words in the universe. Thus, during the inference stage, your LLM will always encounter new slang, new typos, and new languages which it hasn't seen during training.

So, we actually need something that is in between characters and words: **subword units**.

A very popular type of subword tokenization is **BPE** (**Byte pair encoding**). The idea is the following:

- We define the target vocabulary size {formula}N{/formula} (the number of tokens we want).
- We initialize the list of tokens with the characters that are present in the training corpus.
- We find the most frequent token pair in the training corpus and add it to the list of tokens (and we repeat this step) until we get {formula}N{/formula} tokens.

<aside>
📌

Here's an example:

Imagine our text corpus is `["low", "lower", "lowest"]` and the target vocabulary size is {formula}10{/formula}.

We start with `vocab = [#l, #o, #w, #e, #r, #s, #t]`. (We've added `#` symbols to differentiate the BPE tokens from the characters and words.) Then, we continue with the iterations:

1. The most frequent token pairs now are `lo = #l + # o` and `ow = # o + # w`, both a frequency of {formula}3{/formula}. Somehow, we need to choose one of them. I would say that `#lo` is earlier in the lexicographic order, so we'll take that one:
    
    `vocab = [#l, #o, #w, #e, #r, #s, #t, #lo]`
    
    Note that `low` is now tokenized as `#lo + #w`.
    
2. The most frequent token pair is now `low = #lo + #w` with a frequency of {formula}3{/formula}. Note that there is no longer `ow = #o + #w` because `low` no longer has the `#o` token.
    
    `vocab = [#l, #o, #w, #e, #r, #s, #t, #lo, #low]`
    
3. Now, the most frequent token pair is `lowe = #low + #e`, with a frequency of {formula}2{/formula}. Let's add it:
    
    `vocab = [#l, #o, #w, #e, #r, #s, #t, #lo, #low, #lowe]`
    
    Since we've reached the vocabulary size {formula}10{/formula}, we can stop now.
    

The final tokenization of our corpus is as follows:

`low = #low`

`lower = #lowe #r`

`lowest = #lowe #s #t`

</aside>

**Note 1**: The original BPE algorithm only allowed token merging inside words, but later approaches can sometimes allow this: For example, an LLM can have `import pandas as pd` as one token.

**Note 2**: BPE is frequency-based, so words and subwords originating from languages that are not as widespread as English may be mercilessly cut into pieces, potentially damaging the LLM's multilingual proficiency. With this in mind, some LLM creators spend additional effort to ensure a more equal representation of different languages and writing systems in the vocabulary.

**Note 3**: Even listing all the characters to start BPE can be tough. There are many writing systems in the world, and moreover, there are emoji and other unicode characters to contend with. But still, being able to process emoji could be cool.

To cope with this, LLM creators now often use **byte-level** tokenization on raw unicode bits. This sometimes produces non-interpretable tokens but allows us not to think about the diversity of writing systems.

(**Practical note**: Every LLM has its own tokenizer, and it's a bad idea to try one LLM's tokenizer with another's. We'll learn to load the right tokenizers in Week 3.)

## Measuring dataset length in tokens

When you read papers about LLMs, you often encounter things like "We've trained the model on 1T (one trillion) tokens". Let's figure out what this means.

First of all, training dataset sizes are indeed measured in tokens. So, the passage above says that the total number of tokens in all the texts in the dataset is around one trillion ({formula}10^{12}{/formula}). To understand how huge this number is, imagine that the "Lord of the Rings" has 500-700K tokens (that is, thousands of them) depending on the tokenizer used. So, 1T tokens is like 2 million "Lord of the Rings"! And yes, 1T tokens represents a scale approximately in line with the number a modern LLM may consume during pre-training.

Now, you might ask: how many epochs do they train an LLM on a dataset of this size? You'll probably be surprised, but in most cases, the training only goes on for just one epoch. Partially, this is because of the dataset size and the amount of GPU hours that would be needed for even one pass through 1T tokens. Beyond that though, it also just turns out to be sufficient. That said, there is research that shows that multi-epoch training is also beneficial for LLMs and that training for four epochs may give the same result as training twice on unique data. This is good because, as we saw earlier, we may soon be running low on new data. We'll discuss this in more detail when we talk about scaling laws.
