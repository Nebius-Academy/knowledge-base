---
layout: post
title: "LLM Customization"
categories: blog
permalink: /llm-customization/
---

The most powerful proprietary and open source LLMs are either generalist models or specialized for popular use cases, such as code generation. 

However, often you don't need a general conversationalist for your projects. Moreover, using a model with *too much* general capabilities may actually cause harm, since it might be more prone to hallucinations or out-of-use-case initiatives. 

Thus, we frequently need models that:

- Are proficient in a narrow field. For example a model that can consult a customer about your products; it doesn't need to be able to discuss theory of relativity for that.
- Don't hallucinate.
- Create output with the required tone of voice, formatting, and so on. In a sense, you want to narrow the stylistic range. So, no more epic poems as output — even if the user wishes so.

For that, we have three potential instruments:

- Prompt Engineering (role-assigning system prompt + few-shot examples)
- Fine tuning
- RAG

Let's discuss and compare their capabilities.

## RAG vs. Fine tuning vs. Prompt Engineering

Fine tuning and RAG generally have different purposes:

- **RAG excels at allowing an LLM to use additional knowledge**. Using RAG, we give the model a possibility to use application-specific data, up-to-date information etc.
    
    Setting up a RAG system is relatively cheap and fast. However, you need to process retrieved context with the LLM at each call, which can be costly.
    
    The LLMs itself remains at its generalist (or whatever) capability.
    
- **Fine tuning is great for imposing format and style restrictions**. This doesn't mean that we can't help the model to learn new information through fine tuning, but it may be less effective, and you definitely don't want to continuously fine tune an LLM as new data flows into your database.
    
    The good news is that fine tuning removes the need for additional retrieved context or over-complicated prompts, so this means no computational overhead at inference. 
    
    Additionally, the computational cost of fine tuning may be mitigated by **parameter-efficient fine tuning** (see the next chapter for more on this). That being said, collecting a dataset for fine tuning may be challenging for some applications.
    
    Note also that fine tuning may harm generalist capabilities. Now, usually this is actually not too bad because you don't need the LLM to be proficient outside the target topic anyway. But you do need to be aware of this **generality vs. specificity trade-off**.
    
- **Prompt engineering is a simple and flexible alternative to fine tuning**; it increases inference cost and partially relieves the need for a fine tuning dataset (still, you may need some few-shot examples), but can be easily customized for new scenarios.

## Fine tuning and knowledge old and new

AT the beginning of this week, we discussed the difference in usage between RAG and fine tuning. We mentioned that RAG is better and more flexible for adding new information into an LLM-based system, and now let’s return to that here and reaffirm and augment this point.

First of all, it has been shown in the paper [Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?](https://arxiv.org/pdf/2405.05904) that an LLM presented with new knowledge (not grounded in what is “knew” from pre-training) learns this new knowledge slower; while learning on such data actually encourages the LLM to hallucinate. 

At the same time, fine-tuning with something known to the LLM is correlated with better utilization of pre-existing knowledge.

From a practical point of view, the authors recommend the following. Let’s consider a multi-choice Q&A task. A data pair $(q, a)$ is:

- `HighlyKnown`, if the pre-fine-tuning LLM consistently predicts $a$ from $q$ with zero temperature and  few-shot (4-shot in the paper) prompt with questions semantically close to $q$
- `MaybeKnown`, if it sometimes predicts $a$
- `Unknown`, if it never predicts $a$ even with positive temperature

For fine tuning, you can throw away the `Unknown` examples from the fine-tuning dataset and expect only improvement from that, because it may increase the tendency for hallucinations. In the same time, having some `MaybeKnown` examples is vital for the success of fine-tuning.

![]({{ site.baseurl }}/assets/images/llm-customization/known-and-unknown.png){: .responsive-image style="--img-desktop:60%; --img-mobile:90%;"}

[Source](https://arxiv.org/pdf/2405.05904)

It worth adding that the threat of lobotomizing an LLM with full fine tuning is quite real, but sometimes you're ready to sacrifice generalist capabilities for a particular goal. (An example here is the [Gorilla](https://arxiv.org/pdf/2305.15334) model which was fine tuned for function calling and handles this reasonably well, at least for a model from 2023 — but it has largely lost all other capabilities.)

# Parameter-efficient Fine Tuning (PEFT)

Full fine tuning might be out of reach for most LLM users. But why is that?

Well, the main reason is that, during training, you have much higher compute requirements. This means that, if you take, for example, Adam, you need to store at minimum the gradients, as well as first and the second order statistics:

$$\color{blue}{g_t} = \nabla_\theta f_t(\theta_{t-1}),$$

$$\color{blue}{m_t} = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t,$$

$$\color{blue}{v_t} = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2,$$

$$\hat{m}_t = m_t / (1 - \beta_1^t),$$

$$\hat{v}t = v_t / (1 - \beta_2^t),$$

$$\theta_t = \theta{t-1} - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon),$$

And this is 3 times the required memory for storing the weights (and we haven't even started to discuss the implementation itself!). Given that we often struggle just to fit the LLM weights into a GPU, something 4 times this size will most likely be absolutely prohibitive.

Thus, we don't want to fine tune all the LLM parameters; instead, our goal is to optimize only a tiny fraction of the full amount of parameters. These strategies are referred to as **Parameter-efficient fine-tuning**.

In the next several subsections, we'll discuss in more detail two approaches for parameter-efficient fine-tuning: the historical **prompt tuning** and the widely-used **LoRA**.

## Prompt tuning

Imagine that you want your LLM to play the role of a grumpy and unhelpful AI assistant. Of course, you could do it with a system prompt, but you can also try to encode your intention into a special new synthetic token. Let’s give this a shot, and we’ll call this token `#System_prompt`. Thus, instead of feeding to the LLM the full `You are a grumpy and unhelpful AI assistant. \{User's request\}` , we want to only pass `\#System\_prompt. \{User's request\}`

Now, the new token doesn't have an embedding defined for it:

![]({{ site.baseurl }}/assets/images/llm-customization/prompt-tuning.png){: .responsive-image style="--img-desktop:90%; --img-mobile:100%;"}

And we’ll train this green vector during the fine tuning. **And the good thing about this is that you have no need for prompt engineering! Gradient descent will create an optimal vector for you.**

For fine-tuning, you need a dataset of strings `(user\_prompt, grumpy\_assistant's\_answer)`.

**Note**: ****There’s a huge difference between prompt engineering and prompt tuning.

Prompt tuning trains a single vector which usually doesn't correspond to any real token. Our resulting vector might be close to the embedding vectors of `“grumpy”` or `“unhelpful”`,  but generally this is not guaranteed; it can have no interpretation whatsoever.

# LoRA (Low-rank adapters)

The purpose of LoRA is as follows: we want to be able to fine tune every layer of the model, but at the same time we want to keep the number of optimized parameters low. This is achieved with **low-rank updates**. To understand the math of it, we recommend you to check the [Low Rank Fine-Tuning](https://medium.com/nebius/fundamentals-of-lora-and-low-rank-fine-tuning-e748f2f1255d) long read, and meanwhile, we'll only discuss the technical side here.

First of all, we’ll choose the layers to apply LoRA to. This can be any linear layers, that is, any layers where an input vector is multiplied by a matrix:
$x\mapsto Wx,\quad x\in\mathbb{R}^d, W\in\mathbb{R}^{m\times d},$
The latter means that $W$ is a matrix of size $m\times d$.

LoRA suggests not changing the $W$ but instead adding a trainable summand
$x\mapsto (W + {\color{orange}{BA}})x,$
where $A$ is a matrix of size $d\times r$ and $B$ is a matrix of size $m\times r$, where $r$ is typically small, say: $4$, $8$, $16$, or $64$.

**Example**. If $W$ is $4096\times4096$ which is quite a typical scale for intra-self-attention matrix multiplications, and $r = 16$,

- $W$ has $16,777,216$ parameters,
- $BA$ has $4096\times16 + 16\times4096 = 131,072$ parameters which is two orders of magnitude less.

So, even with the optimizer's memory overhead, the LoRA's additional memory requirements are dwarfed by the model storage size.

Let's illustrate this:

![]({{ site.baseurl }}/assets/images/llm-customization/lora-scheme.png){: .responsive-image style="--img-desktop:90%; --img-mobile:100%;"}

**Note**: **when** we train LoRA, we store both the initial $W$ and the two new matrices $A$ and $B$; this way, we can access the base model at any time.

**Here’s an optional comment that you may skip this until the “LLM Internals” module.** There are several places in a transformer-based LLM where you can encounter linear layers. The two main are:

- The self-attention mechanism, where linear transformations are used to produce queries, keys, values, and the final outputs.
- The feedforward block, which consists of several fully connected (linear) layers.

Usually, both groups are fine tuned, but it is generally advised that the attention block is fine tuned, because it is more critical for the final model quality.

## LoRA vs. full fine tuning

It’s generally agreed upon that LoRA works worse than full fine tuning.

The authors of this paper, “[Lora learns less and forgets less](https://arxiv.org/pdf/2405.09673)”, specifically compared LoRA will full fine tuning in context of learning specific knowledge. They reaffirmed that it indeed performs worse than full fine tuning, but they also noticed that LoRA preserves generalist capability better than full fine-tuning.

# Model distillation

A larger LLM is generally more capable than a smaller one. For instance, **Llama-3.1-405B** is likely to outperform **Llama-3.1-8b** in many tasks. However, power comes at a price: a massive model is slower, more compute-intensive, and significantly more expensive. This creates a strong incentive to **compress** large LLMs into smaller, more compute-efficient models while retaining some of the capabilities of the original.

Take **DeepSeek-R1**, for example. Though it's a strong reasoner, deploying efficiently it goes far beyond the capabilities of most AI-engineers (even if they have enough GPUs). However, it has smaller distilled versions - for example, **DeepSeek-R1-Distill-Llama-70B**. Let's discuss what is distillation.

**Knowledge distillation** is the process of transferring knowledge from a “**teacher**” model to a “**student**” model. The student model is typically smaller than the teacher, hence the term “distillation.”

A widely known distillation scheme was proposed by Geoffrey Hinton and his co-authors in their [paper](https://arxiv.org/pdf/1503.02531). Below is an illustration for the multiclass classification task:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/662b586e-86b7-4f44-9740-1dc06c7a67a4/f56ee372-d9d1-4712-b9d4-8f46fca8391a/image.png)

![]({{ site.baseurl }}/assets/images/llm-customization/hintons_distillation.png){: .responsive-image style="--img-desktop:90%; --img-mobile:100%;"}

The loss has two components:

- The first is the **cross-entropy** loss, which guides the student model toward solving the initial task correctly.
- The second is the [**Kullback-Leibler divergence**](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between the teacher’s and student’s predictions. This term guides the student to replicate the teacher’s predictions.

The coefficient $\lambda$ balances these two objectives. An interesting observation is that following the teacher’s predictions may, in some cases, be more beneficial for the student than predicting the initial targets. In a sense, the teacher may “filter out” noise and hard-to-predict peculiarities in the initial data.

**Note**: While Hinton’s distillation process is quite efficient, training a small LLM (such as an 8B model) from scratch still requires a substantial amount of data. In the next chapter, we’ll learn how NVIDIA has approached this challenge to alleviate the data demands.
