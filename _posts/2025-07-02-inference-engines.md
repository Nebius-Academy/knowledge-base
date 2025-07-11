---
layout: post
title: "Inference Engines"
categories: blog
permalink: /inference-engines/
---

**By: [Alexey Bukhtiyarov](https://www.linkedin.com/in/leshanbog/)**

# Introduction

Running LLMs in a Jupyter Notebook using frameworks like [Transformers](https://huggingface.co/docs/transformers/en/index) can be a great way to experiment. However, deploying these models for real-world user interactions presents a different set of challenges. For this, you'll need an **Inference Engine**.

LLMs are typically accessed via an API: you provide input context and generation parameters, and you receive generated text in response. And this entire interaction is managed by the Inference Engine. 

Here's a simplified overview of how a typical request flows through the system:

1. **The client application formulates a request** containing parameters like the input context and generation options.
2. **The request is sent** to the Inference Engine's endpoint using a protocol like HTTP or gRPC.
3. **Scheduling and batching**: The scheduler queues incoming requests and determines when they will be processed. Requests may be grouped into batches to optimize computational resources. In some cases, a request might be partially processed and then deferred due to resource constraints or scheduling policies.
4. **Model inference**: The model performs a forward pass on the current batch of requests to generate a next token.
5. **Output processing**: The generated tokens are processed and converted into human-readable text.
6. **The system checks if the generation should stop** — either because an End-of-Sequence (EOS) token is generated or the maximum number of new tokens limit is reached. **The response is then prepared**, containing the generated text and any relevant metadata (e.g., reasons for finishing, token probabilities).
7. While the Inference Engine is running, **it provides ways to monitor its performance**, such as logs or metrics in Prometheus format.

![]({{ site.baseurl }}/assets/images/inference-engines/inference-server.png){: .responsive-image style="--img-desktop:90%; --img-mobile:90%;"}


# vLLM: A High-Performance Inference Engine for LLMs

When deploying LLMs in real-world applications, achieving both high efficiency and ease of deployment is crucial. **vLLM** is a state-of-the-art inference engine specifically designed to meet these needs, making it an excellent choice for your deployment strategy.

Let’s lay out some of the reasons that make considering vLLM for your LLM deployments compelling:

- **Efficiency and ease of deployment**: vLLM is optimized for performance, providing fast inference times while being straightforward to set up and integrate into your existing systems.
- **It supports all popular models**, including both text-based LLMs and multimodal models that handle text and images. This wide-ranging support means you can deploy various models without worrying about compatibility issues.
- **vLLM supports popular quantization methods**, enabling you to run large models even on hardware with limited GPU memory; this feature is particularly beneficial when resources are constrained.
- **Built-in monitoring with Prometheus**: vLLM comes with a `/metrics` endpoint compatible with Prometheus, along with a ready-to-use dashboard template. This makes it easy to monitor performance metrics and ensure your deployment runs smoothly.
- **It offers an API compatible with OpenAI's standards**, allowing for seamless integration and minimal changes to your existing codebase.

## Optimizations

Deploying LLMs efficiently requires more than just powerful hardware; it also involves smart optimization techniques to reduce latency and improve throughput. Below are some key techniques that vLLM and other advanced inference servers employ to achieve efficient inference.

### **1. Paged Attention**

As the size and complexity of LLMs increase, efficient memory management becomes crucial for deployment. In traditional inference, token representations (the cache discussed in a previous lecture) are stored contiguously and pre-allocated in memory, leading to inefficiencies, especially with long sequences.

**Paged Attention** in vLLM addresses this by breaking the cache into smaller, flexible blocks (pages) that don’t require contiguous storage. Inspired by virtual memory, this approach efficiently maps blocks to physical memory, reducing waste and allowing more sequences to be processed simultaneously, improving throughput.

Another key benefit of Paged Attention is its ability to share memory during parallel tasks like sampling multiple outputs from the same prompt. This block-sharing dramatically reduces memory usage in complex tasks like beam search, improving overall performance. By utilizing these optimizations, vLLM achieves higher throughput with reduced memory overhead, making large-scale LLM deployments feasible on limited hardware.

And yet another advantage of Paged Attention is that it allows multiple tasks to share memory at the same time. 

For instance, imagine you want to generate several possible responses from the same prompt. With traditional approaches, each of these generations would need its own separate memory, which means a lot of duplication and wasted space. Paged Attention solves this problem by breaking memory into smaller blocks that can be shared. So, when multiple tasks use similar information, they can simply share those blocks instead of each keeping a separate copy. 

As a result, vLLM can handle more tasks simultaneously with less memory, which means it runs more efficiently and can achieve higher throughput. These optimizations make it possible to deploy large-scale language models — even on hardware that doesn’t have enormous memory capacity.

[vLLM’s official blogpost on the topic](https://blog.vllm.ai/2023/06/20/vllm.html)

### **2. Prefix Caching**

Let’s next imagine that you’re deploying LLMs for a text evaluation task, and you need to assess multiple aspects of the texts such as "grammatical correctness", "interestingness", "relevancy," and so on. 

Your prompt would typically include a comprehensive task description, detailed explanations of each aspect, and several few-shot examples to enhance the model's performance. 

To demonstrate, the prompt structure might look like this:

```
<Comprehensive description of the task, e.g.,
"Your task is to classify text based on several aspects...">

<Detailed explanation of what "Grammatically Correct" exactly means>

<Detailed explanation of what "Interesting" exactly means>

...

<Few-shot example text 1>
<Few-shot labels for text 1>

...

<Few-shot example text 5>
<Few-shot labels for text 5>

<Actual text you need to evaluate>
```

Notice that for different texts, the prompt's large prefix remains the same. This means we're repeatedly processing the same information every time we evaluate a new text — which isn't very efficient. **Further, if you're using APIs, you're also paying for these repeated tokens each time.**

This is where **prefix caching** comes to the rescue! By caching the representations of the shared input text — the common prefix — you can reuse them for future requests. This way, you save computational resources and reduce costs because you're not re-processing the same data over and over again.

Prefix caching is especially useful when your requests share a significant common text prefix (like with few-shot examples, or long documents) and only need to generate a small amount of new text. Keep in mind, though, that if you're generating long outputs, the performance gains might be minimal since prefix caching only skips the initial prompt processing, not the computations required for generating output tokens.

Note that this feature is implemented not only in vLLM, but also in API providers like [Gemini](https://ai.google.dev/gemini-api/docs/caching?lang=python).

### **3. Continuous Batching**

Static batching (in transformers, for example) works like this:

1. **Batch formation**: collects a fixed number of requests, say **K**, to form a batch.
2. **Synchronous processing**: processes these **K** requests together, token by token.
3. **Completion wait time**: this means that if some sequences finish early (generate an end-of-sequence token), the system waits until the longest sequence in the batch completes before starting new requests.

This method leads to two main inefficiencies:

- **Waiting for batch formation**: new requests must wait until a full batch is assembled, increasing latency.
- **Underutilized resources**: GPUs remain idle for shorter sequences while waiting for longer ones to finish, wasting computational resources. In real-world applications, input and output sequences can vary greatly in length, making worse the inefficiency.

The image below illustrates the inefficiencies of static batching; each row represents a request (S1, S2, S3, S4) processed over time (T1 to T8).

- **Yellow cells** represent the initial tokens from each sequence's prompt, which are processed in the first forward pass.
- In the first iteration (left side), each sequence generates one new token (represented by the **blue cells**) based on the prompt tokens.
- Over time, as we move through each iteration (right side), different sequences complete their generation by emitting an end-of-sequence token (marked by **red cells**).
- Notice that some sequences, like S3, finish early, while others, like S2, continue until T8.

Despite the sequences finishing at different times, static batching keeps the entire GPU engaged until the longest sequence completes. This approach results in **underutilized GPU resources** for the shorter sequences, which must sit idle until all sequences in the batch are done.


![]({{ site.baseurl }}/assets/images/inference-engines/continuous-batching1.png){: .responsive-image style="--img-desktop:90%; --img-mobile:90%;"}

[Source](https://www.anyscale.com/blog/continuous-batching-llm-inference)

**Continuous batching** addresses these inefficiencies by allowing the batch size and sequences in batch to change dynamically during processing, like so:

1. **Dynamic batch updates**: when a sequence completes, a new request can immediately take its place in the batch.
2. **Efficient GPU utilization**: GPUs remain fully utilized as new sequences are continuously added to the batch.
3. **Reduced latency**: New requests don't have to wait for a batch to form; they join the processing stream as soon as possible.

The image below shows continuous batching in action:

- After each sequence completes (red cells), a new sequence (e.g., S5, S6, S7) is immediately added to the batch.
- This ensures the GPU remains fully utilized by dynamically filling open slots, maximizing efficiency and reducing idle time.

![]({{ site.baseurl }}/assets/images/inference-engines/continuous-batching2.png){: .responsive-image style="--img-desktop:90%; --img-mobile:90%;"}

[Source](https://www.anyscale.com/blog/continuous-batching-llm-inference)

Continuous batching is possible due to advancements in dynamic scheduling algorithms, and efficient memory management techniques like PagedAttention; here’s a [good blog post on the topic](https://www.anyscale.com/blog/continuous-batching-llm-inference).

# SGLang

At times, you may need your LLM to generate a valid JSON output. However, since LLMs are inherently stochastic, ensuring accuracy can be challenging. Additionally, if you require the JSON to contain specific predefined keys, you risk wasting computational resources on tokens that should be generated deterministically, increasing the chances of output errors.

There are techniques out there to deal with this: [OpenAI’s structured output guide](https://platform.openai.com/docs/guides/structured-outputs) is one such example.

In the open-source world, there is a great framework called **SGLang (Structured Generation Language)** that achieves this — and has even cooler features.

Similar to vLLM, SGLang is easy to deploy, supports numerous LLMs, and offers significant flexibility for integration. However, SGLang provides several advanced capabilities that make it particularly attractive for scenarios where structured outputs (also called constrained generation) and flexible programming control are essential.

## **Constrained Generation with Compressed FSM**

When dealing with structured outputs like JSON or any other format requiring consistency, the generation process must control which tokens are allowed at every step. 

Traditional approaches enforce this by building a **Finite State Machine (FSM)** that keeps track of valid token sequences. At each step of token generation, the FSM validates whether a token is allowed, assigning zero probability to any token that doesn’t match the expected structure. While effective for maintaining the output’s format, this method can be highly **inefficient** — ****especially when deterministic sequences (such as the opening braces of a JSON object) must be generated one token at a time. 

**SGLang** addresses this inefficiency with an advanced technique called **Compressed FSM for multi-token decoding**. Instead of generating deterministic sequences token-by-token, SGLang compresses these sequences into a single action, allowing the model to produce multiple tokens all at once. 

For example (see the image below; and for simplicity we assume each token encodes a single character):

- In traditional FSM-based constrained generation, a sequence like `{"summary": "` would be decoded one token-by-token, even though the entire string is predetermined.
- SGLang optimizes this process by combining consecutive parts of the sequence that only have one possible outcome. This means it can generate the whole sequence in one step.
- For example, instead of outputting each character (given our assumption about 1 token equals 1 character) in `{"summary": "` individually, SGLang recognizes that this sequence can be generated all at once, dramatically improving the decoding speed.

In short, by using this compressed FSM, SGLang allows for faster generation by recognizing multi-token sequences that can be produced in a single forward pass.

By utilizing this optimized FSM, SGLang makes sure that structured output is not only **valid** but also generated efficiently, greatly reducing latency compared to traditional FSM-based constrained decoding.

![]({{ site.baseurl }}/assets/images/inference-engines/compressed-fsm.png){: .responsive-image style="--img-desktop:90%; --img-mobile:90%;"}

## **Example of Constrained Generation with SGLang**

Let’s take a simple example to illustrate how SGLang’s constrained generation works:

```python
quest_regex = (
    r"""\{\n"""
    + r"""    "quest_name": "[\w\d\s]{1,32}",\n"""
    + r"""    "difficulty": "(Easy|Medium|Hard|Extreme)",\n"""
    + r"""    "level_requirement": [0-9]{1,2},\n"""
    + r"""    "is_timed": "(True|False)",\n"""
    + r"""    "reward": "[\w\d\s]{1,32}",\n"""
    + r"""    "objective": "[\w\d\s]{1,64}"\n"""
    + r"""\}"""
)

@sgl.function
def quest_gen(s, quest_name):
    s += quest_name + " is a quest in a text-based game. Please fill in the following information about this quest.\n"
    s += sgl.gen("json_output", max_tokens=128, regex=quest_regex)

```

In this example, SGLang ensures that the output JSON format strictly matches a **regular expression** (stored in `quest_regex`). By compressing the FSM transitions, SGLang allows the entire structure to be generated quickly and correctly, ensuring a valid output without going through each token one at a time inefficiently.

SGLang’s powerful combination of **regex-based constraints**, **multi-token optimizations**, and **programmatic control** makes it a highly effective tool for deploying LLMs in production environments that require **predictable and formatted** output.

## **Other Features of SGLang**

SGLang offers a rich set of features that make it ideal for deploying and managing LLMs in sophisticated applications. Here are some of the standout capabilities.

**1. Radix Attention for Efficient Memory Management**

Similar to vLLM’s Paged Attention, SGLang uses Radix Attention to efficiently manage memory by reusing internal representations across multiple generation calls. The radix tree data structure is well-suited for this task because it efficiently handles many overlapping string prefixes, which commonly arise in complex workflows. This allows the system to store and retrieve sequences more effectively, ensuring that overlapping parts of different requests can share memory, leading to better performance and reduced computational overhead. For more detailed information, you can [explore the original paper here](https://arxiv.org/pdf/2312.07104).  

**2. A Flexible Frontend Language**

One of SGLang’s core strengths is its **flexible frontend language** which allows users to programmatically manage and interact with LLMs in an intuitive way. This flexibility is particularly useful for applications that require complex, multi-step workflows or custom interactions beyond simple text prompts.

Here are some key features of this language:

- You can structure prompts with **conditional logic** and **control flow**, similar to conventional programming languages. This means that depending on model outputs, you can explicitly direct the model to take different actions, which is very useful for tasks that require decision-making. For example:
    
    ```python
    @sgl.function
    def tool_use(s, question):
        s += "To answer this question: " + question + ". "
        s += "I need to use a " + sgl.gen("tool", choices=["calculator", "search engine"]) + ". "
    
        if s["tool"] == "calculator":
            s += "The math expression is" + sgl.gen("expression")
        elif s["tool"] == "search engine":
            s += "The key word to search is" + sgl.gen("word")
    ```
    
    In this example, depending on the tool chosen by the model (`calculator` or `search engine`), the appropriate prompt is generated, allowing dynamic and context-sensitive workflows.
    
- SGLang also supports **parallel prompt execution**. This means you can create multiple generations simultaneously, especially beneficial for speeding up workflows that involve generating multiple independent outputs. For instance, if you're expanding different parts of a text in parallel:
    
    ```python
    @sgl.function
    def tip_suggestion(s):
        s += (
            "Here are two tips for staying healthy: "
            "1. Balanced Diet. 2. Regular Exercise.\n\n"
        )
    
        forks = s.fork(2)
        for i, f in enumerate(forks):
            f += f"Now, expand tip {i+1} into a paragraph:\n"
            f += sgl.gen(f"detailed_tip", max_tokens=256, stop="\n\n")
    
        s += "Tip 1:" + forks[0]["detailed_tip"] + "\n"
        s += "Tip 2:" + forks[1]["detailed_tip"] + "\n"
        s += "In summary" + sgl.gen("summary")
    ```
    
    In this example, the model expands each health tip in parallel, maximizing efficiency and reducing latency. 
    
    For more details, [check out the original paper here](https://arxiv.org/pdf/2312.07104).
    

# Triton Inference Server with TensorRT Backend

If you're looking for even faster inference speeds and don't mind a bit more setup complexity, NVIDIA's **Triton Inference Server with TensorRT Backend** could be the solution for you. Built specifically to get the most out of NVIDIA GPUs, it leverages advanced optimization techniques to squeeze out every bit of performance from your LLM deployment.

**TensorRT-LLM**, a part of this solution, focuses on optimizing LLMs for production use. Unlike other engines that serve raw model weights directly, TensorRT-LLM first **compiles and optimizes the model**, fine-tuning the kernel operations to ensure every step is as efficient as possible. (A key thing to note is that TensorRT-LLM is designed solely for NVIDIA GPUs, and the same GPU must be used for both model compilation and inference.)

The **Triton Inference Server** acts as the serving framework that utilizes TensorRT-LLM for optimized deployment. Essentially, Triton manages the entire pipeline — handling incoming requests, scheduling them, and so on. TensorRT-LLM, on the other hand, works under the hood to make the inference process itself faster by using the optimized, compiled version of the model. This combination allows developers to take advantage of both Triton's flexible serving capabilities and TensorRT's speed-focused model optimizations.

This solution incorporates numerous optimizations, including Paged Attention, Continuous Batching, and Prefix Caching, which we discussed earlier. Also, Triton Inference Server provides a **metrics endpoint**, enabling easy performance monitoring. Similar to vLLM's `/metrics` endpoint, it integrates seamlessly with **Prometheus** (this tool will be reviewed in the next lessons), allowing you to track key metrics like throughput, latency, GPU utilization, and more.

In summary, if you're deploying large-scale language models and need to maximize inference speed and efficiency on NVIDIA hardware, Triton Inference Server with TensorRT Backend offers a highly optimized solution.
