---
layout: post
title: "LLM Inference Essentials"
categories: blog
permalink: /llm-inference-essentials/
---

**By: [Sergei Skvortsov](https://www.linkedin.com/in/sergei-skvortsov/)**

Up to this point, it’s natural to wonder: *How can I ensure I’m getting the most out of my LLM developments?*  In this lesson, and the following 2, you will delve into the essential concepts of inference, its metrics, and the underlying mathematics, which are crucial for developing and optimizing LLMs. Understanding these concepts is vital for efficiently deploying LLMs, as it enables you to measure and improve their performance, ensuring they meet the desired standards for various applications. By mastering these topics, you'll be equipped with the knowledge needed to make informed decisions about model deployment, resource allocation, and overall system efficiency.

# Introduction to LLM Inference

In the context of LLMs, **inference** is the process where the model takes in a given input (like a sentence or a question) and generates an output (like a response or a continuation of the text). It relies heavily on GPU acceleration and many optimization tricks to handle the intensive computations.

In Topic 4, we discuss how inference operates and what are potential bottlenecks. In Topic 5, we'll be covering optimizations that make inference more efficient.

## Inference Process

You can think of inference as the model's way of *"thinking"* and producing answers based on the patterns and knowledge it has learned during training. During inference, the model's parameters (weights and biases) are used to process the input and generate the output. In simple terms, **weights** determine the importance of each input feature, kind of like highlighting the key points in a text. **Biases**, on the other hand, help adjust the output along the way, ensuring it's accurate and relevant. Together, these parameters play a vital role in how the model understands the input and generates responses.

Here is a schematic overview of the LLM inference steps:

![]({{ site.baseurl }}/assets/images/llm-inference-and-its-metrics/llm-inference-schematics.png){: .responsive-image style="--img-desktop:90%; --img-mobile:90%;"}

There are three key steps:

1. **Downloading**: This stage involves downloading the model weights, typically from a cloud storage service like Hugging Face. These weights are then stored on the computer’s disk where you plan to run the LLM, whether on a local machine or a remote server.
2. **Loading**: In this step, the LLM’s weights are loaded into RAM or GPU memory, preparing the model for generating new tokens. Depending on the hardware, you can run the model on either a CPU or GPU, with GPUs predominantly used for modern LLMs due to their superior performance.
3. **Generation**: Finally, the model is ready to generate new tokens. As tokens are generated, they are concatenated into the output sequence, which is returned as the result of the inference process.

> **Important note**: During inference, the model’s weights are not updated; they remain unchanged.
> 

During the **generation** step, we can distinguish two stages:

- **Prompt processing**: In this stage, the LLM processes the prompt. When we introduced transformer architecture, we mentioned that self-attention mechanism allows each token to "look" at previous tokens, transforming their internal representations. While this remains true, we can optimize this process for an input sequence by performing these “looks” simultaneously, making it faster (though still quadratic in complexity).
    
    Prompt processing results in the generation of the first token in the output sequence (New token 1).
    
- **Autoregressive decoding**: In the second stage, the LLM generates new tokens **autoregressively**. This means tokens are generated one by one. To generate the (N+2)-th token, the model must first "look" at the (N+1)-th token (and all the previous ones). This process cannot be parallelized, so each token is computed sequentially. The decoding continues until either an **End Of Sequence (EOS)** token is generated or the maximum sequence length is reached.

### Processing Inference at GPUs

In LLM inference (as everywhere in Deep Learning), GPUs play a critical role by parallelizing tasks, enabling faster computations and efficient resource utilization. 

When examining any GPU specific hardware specifications, you'll notice a variety of different parameters. For instance, you can find the specifications for the H100 GPU on Nvidia's official website: [link](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-tensor-core-gpu-datasheet)

Let’s take a closer look at these parameters:

![]({{ site.baseurl }}/assets/images/llm-inference-and-its-metrics/h100-specification.png){: .responsive-image style="--img-desktop:70%; --img-mobile:90%;"}


For our purposes, we need to focus on the following:

- **Peak FLOPS**: This refers to how many floating-point operations (such as multiplication and addition) the GPU can perform per second. These values can vary depending on the precision of the model’s weights, meaning how many bits are used to store each parameter.
    
    For inference, FP16 and BFLOAT16 are commonly used, which are formats for storing floating-point numbers in 16 bits of memory.
    
    For the H100 GPU, the peak FLOPS for FP16 and BFLOAT16 is 1,979 or 1,671 teraFLOPS depending on the configuration. Nebius serves H100 NVL, so we will use this number in further calculations.
    
- **GPU memory size**: This is the total amount of memory available on the GPU. For the H100, depending on the model, it can be either 80 GB or 94 GB.
- **GPU memory bandwidth**: This is the maximum speed at which data can be transferred between the compute engine (or CUDA cores) and the GPU memory. It determines how quickly data can be loaded or stored during computation.

To understand these parameters better, let’s explore how a modern GPU works using a simple analogy. There is an excellent article called [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html), where the author explains that a modern GPU can be compared to a factory.

To produce goods, raw materials must be transported from a warehouse (GPU memory) to the factory (compute). The speed at which these materials are delivered from the warehouse to the factory is determined by the **memory bandwidth**. Once the factory receives the materials and processes them (i.e., performs computations, ideally at maximum **Peak FLOPS**), the results are then transported back to the warehouse (GPU memory).

![]({{ site.baseurl }}/assets/images/llm-inference-and-its-metrics/factory-metaphor.png){: .responsive-image style="--img-desktop:70%; --img-mobile:90%;"}

Keeping this in mind, let’s return to how GPUs work. A typical GPU consists of GPU memory (or HBM) and multiple streaming multiprocessors (SMs). An SM is like a large factory with small local storage and several conveyor belts or furnaces (CUDA cores) that can work simultaneously, but should all perform the same computation.

To perform operations, data must first be loaded from the GPU memory (turquoise colour) into the local memory (or SRAM) of each SM (purple colour) at the speed determined by the GPU memory bandwidth (orange). Then, all compute engines (or CUDA cores) within the SMs process the data simultaneously. Once the computation is complete, the results are stored back in the GPU memory.

It’s important to note that to achieve peak FLOPS, the operation must fully utilise all compute engines on the GPU.

![]({{ site.baseurl }}/assets/images/llm-inference-and-its-metrics/gpu-memory.png){: .responsive-image style="--img-desktop:70%; --img-mobile:90%;"}

# Inference metrics

## Inference Metrics Overview

Measuring inference is essential for understanding and optimizing the performance of LLMs in real-world applications. Inference metrics provide critical insights into how efficiently a model processes data, how quickly it responds to user requests, and the resources required for deployment. These measurements are key to identifying bottlenecks, balancing trade-offs, and tailoring models to specific use cases, ensuring that they meet performance, scalability, and budgetary requirements. By systematically evaluating inference, organizations can make informed decisions to maximize the impact of their AI systems

There are three important metrics:

**Throughput**: This refers to the number of requests or tokens an LLM can process per second. Increasing throughput allows the system to handle more tokens and requests in the same amount of time, improving overall efficiency.

**Latency**: This is the time elapsed from when a user makes a request until the LLM provides a response. This metric is especially critical in real-time and interactive applications, such as chatbots and real-time systems, where quick response times are crucial.

**Cost**: This represents the amount of money spent to process a request. Reducing costs is a common goal, and one way to achieve this is by deploying your own LLM, which you can optimize to lower expenses compared to using third-party services.

When considering the cost of using LLM inference services (for example, Nebius AI: [https://studio.nebius.ai](https://studio.nebius.ai/)), you'll notice that input tokens typically cost about three times less than output tokens. As discussed earlier, this difference arises because:

- Processing input tokens generally requires one optimized (although quadratic) pass through the LLM, whereas
- Generating output tokens involves multiple passes — one for each token produced. This increased computational demand for generating output is the reason output tokens cost more than input tokens.


![]({{ site.baseurl }}/assets/images/llm-inference-and-its-metrics/nebius-ai-studio-screen-06.06.2025.png){: .responsive-image style="--img-desktop:90%; --img-mobile:90%;"}


### **Memory-bound and Compute-bound Programs**

Distinguishing between memory-bound and compute-bound programs is crucial for interpreting inference metrics and optimizing LLM performance. These classifications describe whether a model's performance is limited by memory access speeds or computational power, directly influencing key metrics like throughput, latency, and cost. Therefore, linking these concepts to inference metrics provides deeper insights into how resource utilization and workload dynamics shape the performance of large language models, enabling more effective optimization strategies.

The main difference between these two types of programs is as follows:

- **Compute-bound program**: Its execution time mostly depends on the time spent on calculations. A good example of a compute-bound program is matrix multiplication, which requires intensive mathematical calculations.
    
    One way to speed up this type of program is by increasing the speed of the compute engine.
    
    In our factory metaphor, this would be a factory where raw material is processed more slowly than it can be transported from the warehouse.
    

- **Memory-bound program**: Its execution time primarily depends on the speed of data movement. An example of a memory-bound program is matrix transpose. This task involves loading the data into the compute engine, transposing it, and saving it back to memory. Since transposition is a very simple calculation, the operation’s total time is dominated by the time it takes to move the matrix in and out of memory.
    
    Enhancing performance for memory-bound programs typically involves increasing memory bandwidth.
    
    In the factory metaphor, this would be a factory with fast processing but slower delivery of raw materials from the warehouse.
    

It's important to note that a program can be either compute-bound or memory-bound depending on particular optimizations, parameter values, and hardware used. For LLMs, adjusting parameters like batch size affects how a program utilizes computational resources and memory, impacting overall performance. In the next sections, we’ll explore how changing batch size can shift LLM inference between being memory-bound and compute-bound.

### Throughput - batch size plane

The throughput-batch size plane is a crucial framework for understanding the interplay between inference metrics and the characteristics of memory-bound and compute-bound programs. This concept highlights how varying batch sizes can influence throughput and latency, depending on whether the program is constrained by memory access or computational capacity. For memory-bound programs, increasing batch size may improve throughput up to a point before hitting memory limitations, while compute-bound programs may scale more effectively with larger batches until processing power becomes a bottleneck. By analyzing performance across the throughput-batch size plane, we can uncover optimal configurations that balance these metrics and constraints, enabling efficient and cost-effective deployment of LLMs. Let’s illustrate these relationships using the performance plane.

![]({{ site.baseurl }}/assets/images/llm-inference-and-its-metrics/throughtput-batch-size-plane.png){: .responsive-image style="--img-desktop:70%; --img-mobile:90%;"}

On this plane, there are two areas:

- When the batch size is small, the LLM operates in a memory-bound regime.
- As the batch size reaches a certain threshold, the model transitions to a compute-bound regime. We’ll discuss the intuition behind this a bit later.

Let’s use the factory metaphor again to get a rough understanding of this concept. Batch size is like the amount of raw materials delivered to the factory in a single shipment. The factory’s furnaces also have a maximum capacity for how much raw material can be processed at once. When the batch size is small, the factory can process the entire batch, so the throughput (the amount of raw material processed per unit of time) increases linearly. However, once the furnaces reach their peak capacity, the throughput can no longer increase; instead, any extra raw material will have to wait its turn. While this isn’t a perfectly accurate description, it helps illustrate the basic idea.

An important point here is that when the model is in the memory-bound area, its throughput increases almost linearly. However, in the compute-bound area, the throughput remains constant and reaches the peak throughput.

The **peak throughput** is the maximum throughput the model can achieve. It depends on several factors, including the model's architecture, size, and the underlying hardware. Understanding how to calculate peak throughput involves analyzing the model’s computational demands and the capabilities of the hardware. We will explore this in detail shortly.

