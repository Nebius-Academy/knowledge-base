---
layout: post
title: "Quantization"
categories: blog
permalink: /quantization/
---

**By: [Alexey Bukhtiyarov](https://www.linkedin.com/in/leshanbog/)**

# Quantization

Imagine you want to run a huge language model on your phone (or on a small server with limited GPU memory). Using a full-sized model would take up so much memory and computing power that it might not even run! And even if it did — it could be painfully slow. **Quantization** helps by making the model *smaller* and *faster*, so it can work efficiently even on hardware with less power.

In typical deep learning models, numbers are stored in formats like **float32** (32-bit floating-point); this offers high precision but takes up a lot of memory. At its core, quantization is about reducing the precision of the numbers that represent a model’s parameters (the weights and activations). Quantization reduces this level of precision, using fewer bits to represent these numbers, which reduces both memory usage and computational cost.

## Number Representation: From Float to Integer

### Floating point numbers

To actually understand how quantization reduces the precision of numbers, it’s important to first explore how floating-point numbers are represented in computers. In formats like **float32**, numbers are represented with three main components: **sign**, **exponent** (which sets the range), and **mantissa** (which holds the precision). These components allow floating-point numbers to cover a wide range of values while maintaining a certain level of precision.


![]({{ site.baseurl }}/assets/images/quantization/comparing-number-formats.png){: .responsive-image style="--img-desktop:90%; --img-mobile:90%;"}

In general, a floating-point number is represented as follows:

$$
(-1)^{\text{sign}} \times (1 + \text{mantissa}) \times 2^{(\text{exponent} - \text{bias})}
$$

Let’s expand on what the above implies:

- **Sign** determines whether the number is positive or negative. It’s represented by a single bit, where `0` indicates a positive number and `1` indicates a negative number.
- **Exponent** represents the scale or magnitude of the number. It shifts the value by a power of 2. For float32, the exponent is stored using 8 bits. To accommodate both positive and negative exponents, a bias is added, and in the case of float32, this bias is 127.
- **Mantissa (fraction)**: Represents the precision bits of the number. In float32, this uses 23 bits.

The chart below shows how different values are represented using a simulated naive "Float5" format, which has a reduced number of bits compared to standard floating-point formats like Float32. Each row represents numbers simulated with a different exponent value, and only positive numbers are displayed.

![]({{ site.baseurl }}/assets/images/quantization/float5-numbers.png){: .responsive-image style="--img-desktop:80%; --img-mobile:90%;"}

Here we have the following:

- 1 bit for sign.
- 2 bits for the exponent, so that the biased exponent may be 0, 1, 2, or 3.
- A bias equal to 1, so that the final exponent is -1, 0, 1, or 2,
- 2 bits for the mantissa. Mantissa thus represents numbers `0.**` with two binary digits, giving 0, 0.25 (binary `0.01`), 0.5 (binary `0.10`), and 0.75 (binary `0.11`).

Note that in this example, we’ve omitted several important things for the sake of simplicity: how to encode zero (in the most common formats there are two zeros of different signs, both with zero exponent and mantissa), how to encode `nan` and `inf` of various kinds. You can find more details if you google the **IEEE Standard for Floating-Point Arithmetic (IEEE 754)**.

Note also that floating-point formats allow to represent both large and small numbers, but with different density. So, in the above visualization:

- For lower exponent values (closer to the bottom of the chart), the represented values are densely packed.
- As the exponent increases (toward the top of the chart), the distance between consecutive represented values grows, illustrating the sparsity in represented numbers.

This effect demonstrates the trade-off in floating-point representations: higher exponent values allow for a larger range of numbers but reduce precision, while lower exponent values provide finer precision but cover a smaller range.

### Integer numbers

Quantization often goes further by converting numbers into **integers**, such as **int8** (8-bit integers) or even **int4** (4-bit integers). These representations sacrifice both range and precision, but the trade-off significantly reduces the size of the model. For example, int8 representation requires only 8 bits per number, compared to 32 bits for float32, reducing memory usage by a factor of four.

Unlike **float32**, which represents numbers with a sign, exponent, and mantissa, **int8** uses a simple fixed-point format. In an **int8** representation, the number is stored as a signed or unsigned 8-bit integer. For a signed **int8**, the range of representable values is from **-128 to 127** (inclusive), with one bit used to denote the sign (positive or negative) and the remaining 7 bits used for the value itself. In the case of unsigned **int8**, the range is from **0 to 255**.

## Another angle on the basic idea of quantization

Quantization reduces the precision of a model’s parameters (such as weights) to make it smaller and more computationally efficient. Through this process, the original high-precision values (often stored in formats like float32) are approximated using a lower precision representation, such as int8. This involves mapping the original range of values to a smaller set of discrete levels, effectively "scaling down" the granularity.

The diagram helps illustrate this concept:

![]({{ site.baseurl }}/assets/images/quantization/number-compression.png){: .responsive-image style="--img-desktop:90%; --img-mobile:90%;"}

The **curve** at the top represents the original weight distribution, where each **blue dot** corresponds to a distinct original weight value, reflecting **higher granularity** (high precision).

Meanwhile, the **grid** at the bottom shows the reduced set of possible values after quantization, corresponding to **lower granularity** (lower precision). Each original weight is mapped to the closest available quantized value, as indicated by the vertical red lines.

During quantization, this approximation process reduces the number of unique values the model uses, compressing the model while retaining a reasonable level of accuracy. By using fewer bits (e.g., 8 bits in int8 compared to 32 bits in float32), quantization enables models to fit onto memory-constrained hardware and run more efficiently.

The trade-off lies in the precision of these mapped values. The fewer levels available (lower granularity), the more information is lost in the mapping, but the gains in memory efficiency and speed can significantly benefit deployment on resource-limited devices.

## Key Ideas to Wrap Up

Let’s recap the lesson:

- **Quantization**: Reduces the precision of a model’s parameters to make it smaller and more computationally efficient.
- **Floating Point Numbers**: Represented with sign, exponent, and mantissa, allowing a wide range of values with high precision.
- **Integer Numbers**: Quantization often converts numbers into integers (e.g., int8 with only sign and mantissa), reducing memory usage and computational cost.
- **Trade-offs**: Quantization involves a trade-off between precision and efficiency, mapping high-precision values to lower precision representations.

Quantization is a crucial technique for optimizing deep learning models, enabling them to run efficiently on hardware with limited resources. By simplifying the representation of model parameters, quantization significantly decreases memory usage and computational cost, making it possible to deploy large models on devices with constrained capabilities. In the next lesson, there will be a practice session to apply the concepts of quantization, reinforcing the understanding of this important topic.



