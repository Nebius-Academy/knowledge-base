---
layout: post
title: "LLM Inference Essentials"
categories: blog
permalink: /evaluation-1-basics/
---


**Authored by** **Emeli Dral** and **Elena Samuylova**, creators of [**Evidently**](https://www.evidentlyai.com/) ([GitHub](https://github.com/evidentlyai/evidently)), an open-source ML and LLM evaluation framework with 25M+ downloads:


![]({{ site.baseurl }}/assets/images/evaluation/evidently_ai_logo_docs.png)
){: .responsive-image style="--img-desktop:50%; --img-mobile:75%;"}

This is the first of five chapters.

* [Chapter 1](https://nebius-academy.github.io/knowledge-base/evaluation-1-basics/)
* [Chapter 2](https://nebius-academy.github.io/knowledge-base/evaluation-2-generative-systems/)
* [Chapter 3](https://nebius-academy.github.io/knowledge-base/evaluation-3-deterministic/)
* [Chapter 4](https://nebius-academy.github.io/knowledge-base/evaluation-4-model-based/)
* [Chapter 5](https://nebius-academy.github.io/knowledge-base/evaluation-5-production-monitoring/)

Together with this theoretical introduction, you can explore a [practical Python example on different LLM evaluation methods](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic5/5.1_llm_evaluation.ipynb).

# Introduction to AI Evaluation

When you build an AI system — whether it’s helping users write emails, answer questions, or generate code — you need to know: is it actually doing a good job? To figure that out, you need evaluations — or **"evals."** And not just once. Evaluations are essential throughout the entire AI system lifecycle.

- **During development**, you're experimenting: trying out different models, architectures, or prompts. Evals help make better decisions and design a system that performs well.  
- **In production**, it's about monitoring. You need to ensure the system keeps delivering high-quality results when real users rely on it.  
- **When making updates**, like changing a prompt or switching to a new model version, evaluations help you test for regressions and avoid making things worse.

This need for evaluation isn’t new. In fact, it’s a well-established practice in traditional machine learning — where you may be training models to solve tasks like spam detection or predicting sales volumes. In those cases, evaluation is typically objective: there's a correct answer, and you can directly measure whether the model got it right.

But things get more complex with LLM-powered applications. Instead of predicting a single outcome, these systems often help write text, summarize articles, translate languages, etc. And in these cases, there isn’t always one "correct" answer.  

Instead, you're often dealing with a range of possible responses — some better than others. Quality also often involves more subjective criteria, like tone, clarity, helpfulness. For example:

- If an AI writes an email, how do you know it’s not just grammatically correct, but actually moves the conversation forward or resolves the issue?
- If an AI answers questions about internal processes, how do you know it's accurate, up-to-date, and truly useful?

These kinds of tasks are what make generative systems so powerful — you can automate entirely new kinds of work. But this also adds significant complexity when it comes to evaluation.

*In this guide, we’ll explore how to evaluate AI systems built with large language models — from development to real-world monitoring.*

# Chapter 1: Basics of AI system evaluation

*In this chapter, we introduce different types of AI tasks — and explain how they shape the evaluation process. We also briefly cover how predictive systems are typically evaluated to learn from existing foundations*.

## Discriminative vs Generative tasks 

AI systems and models can be broadly divided into two categories: **discriminative** and **generative**. Each serves a different purpose — and their evaluation methods reflect these differences.

### Discriminative tasks

Discriminative models focus on making decisions or predictions.

They take input data and classify it into predefined categories or predict numerical outcomes. In simpler terms, they’re great at answering "yes or no," "which one," or "how many" questions.

**Examples:**
- Classifying images as cat or dog.  
- Predicting the probability that an email is spam.  
- Identifying the sentiment of a movie review as positive or negative.  
- Predicting sales volumes.  

You can think of discriminative models as decision-makers. Their goal is to predict the target variable given a set of input features.

Common categories of tasks include:
- **Classification**: Predicting discrete labels (e.g., fraud vs. not fraud)  
- **Regression**: Predicting continuous values (e.g., monthly revenue)  
- **Ranking**: Predicting the relative ordering of items (e.g., search, recommendations)  

You would typically train narrow ML models on your own historical data to solve such predictive tasks. But you can also use pre-trained models — including LLMs — for some of these.  For example, use an LLM to classify support queries into a set of predefined topics. In these cases, you can apply the same evaluation approaches as you would with traditional ML models.

### Generative tasks

Generative models, on the other hand, create new data that resembles the data they were trained on. Instead of classifying or predicting, they generate content — such as text, images, code, conversations, or even music.

**Examples:**
- Writing a creative story  
- Summarizing an article  
- Generating realistic images of non-existent people  
- Translating text from one language to another  
- Drafting a response to an email using the provided information  

Generative models are creators. They learn the underlying patterns in the data and use that to produce something new.

**Quick Comparison: Discriminative vs. Generative Tasks**

|                        | **Discriminative Tasks**       | **Generative Tasks**            |
|------------------------|-------------------------------|----------------------------------|
| **Goal**               | Make decisions                 | Create new content               |
| **Output**             | Labels, scores or predictions  | Text, code, images, or audio     |
| **Evaluation**         | Clear, objective metrics       | Complex, subjective metrics      |

Many modern applications of LLMs including AI agents and chatbots fall under generative tasks. However, before we dive into evaluating them, let’s first quickly review how this works for more traditional predictive (discriminative) systems. This helps with understanding core evaluation principles.

## Evaluating predictive systems

The process of evaluating discriminative systems is well-established and relatively straightforward.

### Evaluation process

**Offline evaluation**. Evaluation typically happens offline: you run an evaluation script or pipeline as you experiment and build your ML model. You assess its performance using a designated test dataset.

**Test dataset**. The test dataset contains "groud truth" labels or values for a set of known test inputs. You make the evaluation by comparing model predictions against what's expected. This test dataset that represents your use case can be manually labeled or curated from historical data. 

In fact, having this dataset is a prerequisite for ML model development: you need it first to train the model itself! You then use part of this same dataset for evaluation. Typically, the data is split into:
- **Training set**: used to fit the model.  
- **Validation (or development) set**: used to tune hyperparameters and guide model improvements.  
- **Test set**: reserved for evaluating final model performance on unseen examples.  

For example, you might work with a collection of emails labeled as spam or not, or a historical record of product sales transactions. After training the model using most of the data, you evaluate it by comparing its predictions against the known outcomes in the remaining test set.

### Evaluation metrics

Depending on the type of task and the types of errors you prioritize, you can choose from several standard evaluation metrics. Here are examples based on the type of the task.

- **For regression tasks**, common evaluation metrics include:  
  - Mean Absolute Error (MAE)  
  - Mean Squared Error (MSE)  
  - Root Mean Squared Error (RMSE)  
  - R-Squared (Coefficient of Determination)  
  - Mean Absolute Percentage Error (MAPE)

- **For classification tasks**, typical metrics are:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  
  - Logarithmic Loss  
  - AUC-ROC

- **For ranking tasks**, you often use metrics like:  
  - Mean Reciprocal Rank (MRR)  
  - Normalized Discounted Cumulative Gain (NDCG)  
  - Precision at K  

> 📖**Further reading:** You can find more detailed guides on [Classification Metrics](https://www.evidentlyai.com/classification-metrics) and [Ranking Metrics](https://www.evidentlyai.com/ranking-metrics).

While there is some diversity in metrics, the evaluation process for predictive systems is relatively straightforward: there is a single "right" answer and you can essentially quantify how many predictions were correct or close enough. (In tasks like multi-label classification, where multiple correct labels can apply to a single instance, similar metrics are adapted to evaluate across all relevant labels.)

### Production monitoring

In production settings, it’s often not possible to immediately verify whether model predictions are correct. Real labels are often only available after some time — either through manual review or once real-world outcomes are known.

You typically continue monitoring model quality by:

- **Collecting these delayed labels or actual values.**  You can manually label a sample of model outputs or collect real-world outcome data as it becomes available. For example, after predicting the estimated time of delivery (ETA), you can wait and capture the actual delivery time. This allows you to periodically recompute evaluation metrics, such as mean error, accuracy etc.

- **Using proxy metrics.** When labels are unavailable or too slow to collect at scale, you can rely on indirect monitoring signals, such as:
  - **Prediction drift monitoring**, which tracks changes in the distribution of model outputs over time (e.g., helps detect a shift toward more frequent fraud predictions).
  - **Input data quality checks**, which helps make sure that incoming data is valid and not corrupted.
  - **Data drift monitoring**, which tracks changes in input data distributions to verify that the model operates in a familiar environment.

> 📖 **Further reading:** Learn more about [data drift](https://www.evidentlyai.com/ml-in-production/data-drift). 

### Evaluating LLM outputs

Whenever you use LLMs for predictive tasks, you can apply the same evaluation approaches as with traditional machine learning models — both offline and online. However, evaluating **generative tasks** is more complex. When multiple valid answers are possible, you can't simply check if an answer exactly matches a given reference. You need more complex comparative metrics as well as an option to evaluate more subjective quality dimensions such as clarity or helpfulness.

*In the next chapter, we’ll introduce methods and tools for evaluating generative tasks*.

[Continue to the next chapter](https://nebius-academy.github.io/knowledge-base/evaluation-2-generative-systems/)
