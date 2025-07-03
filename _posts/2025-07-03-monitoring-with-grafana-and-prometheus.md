---
layout: post
title: "LLM Monitoring with Grafana and Prometheus"
categories: blog
permalink: /monitoring-with-grafana-and-prometheus/
---

**By: Alexey Bukhtiyarov**


# Basic monitoring tools: Grafana and Prometheus

During the lifecycle of our LLM deployment, we may encounter various situations that raise important questions:

- **The response time seems too long** — how much of that time is spent on LLM inference, and where might there be bottlenecks?
- In terms of **user experience**, what percentage of our users are getting responses in under 5 seconds?
- **How many requests can our current hardware handle** without performance degradation?

These are the kinds of questions that effective monitoring can answer. Monitoring plays a crucial role in understanding the behavior of your system, ensuring performance is maintained, and identifying issues as they arise. So, using some basic tools, we can create a dashboard to monitor the performance of an LLM in real time:

![]({{ site.baseurl }}/assets/images/monitoring-with-grafana-and-prometheus/llm-monitoring-plots.png){: .responsive-image style="--img-desktop:90%; --img-mobile:90%;"}

Looking at this real production dashboard for an 8B parameter LLM, we can observe the following:

- **First line**: The average end-to-end request latency indicates that the LLM inference time for a single request is ~2.6 seconds.
- **Second line**: The processing speed for input tokens is significantly faster than that for output tokens, ~10 times faster in our production scenario — 3500 prompt tokens per second versus 350 generated tokens per second.
- **Third line**: The average time taken to process all input tokens, also known as the "**Time to First Token**" (**TTFT**),  is only about five times longer than the time required to generate a single output token.
- **A daily cycle in compute load**: A distinct daily cycle is evident, with fluctuations in compute load aligning with peak usage times, impacting latency and throughput patterns.

### **Prometheus and Grafana: Monitoring Duo**

To monitor LLM deployments effectively, **Prometheus** and **Grafana** are two powerful tools, widely used in the industry:

1. **Prometheus** is an open-source system that collects real-time metrics from your application, like how many requests are being processed, how long each of them take, how much GPU memory is used, and so on. These metrics are collected at specific intervals and stored in a time-series database, making it easier to track performance over time.
2. **Grafana** is a visualization tool that allows you to take all the metrics collected by Prometheus and create beautiful and insightful dashboards. These help you see — in real-time — the current state of your application and make informed decisions based on the data.

And **these two tools work together seamlessly**: Prometheus acts as a data collector and a database, while Grafana presents that data in a way that's easy to understand and interpret. More specifically, below is the diagram showing how Grafana and Prometheus work together:

![]({{ site.baseurl }}/assets/images/monitoring-with-grafana-and-prometheus/prometheus-and-grafana.png){: .responsive-image style="--img-desktop:90%; --img-mobile:90%;"}

(image [source](https://www.linkedin.com/pulse/getting-started-prometheus-grafana-beginners-guide-majid-sheikh-1f/))

- The **API Server** and **Client Server** act as the "targets" for Prometheus, each exposing a metrics endpoint.
- **Prometheus** is responsible for periodically **pulling metrics** from these targets. This pulled data is stored in Prometheus's time-series database, allowing easy access to both current and historical performance information.
- Then, **Grafana** **queries Prometheus** to gather this stored data and present it visually on customized dashboards.

### An example in action

If you want to track something apart from the default metrics, you can do in a fashion similar to the following example.

1. **Add Prometheus to your application**. This process essentially adds a `/metrics` endpoint to your application, which exposes all the metrics that can be tracked.
    
    ```python
    from flask import Flask, request
    from prometheus_flask_exporter import PrometheusMetrics
    
    app = Flask(__name__)
    metrics = PrometheusMetrics(app)
    ```
    
2. **Define a Prometheus counter** (there are several types of metrics available, but we'll use a counter for this example). You can associate different labels with a metric, for instance, to differentiate between clients by their names:
    
    ```python
    from prometheus_client import Counter
    
    request_counter = Counter(
        'openai_requests_total',
        'Total number of OpenAI requests',
        ["client_name"]
    )
    ```
    
3. **Update the counter in your application**; since we're using a counter, we'll increment it every time an event occurs.
    
    ```python
    request_counter.labels(client_name=current_client_name).inc()
    ```
    
- Working example: a Prometeus counter keeping ~~trach~~ track of the total number of OpenAI requests
    
    ```python
    from flask import Flask, request
    from openai import OpenAI
    from prometheus_client import Counter
    from prometheus_flask_exporter import PrometheusMetrics
    
    app = Flask(__name__)
    metrics = PrometheusMetrics(app)
    
    request_counter = Counter('openai_requests_total', 'Total number of OpenAI requests', ["client_name"])
    
    @app.route('/call_openai_api', methods=['POST'])
    def call_openai_api():
        current_client_name = request.get_json()["client"]
        request_counter.labels(client_name=current_client_name).inc()
    
        prompt = request.json.get('prompt')
    
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
    
        return completion.choices[0].message.content
    
    if __name__ == '__main__':
        # Run the Flask application
        app.run(host='0.0.0.0', port=8765)
    
    ```
    

Similarly, you can create **histogram metrics** to measure request latency and track the distribution of response times, providing a more detailed understanding of user experience. You can also monitor **resource usage** like CPU, GPU, and memory utilization, which are critical to ensure that your model inference remains efficient.
