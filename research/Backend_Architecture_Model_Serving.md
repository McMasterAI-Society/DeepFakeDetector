# Backend Architecture & Model Serving
Contributer: Andrew Wu

## Table of Contents
[Choosing API Framework](#flask-vs-fastapi)<br/>
[API Endpoints](#api-endpoints)<br/>
[Model Optimization Techniques](#model-optimization-techniques)<br/>
[GPU vs CPU Deployment](#gpu-vs-cpu-deployment)<br/>
[Batch Processing](#batch-processing)<br/>
[Caching Strategies](#caching-strategies)<br/>
[Containerization](#containerization)<br/>
[Extra Notes](#other-notes)<br/>

## Flask vs FastAPI?

| Aspect | Flask | FastAPI |
| :----- | :---- | :------ |
| Purpose| lightweight WSGI web application framework | modern, async-first, web framework |
| Async support | no in-built support | native support|
| Performance | slower speed b/c synchronous processing ~4,000-5,000 requests per second in benchmarks | high performance b/c asynchronous and non-blocking request handling`` ~20,000+ request per second in benchmarks |
| Data validation | manual | automatic via Pydantic |
| Templating | built-in using Jinja2 | possible with Starlette support |
| Type hints | optional | required by default |
| Auto API docs | not included | built-in using SwaggerUI + ReDoc |
| ORM support | uses plugins like Flask-SQLAlchemy | External |
| Use cases | web apps, dashboards, small APIs | APIs, microservices, ML serving |

<br/>
For our DeepFake Detection project, <b>FastAPI is a much better choice</b>.
<br/>
- FastAPI also provides better support for concurrency and async I/O operations, making the application more scalable.
- Flask would need extra tools like Celery or threading to achieve similar concurrency
<br/>

## API endpoints

Single POST endpoint which accepts image files and returns a structured JSON response with detection results and explainibility data. <br/>
Example pseudocode:

```python
async def detect_image(file: UploadFile = File(...)) -> Dict:
  # 1: read image
  image = await file.read()
  # 2: preprocess image
  prepreprocessed_image = preprocess(image)
  # 3: run inference
  results = inference(proprocessed_image)
  # 4: generate explanibility outputs
  explanation = generate_explanation(preprocessed_image, results)
  # 5: return
  return JSONResponse({
    file,
    prediction,
    confidence,
    explanation
  })
```

FastAPI handles multipart file uploads via **UploadFile**

- can read the file into memory or temp save to disk for model processing
- preprocessing steps are applied before inference

The model outputs a label and a confidence score

- can include per sub-model confidence scores in the JSON response

We can generate visual heatmaps or saliency maps using:

- **Grad-CAM** or **Integrated Gradients** for CNN model
- Captum in Pytorch

## Model Optimization Techniques

### Quantization

> Quantization reduces the computational and memory costs of running inference by representing the weights and activations with low-precision data types like 8-bit integer (int8) instead of the usual 32-bit floating point (float32). (HuggingFace)

This enables huge memory savings for apparently small losses in performance.<br/>
[Dettmers (2022)](https://arxiv.org/abs/2212.09720) finds 4-bit precision almost universally optimal for total model bits and zero-shot accuracy

Post-training quantization (PTQ) can accumulate approximation errors, especially in the task of detailed image recognition.<br/>
A critical component is calibration data – a representative subset of the dataset used to:

1. Determine quantization parameters: Observes value distributions to select scale factors and zero points that minimize quantization error.
2. Mitigate approximation errors: Estimates the impact of reduced precision on outputs and adjusts parameters to preserve accuracy.

### Pruning

> Pruning reduces model size by removing less important neurons, involving identification, elimination, and optional fine-tuning

Involves three key phases: identification, elimination, and fine-tuning

1. **Identification**: Analytical review of the neural network to pinpoint weights and neurons with minimal impact on model performance
2. **Elimination**: Based on the identification phase, specific weights or neurons are removed from the model. This strategy systematically reduces network complexity, focusing on maintaining all but the essential computational pathways.
3. **Fine-tuning**: (optional but beneficial) Retrain the model’s reduced architecture to restore or enhance its task performance.

There are two main strategies for the identification and elimination phase:

- **Structured Pruning**: Remove entire groups of weights, such as a channel or layer, resulting in a leaner architecture. However, removing entire sub-components can significantly decrease its task performance.
- **Unstructured pruning**: Targets individual, less impactful weights across the neural network. This reduces the memory footprint but often doesn’t lead to speed improvements on standard hardware optimized for densely connected networks

### Knowledge Distillation

> Knowledge distillation transfers insights from a complex “teacher” model to a simpler “student” model, maintaining performance with less computational demand.

This approach is based on the idea that even though a complex, large model might be required to learn patterns in the data, a smaller model can encode the same relationship and reach a similar task performance.<br/>
Two key concepts:

- **Teacher-student architecture**:  teacher model is a high-capacity network with strong performance on the target task. The student model is smaller and computationally more efficient.
- **Distillation loss**: The student model is trained to replicate the output of the teacher **and** to match the output distributions produced.
  This allows it to learn the relationships between data samples and labels by the teacher, namely the location and orientation of the decision boundaries.
  <br/><br/>

### Comparison

| Technique | Pros | Cons | When to use |
| :-------- | :--- | :--- | :---------- |
| **Quantization** | Significantly reduces the model’s memory footprint while maintaining its full complexity. Accelerates computation. Enhances deployment flexibility | Possible degradation in task performance. Optimal performance may necessitate specific hardware acceleration support | Suitable for a wide range of hardware, though optimizations are best on compatible systems. Balancing model size and speed improvements. Deploying over networks with bandwidth constraints |
| **Pruning** | Reduces model size and complexity. Improves inference speed. Lowers energy consumption | Potential task performance loss. Can require iterative fine-tuning to maintain task performance | Best for extreme size and operation reduction in tight resource scenarios. Ideal for devices where minimal model size is crucial |
| **Knowledge distillation** | Maintains accuracy while compressing models. Boosts smaller models’ generalization from larger teacher models. Supports versatile and efficient model designs | Two models have to be trained. Challenges in identifying optimal teacher-student model pairs for knowledge transfer  | Preserving accuracy with compact models |

## GPU vs CPU deployment

### Overview

When deploying our deep learning model, the choice between CPU and GPU is dependent on the task at hand and based on factors such as throughput requirements and cost.

For deep learning training, **GPUs are preferred because they are much faster than CPUs**.
However, since inference is less resource-intensive, CPUs are often used for cost savings.
Still, if inference speed becomes a bottleneck, GPUs can offer major performance and cost-efficiency advantages.

### Core Differences

| Aspect | CPU Deployment | GPU Deployment |
| :---- | :-------------- | :------------- |
| **Processing Style** | Sequential, optimized for general-purpose tasks | Highly parallel, optimized for tensor and matrix computations |
| **Speed** | Slower for deep learning inference | Significantly faster for CNNs, transformers, and large models |
| **Scalability** | Easy to scale horizontally (many instances) | Scales vertically and benefits from batch parallelism |
| **Cost** | Lower per node but less throughput per dollar for large workloads | Higher per node cost but higher throughput for large models   |
| **Energy Use** | Generally lower | Higher due to power-hungry cores |
| **Best Use Cases** | Lightweight, low-throughput, or edge workloads | High-volume, real-time, or large model inference workloads |

<br/>
We can use CPUs if the model is smaller, quantized, or distilled. The overhead of setting up computation on the GPU eclipses the speedup of the calculations.  
For small networks with smaller number of parameters, the CPU can be a lot more efficient.
<br/><br/>

However, [Dan Grecoe’s testing](https://azure.microsoft.com/en-us/blog/gpus-vs-cpus-for-deployment-of-deep-learning-models/) shows that GPU clusters **consistently outperform** CPU clusters across all models and frameworks, making GPUs the more economical choice for deep learning inference. A 35‑pod CPU cluster was outperformed by <u>a single GPU cluster</u> by **186%** and by a <u>3‑node GPU cluster</u> by **415%** at similar cost. For smaller models like MobileNetV2 (TensorFlow), the single GPU node achieved **392%** and the 3‑node GPU **804%** higher throughput than the CPU cluster.

For traditional machine learning models with fewer parameters, CPUs remain more effective and cost-efficient. CPU performance can be optimized using libraries like **MKL-DNN** and **NNPACK**, while GPUs have similar tools such as **TensorRT**. Overall, GPU cluster performance tends to be more consistent than that of CPUs.

### Practical Deployment Considerations

- **Batching:** GPUs perform best with batched inference to fully utilize cores. CPUs are better for single, sporadic requests.
- **Cost Efficiency:** For constant high load, GPUs deliver better performance per dollar; for intermittent loads, CPUs may be more cost-effective.
- **Preprocessing Split:** Use CPU for I/O and image preprocessing, and GPU for model inference.
- **Hybrid Strategy:** Combine both — CPU nodes handle orchestration and preprocessing, GPU nodes handle inference.
- **Monitoring:** Track GPU utilization, latency, and memory; optimize batch size and concurrency settings.

### Example GPU Deployment in Kubernetes

Based on Microsoft’s [GPU deployment guide](https://learn.microsoft.com/en-us/archive/blogs/machinelearning/deploying-deep-learning-models-on-kubernetes-with-gpus) and [az-deep-realtime-score](https://github.com/microsoft/az-deep-realtime-score):

1. **Model Development**: Train and export the deep learning model used for inference.
2. **API Module**: Build a REST API (Flask or FastAPI) that loads the model on startup and handles prediction requests.
3. **Containerization**: Package the API and model in a Docker image using a CUDA base image (Nvidia/CUDA) with Flask and Nginx for serving.
4. **Local Testing**: Run and validate the container locally to ensure model loading and inference work correctly.
5. **AKS Cluster Setup**: Create an Azure Kubernetes Service (AKS) cluster with GPU-enabled nodes (e.g., NC-series VMs) and install NVIDIA drivers.
6. **Deployment to AKS**: Push the Docker image to Azure Container Registry and deploy to the GPU node pool using Kubernetes manifests with:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

7. **Testing & Benchmarking**: Access the hosted web app endpoint, verify inference results, and measure latency and throughput between CPU and GPU deployments.

## Batch processing

### Overview

Batch Processing is a technique of using batches to process large volumes of data. The approach splits the data into batches and performs a sequence of unified jobs of consecutively training the model on one batch after another.

### Why train in batches?

We can either feed all the data to the model at once **or** feed some data, wait until algorithm processes, feed another model another part of data.

Both approaches are viable, but by feeding all the data at once, one must simultaneously store every single data asset in the machine's memory. Also, you must store all derivative values associated with the data, such as loss values, processing details, etc. Moreover, you can **only** update the model's weights only after whole dataset is pocessed. 

Servless platforms excel at parallel execution, but without breaking tasks into smaller units, these benefits are lost. Additionally, the economics of serverless computing introduce a performance-cost trade-off. While dividing tasks into smaller batches can reduce execution time, triggering numerous functions can increase costs due to the pay-per-invocation pricing model. 

**Benefits of batch processing:**
- efficient memory utilization
- improvements in training speed (especially when running jobs in parallel on GPUs)
- regular updates of the weights (after each batch is processed)
- introduction of noise into the training process, brining the regularization effect and improve the model's generalization. 

### Performance Metrics

According to [Barrack and Ksontini](https://arxiv.org/html/2502.12017v1), monolithic batch processing maintains relatively stable cost and runtime across different batch sizes, showing only minor decreases as batch size grows from reduced invocation overhead when larger batches are processed sequentially.

In contrast, parallel batch processing demonstrates a highly dynamic behavior. Smaller batch sizes enable massive parallelism, cutting execution time by over **95%** compared to monolithic execution, albeit at a slightly higher initial cost due to many concurrent invocations. As batch sizes increase, costs stabilize while runtime remains significantly lower, typically completing tasks within minutes rather than hours.

Overall, parallel batch processing achieves superior scalability and time efficiency for ML inference tasks. Both approaches use similar memory footprints since inference workloads mainly load model parameters into RAM, but parallelism leverages concurrency to minimize runtime without major cost increases.

## Caching strategies
### Model-Level Caching
#### Model Warm Load
By keeping the model loaded persistently in memory instead of reloading it per request, we can significantly reduce first-request latency.<br/>
On startup, preload the model to GPU and perform a **dummy forward pass** to allocate memory
#### Inference Cache
If users often upload the same or near-identical images, we can cache the ouput to eliminate redundant GPU inference. This can become especially powerful when processing crowdsourced or reused datasets.
- Use a hash of the file bytes as the cache key
- Store JSON results in Redis or local disk
- If hash already exists, return cached results instantly.
### Data Pipeline Caching
#### Preprocessing Cache
Preprocessing can be expensive and repetitive for similar inputs. To avoid this, we can cache preprocessed tensors keyed by the image hash with Redis or DiskCache. This would significantly reduce computation if we re-run inference with slightly different params. 
#### Intermediate Feature Cache
We can cache intermediate results generated during inference. This is especially useful since our model will have multiple stages, for example: face detection followed by feature extraction and classification. 

This cache can be implemented using Redis for smaller intermediate tensors or cloud storage (e.g., Azure Blob, S3) for large feature maps. Keys can be constructed using hashes of the input data and the model stage identifier to ensure uniqueness.
### System-Level Caching
#### GPU Memory Persistence
We can keep commonly accessed tensors or model layers on the GPU to prevent data transfer overhead between CPU and GPU. If multiple models share the GPU, we can use memory pinning or preallocation to reduce reallocation costs.
### Web/API Layer Caching
#### HTTP Response Caching
If our FastAPI endpoint serves public requests, we can use HTTP cache headers
```python
return JSONResponse(result, headers={"Cache-Control": "public, max-age=3600"})
```
CDN caching (with Azure Front Door, Cloudflare) can store common responses closer to uesrs.
#### Reverse Proxy Cache
Depending on what we use to deploy with, we may be able to enable reverse-proxy caching for identical POST payloads. This approach reduces repeated inference for the same uploads and is especially helpful for frontend testing environments.
### Embedding Similarity Cache
For near-duplicate image/video detection, we can also store face embeddings or frame features in a vector database (FAISS, Pinecone, or Milvus). If a new upload is similar to an existed cached embedding (cosine similarity > threshold), reuse the cached prediction.

## Load balancing

### Overview
Load balancing distributes incoming traffic and inference requests across multiple replicas or nodes to ensure high availability, scalability, and optimal performance of the DeepFake Detection backend. It prevents any single instance from being overloaded and allows for horizontal scaling of the API and model-serving components.

### Types

#### 1. Round-Robin
Round-robin load balancing is a simple and widely used strategy.  Each incoming request is directed to the next available instance in a predetermined sequence. This ensures even distribution under uniform load.
- **Pros:** Simple, fast, easy to implement
- **Cons:** Doesn’t account for instance utilization or latency

#### 2. Least Connections
Least connection load balancing directs incoming requests to the instance with the fewest active connections. This dynamically adjusts to uneven workloads and is suited for variable traffic. 
- **Pros:** Adaptive to instance load
- **Cons:** Still does not account for instance utilization or latency

#### 3. IP Hash Load Balancing
IP hash load balancing directs incoming requests to an instance based on the client's IP address. This strategy is useful for maintaining session persistence but may not be suitable for serverless ML inference, as it does not account for instance utilization or latency.
- **Pros:** Useful for stateful requests (e.g., video chunk analysis)
- **Cons:** Can create uneven load distribution if few clients dominate traffic

#### 4. Queue-Based Load Balancing
Queue-based load balancing uses a message queue to buffer incoming requests. This strategy is well-suited for serverless ML inference, as it decouples the request processing from the instance management.
- **Pros:** Absorbs traffic spikes smoothly, supports asynchronous processing,  enables scalable serverless architectures.
- **Cons:** Introduces latency (since requests wait in queue), requires monitoring of queue depth and scaling logic, unsuitable for real-time inference.

### Load Balancing in Deployment

#### a. Application Level (API)
Use **FastAPI’s Uvicorn workers** or **Gunicorn** with multiple worker processes to handle concurrent requests:
At the application layer, load balancing ensrues that incoming HTTP requests are distributed evenly across multiple API worker processes or container replicas. 

**Example setup**

Use Uvicorn or Gunicorn with multiple workers:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```
This means that each worker handles concurrent requests using FastAPI's async model.

In front of the application containers, a reverse proxy (like NGINX or Azure Application Gateway) distributes traffic across multiple replicas and handles connection pooling, retries, and rate limiting.

#### b. Kubernetes Level (Cluster)
In Kubernetes, load balancing happens automatically via **Services** and **Ingress** automatically perform round-robin load balancing across pods. Combine this with:
- **Kubernetes Service**: Performs round-robin load balancing across pods in deployment. Each pod runs one instance of the FastAPI container.
- **Horizontal Pod Autoscaler (HPA):** scales the number of pods based on CPU/GPU/memory and queue length metrics.
- **Cluster Autoscaler:** scales nodes when GPU or CPU capacity is insufficient.
- **Ingress Controller (NGINX, Traefik):** handles external traffic routing, HTTPS/TLS termination, can integrate with CDN caching.

#### c. Model-Serving Level (Inference)
At the model-serving layer, **TorchServe** manages inference, batching, and scaling for PyTorch models. Each instance can serve multiple models or multiple workers per model to fully utilize GPU capacity.
- **Dynamic Batching**: Combines requests to maximize GPU throughput and reduce latency.
- **Autoscaling**: Integrate with HPA to scale replicas based on GPU utilization, latency, or queue length.
- Torchserve provides easy horizontal scaling and monitoring and high GPU utilization with low latency. 


## Containerization
This involves packaging the entire workflow into containers to ensure consistency across development, testing, and production. Different types of containers can be used for specific tasks in the project lifecycle, such as training, inference, batch prediction, and hybrid containers. 
### Container Types
#### Training-only Containers

1. **Package the training script**: This script contains all the necessary steps to train the model (data loading, preprocessing, training logic, hyperparameters tuning).
2. **Define dependencies**: List all required libraries (TensorFlow, PyTorch, Scikit-learn, etc.) in a requirements.txt or environment.yml file.
3. **Create Dockerfile**: Define the environment for the training process, including base image, libraries, and commands to run the training script.
4. **Build and run the container**:

Example DOCKERFILE:
```docker
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
CMD ["python", "train.py"]
```
We can scale these containers horizontally using Kubernetes to train the model with distributed datasets or hyperparameter tuning.

#### Inference-only Containers

1. **Deploy trained model**: Use the model artifact from the training container.
2. **Expose a web service**: Use a framework like Flask or FastAPI to serve the model as an API endpoint.
3. **Create Dockerfile**: Define the necessary environment and the web service that loads the trained model and exposes an API for inference.
4. **Build and deploy the container**: Build the container and expose the prediction service.

Example DOCKERFILE:
```docker
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
CMD ["python", "app.py"]
```
For the API service, app.py can be a FastAPI service that loads the model and exposes `/predict` for requests.<br/>
Use load balancing for high-volume inference services by deploying multiple inference containers for scaling.

### High-level approach
- FastAPI acts as the gateway for pre/postprocessing, caching and auth. 
- The inference code is run inside the container (Torchscript)
- One model replica per GPU (or one container/GPU) for straightforward GPU resource control
- Use a lightweight model image (no dev tools) for runtime; build heavy tools in a multi-stage build
### Image & build best practices
- **Multi-stage Docker build**: build and compile dependencies in one stage, copy only runtime artifacts into a **slim** runtime image
- **Pin dependency versions** in requirements.txt and install wheels to avoid recompiling at runtime. Build wheels in builder to avoid compiling in runtime
- Create **non-root user** in container for security
- For CPU-only deploys use python:3.x-slim. 


## Other Notes

- We can expose Prometheus metrics for a lot of information, including request count, latency, GPU utilization, batch sizes, etc. These metrics can then be used to drive the autoscaling and optimize throughput. 
- Definately need to automate CI/CD for builds, tests, and model updates. 