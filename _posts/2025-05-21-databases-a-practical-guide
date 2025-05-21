---
layout: post
title: "Databases: A Practical Guide"
categories: blog
permalink: /databases-a-practical-guide/
---

By: Nikita Pavlichenko

AI engineering often involves handling various types of data, from structured records to unstructured embeddings. This guide introduces four key database types – [**relational**](https://en.wikipedia.org/wiki/Relational_database), [**in-memory**](https://en.wikipedia.org/wiki/In-memory_database), [**NoSQL**](https://en.wikipedia.org/wiki/NoSQL), and [**vector**](https://en.wikipedia.org/wiki/Vector_database) – and provides hands-on examples for each. We assume you have a software engineering background but little database experience. Each section includes an overview of the database type, why it’s relevant to AI/ML workloads, step-by-step setup instructions (using [Docker](https://www.docker.com/) for convenience), and examples of querying the database using both native tools and Python code. The goal is to get you comfortable with basic database concepts and operations in a **beginner-friendly** way.

## I'm not a backend developer. Why should I care?

Databases are fundamental to all areas of software engineering. Even if your model pipeline doesn’t directly interact with a database (such as in [RAG](https://www.pinecone.io/learn/retrieval-augmented-generation/) queries), your *application* likely relies on multiple types of databases for caching, logging, authorization, analytics, and more. Understanding how to work with them allows you to see the bigger picture. If you aim to be a “full-stack ML engineer" responsible for the entire service, database expertise will be essential.

## What We Expect from Databases

At their core, databases are specialized software designed to store, retrieve, and manage data efficiently. Regardless of the specific type or technology, several key capabilities are universally desired from a database:

- **Data Persistence and Durability**: Ensuring that once data is stored, it remains available even after system restarts, crashes, or failures.
- **Efficient Retrieval**: Quickly accessing data based on various criteria, making it practical for real-time applications and analysis.
- **Consistency and Integrity**: Guaranteeing that the data remains accurate, valid, and consistent throughout its lifecycle.
- **Concurrency and Isolation**: Allowing multiple users or applications to interact with the database simultaneously without interference or corruption of data.
- **Scalability**: Supporting increased workloads or storage requirements without compromising performance.
- **Security**: Protecting data against unauthorized access, ensuring privacy, and maintaining compliance with data regulations.

Different databases provide these features in various ways. For instance, relational databases offer [**ACID**](https://en.wikipedia.org/wiki/ACID) (Atomicity, Consistency, Isolation, Durability) properties, crucial for transactional applications like financial systems, while NoSQL databases often prioritize scalability and flexibility over strict consistency.

### The Database Interface and Architecture

Most modern databases operate as standalone microservices—essentially specialized web applications running independently from your application logic. They typically expose interfaces over network protocols like [**TCP/IP**](https://en.wikipedia.org/wiki/Transmission_Control_Protocol), allowing applications to interact with them through standardized communication protocols. This architectural separation provides numerous benefits:

- **Decoupling and Modularity**: Databases can be managed, scaled, or replaced independently from the application logic.
- **Interoperability**: Applications written in any programming language can communicate with the database using standardized drivers or APIs.
- **Scalability and Reliability**: Databases can be distributed across multiple servers, improving fault tolerance, redundancy, and performance.

To interact with a database, applications establish a connection (often through a client library) and communicate using query languages (like SQL), commands, or APIs specific to the database type. The interaction typically involves:

1. **Connection Establishment**: Setting up a reliable TCP connection from the application to the database service.
2. **Sending Queries/Commands**: Issuing commands or queries to perform operations like reading, writing, updating, or deleting data.
3. **Processing Responses**: Receiving results, status messages, or errors from the database service.

Understanding this interface helps developers effectively integrate databases into software systems, ensuring optimal performance, reliability, and scalability.


## Relational Databases

Relational databases have transformed how we manage data since their inception in the 1970s, when [Edgar F. Codd](https://en.wikipedia.org/wiki/Edgar_F._Codd) introduced the relational model—a revolutionary concept where data is organized into structured tables, defined by rows and columns. These tables are linked through relationships (keys), ensuring data consistency and enabling sophisticated queries using **Structured Query Language (SQL)**. This logical structure quickly became the gold standard for handling structured data reliably and efficiently.

### Why PostgreSQL Stands Out

While several relational databases emerged over the decades—[Oracle](https://www.oracle.com/database/), [MySQL](https://www.mysql.com/), [SQL Server](https://learn.microsoft.com/en-us/sql/sql-server/)—[PostgreSQL](https://www.postgresql.org/) (commonly known as **Postgres**) has distinguished itself as the preferred choice in industry and academia alike. Originally developed at UC Berkeley in the 1980s as a follow-up to the pioneering Ingres project, Postgres combines rigorous standards compliance with extraordinary extensibility. It supports robust **ACID** properties (Atomicity, Consistency, Isolation, Durability), ensuring transactions are executed reliably and data integrity is always maintained.

In the landscape of databases today, many alternatives have faded into niche roles or obscurity because PostgreSQL offers unmatched versatility—supporting not only traditional relational data but also advanced features like JSON storage, GIS capabilities, and powerful analytics extensions. Its open-source nature further solidifies its dominance by enabling continuous improvement and customization, backed by a dedicated global community.

Maybe PostgreSQL is the only database [you ever need](https://www.youtube.com/watch?v=3JW732GrMdg).

### Setting up PostgreSQL with Docker

Let's practically explore how simple it is to set up Postgres using Docker, which ensures a consistent environment without the overhead of local installations.

1. **Download the Docker Image:** Pull the latest official PostgreSQL image:

```bash
docker pull postgres:latest
```

2. **Launch Your Database:** Create and run a Docker container, set a password, and expose the default port (5432):

```bash
docker run --name my-postgres \
    -e POSTGRES_PASSWORD=mysecretpassword \
    -p 5432:5432 \
    -d postgres:latest
```

This starts PostgreSQL in a containerized environment, accessible from your host system.

3. **Ensure It's Running:** Check the logs or running containers:

```bash
docker logs my-postgres
# or
docker ps
```

You should see a confirmation that Postgres is ready for connections.

4. **Connect to Postgres:** Access your new database using `psql`:

```bash
psql -h localhost -U postgres -p 5432 postgres
```

Or use the Docker container directly:

```bash
docker exec -it my-postgres psql -U postgres
```

### Basic PostgreSQL Configuration Tips

For a beginner, PostgreSQL's default configuration in Docker works great for development purposes. Key points to remember:

- **Authentication:** Password authentication is default (`POSTGRES_PASSWORD`). For production-grade security, you might further refine authentication via [`pg_hba.conf`](https://www.postgresql.org/docs/current/auth-pg-hba-conf.html).

- **Persistent Data:** To retain data when your container restarts, mount a Docker volume:

```bash
docker run --name my-postgres \
    -e POSTGRES_PASSWORD=mysecretpassword \
    -v postgres-data:/var/lib/postgresql/data \
    -p 5432:5432 \
    -d postgres:latest
```

- **Performance Tuning:** Parameters like `shared_buffers` and `work_mem` in [`postgresql.conf`](https://www.postgresql.org/docs/current/runtime-config-resource.html) can optimize performance, though defaults typically suffice initially.

### Querying PostgreSQL with SQL and Python

Now, let's see Postgres in action, first through SQL directly in `psql`, then programmatically from Python.

**Using SQL Interactively:**

Within your `psql` shell, you might run:

```sql
CREATE TABLE items (id SERIAL PRIMARY KEY, name TEXT, value INTEGER);
INSERT INTO items (name, value) VALUES ('foo', 42), ('bar', 84);
SELECT * FROM items;
```

This demonstrates how easily PostgreSQL handles schema creation, data insertion, and retrieval:

```
id | name | value
---+------+-------
 1 | foo  |    42
 2 | bar  |    84
(2 rows)
```

**Using PostgreSQL from Python:**

With Python's [`psycopg2`](https://www.psycopg.org/docs/) library, connecting and interacting with your database is straightforward:

```python
!pip install psycopg2-binary
```

```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="postgres",
    user="postgres",
    password="mysecretpassword"
)
cur = conn.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS items (id SERIAL PRIMARY KEY, name TEXT, value INTEGER);")
cur.execute("INSERT INTO items (name, value) VALUES (%s, %s)", ("baz", 100))
cur.execute("INSERT INTO items (name, value) VALUES (%s, %s)", ("qux", 200))
conn.commit()

cur.execute("SELECT * FROM items;")
rows = cur.fetchall()
print(rows)

cur.close()
conn.close()
```

Running this will neatly return your data, including all previously added rows.


## In-Memory Databases: Redis and Its Evolving Landscape

In-memory databases store data primarily in RAM, dramatically reducing latency and offering exceptional speed in read and write operations compared to traditional disk-based databases. One of the most widely adopted examples is [**Redis**](https://redis.io/), an open-source in-memory key-value database initially released in 2009 by Salvatore Sanfilippo. Redis rapidly gained popularity due to its simplicity, performance, and versatility.

Redis stands out due to its diverse set of supported data structures, including strings, hashes, lists, sets, sorted sets, bitmaps, hyperloglogs, and geospatial indices. This makes Redis highly flexible and suitable for various scenarios like caching, real-time analytics, session management, and even message brokering. The speed of Redis operations—typically sub-millisecond—positions it perfectly for scenarios demanding low latency, such as real-time AI applications and high-throughput environments.

### Why Choose an In-Memory Database?

The main benefit of using an in-memory database is speed—essential for quickly accessing or updating small datasets. In-memory databases like Redis excel at caching frequently requested data, session information, temporary user profiles, and transient computed values, all of which require rapid retrieval without the need for long-term persistence.

You may ask: why not just use a Python `dict`? While a Python `dict` can serve as a simple, in-process cache, it is limited by its lack of scalability and advanced features. Unlike a `dict`, Redis supports multiple client connections, offers built-in persistence and replication, provides atomic operations, and seamlessly handles complex data structures. These capabilities ensure data consistency and high availability in distributed environments—features a basic Python `dict` cannot provide. Additionally, Redis is written in C, making it faster than Python with [Uvicorn](https://www.uvicorn.org/) (since external access is required).

However, in-memory storage isn’t ideal for large historical datasets due to higher costs and limited capacity compared to traditional disk-based databases. Redis is best used alongside persistent storage solutions, holding only frequently accessed ([“hot”](https://www.dremio.com/wiki/hot-data/)) data that benefits from immediate availability.

### Quick Redis Setup with Docker

Docker simplifies getting started with Redis significantly:

1. **Pull the Redis image:**
   ```bash
   docker pull redis:latest
   ```

2. **Launch a Redis container:**
   ```bash
   docker run --name my-redis -p 6379:6379 -d redis:latest
   ```

This runs Redis in the background, mapping port 6379 on your local machine to the container.

3. **Verify Redis is running:**
   Check logs with `docker logs my-redis`. You should see `* Ready to accept connections`. Alternatively, `docker ps` will confirm the container’s status.

4. **Interact via Redis CLI (optional):**
   Connect quickly using the Redis CLI:
   ```bash
   docker exec -it my-redis redis-cli
   ```

Try basic commands to test your setup:

```
127.0.0.1:6379> SET mykey "Hello, Redis!"
OK
127.0.0.1:6379> GET mykey
"Hello, Redis!"
```

### Essential Redis Configuration Tips

Redis runs smoothly without extensive configuration, but here are key aspects to keep in mind:

- **Data Persistence:** Redis can persist data to disk through snapshotting (RDB) or append-only files (AOF). Docker's default setup uses snapshotting, sufficient for many applications. Disable persistence for maximum speed if your data can be recomputed easily, or enable persistence for reliability across restarts.

- **Memory Management:** Redis will use available system memory by default. For robust, scalable deployments—particularly in AI and ML environments—configure memory limits (`maxmemory`) and eviction policies (`allkeys-lru`) to avoid unexpected issues.

- **Security Considerations:** Default Redis setups lack authentication and listen on all network interfaces. For local development within Docker, this is safe. However, production deployments must limit network exposure, bind to localhost or private networks, and ideally set authentication passwords ([`requirepass`](https://stackoverflow.com/questions/7537905/how-to-set-password-for-redis)).

### Redis with Python: Practical Usage

Python integration with Redis is straightforward using the [`redis-py`](https://redis-py.readthedocs.io/en/stable/) library:

Install the library:
```bash
pip install redis
```

Example code snippet:

```python
import redis

# Connect to Redis
r = redis.Redis(host="localhost", port=6379, db=0)

# Set and get a simple key-value pair
r.set("framework", "TensorFlow")
value = r.get("framework")

print(value)           # b'TensorFlow'
print(value.decode())  # TensorFlow
```

Redis can also store structured data efficiently:

```python
r.hset("model:1", mapping={"name": "ResNet50", "accuracy": "0.95"})
model_info = r.hgetall("model:1")

print({k.decode(): v.decode() for k, v in model_info.items()})
```

Such structures make Redis invaluable for caching results, maintaining metrics, or storing real-time information in AI/ML applications.

### Redis Licensing Controversy

Historically, Redis was fully open-source, distributed under a permissive [BSD license](https://en.wikipedia.org/wiki/BSD_licenses). However, in 2024, Redis Labs shifted to a "source-available" licensing model, still allowing free usage but imposing certain limitations for commercial scenarios. This shift spurred the open-source community into action, giving rise to several important Redis forks aiming to preserve the unrestricted open-source ethos. [Here](https://www.youtube.com/watch?v=o4QEwAqV0BQ) is a great video from Primagen if you want to be familiar with it.

Among these forks, [**ValkeyDB**](https://valkey.io/) (commonly known as Valkey) stands out. Launched by the Linux Foundation in direct response to Redis's licensing change, ValkeyDB maintains full compatibility with Redis commands and protocols. It promises a community-driven approach and a fully open-source future, providing a straightforward migration path for Redis users who value unrestricted licensing.

Another notable Redis fork is [**KeyDB**](https://www.keydb.dev/), developed by Snapchat with a unique emphasis on multithreading capabilities. Unlike Redis, which traditionally uses a single-threaded event loop, KeyDB can leverage multiple CPU cores effectively. This means significantly higher throughput and better performance on multi-core hardware, all while remaining fully compatible as a drop-in replacement for Redis.

For educational and developmental purposes, Redis itself remains a strong choice—especially earlier open-source versions such as [Redis 6.x](https://github.com/redis/redis/releases/tag/6.2.13), still under the original BSD license. However, knowing alternatives like ValkeyDB and KeyDB provides flexibility in deployment choices based on licensing needs and performance considerations.

Redis’s licensing shift underscores the importance of alternatives. **ValkeyDB** offers open-source flexibility, while **KeyDB** provides enhanced performance through multithreading. Both retain Redis compatibility, ensuring minimal friction when migrating existing Redis workloads.

Exploring these alternatives allows choosing the best fit based on your project's performance, licensing, and scalability needs.


### Performance Comparison with PostgreSQL

Here's a benchmark comparison between Redis and PostgreSQL:

**Write Operation Performance:**

| Number of Records | Redis (v3.0.7) | PostgreSQL (v13.3) |
|-------------------|----------------|--------------------|
| 1,000             | 34 ms          | 29.6 ms            |
| 10,000            | 214 ms         | 304 ms             |
| 100,000           | 1,666 ms       | 2,888 ms           |
| 1,000,000         | 14,638 ms      | 31,230 ms          |

**Read Operation Performance:**

| Number of Records | Redis (v3.0.7) | PostgreSQL (v13.3) |
|-------------------|----------------|--------------------|
| 1,000             | 8 ms           | 0.026 ms           |
| 10,000            | 6 ms           | 0.028 ms           |
| 100,000           | 8 ms           | 0.027 ms           |
| 1,000,000         | 8 ms           | 0.029 ms           |

*Note: The above data is sourced from a performance comparison conducted by [Cybertec PostgreSQL Consulting](https://www.cybertec-postgresql.com/en/postgresql-vs-redis-vs-memcached-performance/).



## NoSQL Databases: A Deeper Look at MongoDB

The term **NoSQL** ("Not Only SQL") encompasses a broad category of database systems designed to handle large volumes of diverse, unstructured, or semi-structured data—something traditional relational databases struggle with. Among these, **document databases** are particularly popular, with [**MongoDB**](https://www.mongodb.com/) leading the way. Unlike SQL databases, which enforce rigid table schemas, document databases store information as flexible, JSON-like documents (specifically called [**BSON**](https://www.mongodb.com/docs/manual/reference/bson-types/) in MongoDB), allowing each record to have a distinct structure. These documents are stored in **collections**, analogous to tables, but each document can differ greatly in content, reflecting the dynamic and evolving nature of modern data, especially relevant in AI and machine learning scenarios.

MongoDB, first developed by 10gen in 2007, quickly became the leading open-source NoSQL document store thanks to its ease of use, scalability, and strong community support. It excels at managing vast amounts of data distributed across multiple servers (a process called [sharding](https://www.mongodb.com/docs/manual/sharding/)), making it well-suited for handling big data workloads in real-time environments such as analytics pipelines and AI model training. For example, AI engineers frequently use MongoDB to store experiment configurations, hyperparameters, training logs, or raw JSON responses from various APIs without needing constant schema migrations. The ability to rapidly iterate and adapt the schema as the application evolves is crucial in fast-paced research environments.

### Setting up MongoDB Using Docker

Docker simplifies the setup and management of MongoDB. Here’s how to quickly launch a MongoDB server:

1. **Download the Official MongoDB Docker Image:**

   ```bash
   docker pull mongo:latest
   ```
   This fetches the latest stable MongoDB version directly from Docker Hub.

2. **Run the MongoDB Container:**

   Launch MongoDB and expose its default port (27017):

   ```bash
   docker run --name my-mongo -p 27017:27017 -d mongo:latest
   ```
   This command starts a MongoDB instance in the background, making it accessible on your local machine at port 27017. It will store data in `/data/db` within the container. Keep in mind, if the container is removed, data stored without volume mapping will be lost.

3. **Check the MongoDB Container:**

   To confirm MongoDB is operational:

   ```bash
   docker logs my-mongo
   ```
   You should see "Waiting for connections," indicating the server is ready.

4. **Interact via MongoDB Shell (Optional):**

   If you've installed `mongosh` locally, connect with:

   ```bash
   mongosh "mongodb://localhost:27017"
   ```
   Alternatively, enter the container directly:

   ```bash
   docker exec -it my-mongo mongosh
   ```
   Test MongoDB with:

   ```javascript
   > db.test.insertOne({message: "Hello, MongoDB!"})
   > db.test.find()
   ```
   MongoDB automatically creates a `test` collection and assigns a unique ObjectId (`_id`) to your inserted document.

### Important MongoDB Configuration Considerations

In development setups, MongoDB typically runs without authentication for convenience. However, this is insecure for production use. For secure deployments, set `MONGO_INITDB_ROOT_USERNAME` and `MONGO_INITDB_ROOT_PASSWORD` environment variables to establish an admin user, or leverage cloud-based services like [MongoDB Atlas](https://www.mongodb.com/atlas), which provide secure, managed environments.

To persist data beyond the container's lifecycle, always map a volume:

```bash
docker run --name my-mongo -p 27017:27017 -v mongo-data:/data/db -d mongo:latest
```

MongoDB uses WiredTiger as its storage engine, allocating up to 50% of available RAM by default for caching. Monitor resource usage carefully, especially on shared development machines.

### Integrating MongoDB with Python

The **PyMongo** driver simplifies working with MongoDB in Python:

Install [PyMongo](https://pymongo.readthedocs.io/en/stable/):

```python
!pip install pymongo
```

Connect and interact with MongoDB:

```python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client.ai_db
collection = db.experiments

experiment = {
    "experiment_id": 1,
    "model": "RandomForestClassifier",
    "parameters": {"n_trees": 100, "max_depth": 5},
    "accuracy": 0.87,
    "timestamp": "2025-03-07T20:00:00"
}

result = collection.insert_one(experiment)
print(f"Inserted document ID: {result.inserted_id}")
```

MongoDB handles document creation flexibly, automatically generating a unique [ObjectId](https://www.mongodb.com/docs/manual/reference/method/ObjectId/) (`_id`) unless explicitly defined. Retrieve documents easily:

```python
doc = collection.find_one({"experiment_id": 1})
print(doc)
```

### Querying and Handling Diverse Data

MongoDB excels at handling diverse document structures. For example, AI experiments might differ significantly in their recorded metrics and configurations:

```python
collection.insert_many([
    {"experiment_id": 2, "model": "SVM", "kernel": "rbf", "accuracy": 0.78},
    {"experiment_id": 3, "model": "NeuralNet", "layers": [64, 64, 10], "accuracy": 0.92}
])

for exp in collection.find({"accuracy": {"$gt": 0.8}}):
    print(exp)
```

This query retrieves all experiments with accuracy above 0.8 using MongoDB's powerful and intuitive JSON-based query syntax.

### Why Choose MongoDB for AI Applications?

AI applications frequently face challenges with evolving data schemas, rapidly changing inputs, and diverse metadata formats. MongoDB's schema flexibility allows AI engineers to quickly adapt without constant schema migrations. For instance, IoT-based AI systems or dynamic logging services frequently evolve their data formats over time. MongoDB's [horizontal scalability](https://www.mongodb.com/docs/manual/sharding/) is invaluable when managing large-scale data ingestion pipelines or rapidly expanding datasets.

In practice, MongoDB is commonly used alongside relational databases—MongoDB manages unstructured or semi-structured data, while relational databases handle transactions, structured data, or complex joins. This hybrid approach leverages the strengths of both systems, delivering both flexibility and consistency essential for modern AI-driven applications.


## Vector Databases (Qdrant)

Vector databases represent an emerging class of databases specifically optimized for performing **similarity searches** on high-dimensional vector data—a data format extensively utilized in modern AI and machine learning applications. Historically, databases like relational databases (e.g., MySQL, PostgreSQL) were excellent for structured data but struggled with tasks involving semantic relationships and approximate similarity searches due to their linear scan inefficiencies.

Modern AI techniques often transform unstructured data types—such as text, images, audio, and video—into numerical representations called embeddings. These embeddings are high-dimensional vectors generated by sophisticated neural network models, such as transformers ([BERT](https://huggingface.co/bert-base-uncased), [GPT](https://platform.openai.com/docs/guides/embeddings)) or convolutional networks ([ResNet](https://arxiv.org/abs/1512.03385), [EfficientNet](https://arxiv.org/abs/1905.11946)). Once data is converted into this vector form, finding similar items involves computing distances between vectors, a task traditional databases aren't optimized for. Vector databases address this gap by employing specialized data structures and algorithms, such as [**Hierarchical Navigable Small World (HNSW)**](https://github.com/nmslib/hnswlib) graphs or inverted file (IVF) indices, enabling fast approximate nearest-neighbor (ANN) searches even among millions or billions of vectors.

Vector databases are crucial for applications including image and audio similarity searches, semantic document retrieval, recommendation systems, and content-driven search engines. For example, imagine an online marketplace wanting to display visually similar products; by representing each product image as a vector, the marketplace can instantly retrieve the most visually similar products using a vector database, something nearly impossible in real-time with traditional databases.

### Introducing Qdrant

One prominent and modern example of a vector database is [**Qdrant**](https://qdrant.tech/), a high-performance, open-source vector search engine designed specifically for AI workloads. Originating as a response to the increased demand for scalable AI-driven similarity search, Qdrant is built entirely in Rust, providing both memory efficiency and excellent runtime performance. It supports multiple similarity metrics, including cosine similarity, dot product, and Euclidean distances, allowing flexibility based on specific application needs.

Qdrant stores high-dimensional vectors along with associated metadata (called "payloads"), enabling rich, filtered search capabilities. Its architecture is designed for ease of integration into AI pipelines, offering a straightforward REST API and gRPC interface, along with official client libraries for various programming languages, prominently Python.

### Setting Up Qdrant with Docker

Deploying Qdrant can be straightforward using Docker containers:

1. **Pulling the Qdrant Docker image:**

   ```bash
   docker pull qdrant/qdrant
   ```

   This command fetches the latest, ready-to-use Qdrant server image.

2. **Running Qdrant Container:**

   Execute the container, mapping essential ports:

   ```bash
   docker run --name my-qdrant -p 6333:6333 -p 6334:6334 \
       -v "$(pwd)/qdrant_storage:/qdrant/storage" \
       -d qdrant/qdrant
   ```

   Breaking down the command:
   - `-p 6333:6333` exposes the REST API port, while `-p 6334:6334` exposes the gRPC API.
   - The volume mapping (`-v`) mounts a local directory (`qdrant_storage`) to Qdrant’s internal storage, ensuring vector data persistence across container restarts—a critical consideration for real-world use.
   - `-d` runs the container in detached mode, operating seamlessly in the background.

   After execution, Qdrant becomes accessible at `http://localhost:6333`, complete with a user-friendly web UI at `http://localhost:6333/dashboard`. By default, Qdrant has no authentication or encryption enabled, suitable for development but requiring proper securing before deployment in production environments.

3. **Confirming Qdrant’s Status:**

   To verify successful deployment, navigate to `http://localhost:6333/dashboard` in your browser or inspect Docker logs:

   ```bash
   docker logs my-qdrant
   ```

   You should see logs indicating Qdrant is operational and actively listening on ports 6333 (REST) and 6334 (gRPC).

### Working with Vectors in Qdrant

Here how you can access your running Qdrant database [from Python](https://github.com/qdrant/qdrant-client):

```python
!pip install qdrant-client
```

Qdrant organizes data into **collections**, analogous to tables in traditional databases, each configured with a specified vector dimension and similarity metric.

**Creating a collection:**

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")

client.recreate_collection(
    collection_name="my_vectors",
    vectors_config=VectorParams(size=4, distance=Distance.DOT)
)
```

This script connects to Qdrant and initializes a collection named `my_vectors` with vectors of dimension 4, using dot product similarity.

**Inserting Vectors:**

```python
from qdrant_client.models import PointStruct

points = [
    PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload={"label": "A"}),
    PointStruct(id=2, vector=[0.0, 0.1, 0.0, 0.9], payload={"label": "B"}),
    PointStruct(id=3, vector=[0.2, 0.2, 0.2, 0.2], payload={"label": "A"}),
    PointStruct(id=4, vector=[0.9, 0.1, 0.1, 0.0], payload={"label": "C"})
]

client.upsert(collection_name="my_vectors", points=points)
```

Vectors are accompanied by metadata, enabling advanced filtering during searches. Confirm insertion:

```python
count = client.count(collection_name="my_vectors", exact=True)
print(count)  # Expected output: 4
```

**Performing Similarity Search:**

Given a new query vector, Qdrant swiftly retrieves similar vectors:

```python
query_vector = [0.1, 0.2, 0.4, 0.3]
results = client.search(collection_name="my_vectors", query_vector=query_vector, limit=2)

for res in results:
    print(f"Found ID={res.id} with score={res.score}, payload={res.payload}")
```

Additionally, filtered searches are possible:

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="my_vectors",
    query_vector=query_vector,
    query_filter=Filter(must=[FieldCondition(key="label", match=MatchValue(value="B"))]),
    limit=1
)

print(results)
```

### Real-World AI Applications

Vector databases like Qdrant significantly streamline AI-driven systems such as semantic search engines, personalized recommendation systems, and retrieval-augmented generation ([RAG](https://www.pinecone.io/learn/retrieval-augmented-generation/)) systems in large language models (LLMs). For instance, vector databases enable the rapid retrieval of contextually relevant documents to feed into a language model, enhancing accuracy and response quality.

Other prominent tools include [**FAISS**](https://faiss.ai/) (developed by Facebook AI Research), suitable for in-memory vector similarity searches within applications. However, Qdrant distinguishes itself with persistent storage, robust API-driven access, distributed deployment options, and broader production suitability.

In summary, vector databases like Qdrant empower AI systems to perform intelligent, content-driven searches and recommendations, moving beyond keyword-based retrieval toward deeper, meaning-based connections between data points.

---

**Conclusion:** We’ve explored four types of databases and how to use them in practice with Docker and Python:

- *Relational (PostgreSQL)* – for structured data and transactions, ensuring consistency (useful for storing structured AI metadata, results, and configurations with integrity).
- *In-Memory (Redis)* – for fast, ephemeral data access, caching, and real-time features (useful in AI for caching model outputs, maintaining state or counters, and speeding up data-heavy operations).
- *NoSQL Document (MongoDB)* – for flexible, schema-free storage of JSON-like data (useful for evolving datasets, logs, or when dealing with varying input/output structures in ML experiments).
- *Vector (Qdrant)* – for similarity search on embeddings, powering semantic lookup and recommendations (invaluable for AI tasks involving high-dimensional feature vectors and nearest-neighbor search).

Each example showed how to get started with Docker (so you can easily replicate it on your machine) and basic queries in the database’s native language and via Python. From here, you can dive deeper into each system – e.g., learn more SQL for PostgreSQL, try out Redis pub/sub or streams for realtime flows, optimize MongoDB with indexes, or experiment with different vector indexes in Qdrant.  Good luck, and happy coding!

## Bonus: Extending PostgreSQL with Plugins: A Versatile Alternative to Redis, MongoDB, and Qdrant

PostgreSQL's extensibility allows it to emulate functionalities traditionally associated with specialized databases like Redis, MongoDB, and Qdrant. Through various [plugins and extensions](https://www.postgresql.org/download/products/6-postgresql-extensions/), PostgreSQL can serve as a multi-faceted backend for diverse application needs.

### Replacing Redis: Caching and In-Memory Data Structures

Redis is renowned for its in-memory data storage, offering rapid data access and caching capabilities. While PostgreSQL isn't inherently designed as an in-memory cache, certain extensions and configurations can approximate this functionality:

- **Unlogged Tables:** These tables do not write data to the write-ahead log, reducing disk I/O and improving performance. However, data in unlogged tables is lost upon a crash or restart, making them suitable for cache-like scenarios where data persistence isn't critical.

- **[pg_prewarm](https://www.postgresql.org/docs/current/pgprewarm.html):** This extension loads relation data into the buffer cache, effectively warming up the cache to improve read performance.

It's important to note that while PostgreSQL can mimic some caching behaviors, it doesn't natively support result caching. Each query is processed afresh, which might not match the performance of dedicated caching solutions like Redis. As one ~~expert~~ redditor points out:

> "One crucial element as to why Postgres doesn't work well as a generic cache is due to the fact it has no result cache. Every result is calculated from scratch using the source rows. Every time a query is executed."

### Emulating MongoDB: JSON and Document Storage

MongoDB's schema-less design and JSON document storage are key attractions for developers. PostgreSQL addresses similar needs through its robust JSON and JSONB data types:

- **[JSONB](https://www.postgresql.org/docs/current/datatype-json.html):** This binary format for JSON storage allows for efficient querying and indexing, enabling PostgreSQL to function effectively as a document store.

- **[Full-Text Search](https://www.postgresql.org/docs/current/textsearch.html):** Combined with JSONB, PostgreSQL's full-text search capabilities facilitate complex querying within JSON documents.

These features position PostgreSQL as a viable alternative to MongoDB for applications requiring flexible schemas and document-oriented storage.

### PostgreSQL as a VectorDB: Vector Similarity Search

Qdrant specializes in vector similarity search, crucial for applications like recommendation systems and semantic search. PostgreSQL can integrate similar capabilities through extensions:

- **[pgvector](https://github.com/pgvector/pgvector):** This extension introduces vector data types and similarity search functions to PostgreSQL, enabling efficient handling of vector embeddings.

- **[Foreign Data Wrappers (FDWs)](https://www.postgresql.org/docs/current/sql-createforeigndatawrapper.html):** FDWs allow PostgreSQL to interface with external databases, including vector databases like Qdrant. This integration enables PostgreSQL to perform vector searches by delegating operations to specialized systems.

For instance, using FDWs, PostgreSQL can query data stored in Qdrant, combining traditional relational data with advanced vector search capabilities.

### Consolidating Application Backends with PostgreSQL

By leveraging these extensions and plugins, PostgreSQL can serve as a comprehensive backend for simple applications, reducing the need for multiple specialized databases. This consolidation simplifies architecture, reduces maintenance overhead, and leverages PostgreSQL's robustness and community support.

However, it's essential to assess the specific requirements of your application. While PostgreSQL's versatility is impressive, specialized databases like Redis, MongoDB, and Qdrant are optimized for particular workloads and may offer performance benefits in scenarios that heavily rely on their unique features.

In conclusion, PostgreSQL's extensibility through plugins and extensions enables it to replicate functionalities of various specialized databases, offering a unified platform for diverse data management needs.
