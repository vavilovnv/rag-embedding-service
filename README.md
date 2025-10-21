## 🗃️ Embedding service

A simple service for generating embeddings based on a given text string. It also saves data in a [ChromaDB](https://www.trychroma.com/) organized by topic. Based on a given search string, the service will return the top n best matches.

The saved data can be used to configure RAG when communicating with LLM.

### Setting up and starting a FastAPI server
1. Install a package manager uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. Install dependencies - `uv sync`
3. Based on the .env_example file, create a .env file and fill in the variable values. he path to the model folder should be specified as the location where the model repository will be cloned (see, deploy of the embedding model).
4. Activate venv - `source .venv/bin/activate`
5. Run `uvicorn`:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080  # or any other port
```

### Main endpoints
The complete list of endpoints is available at http://0.0.0.0:8080/docs

1. `/app/v1/embeddings/get_collection` - get a collection by name or the default collection
2. `/app/v1/embeddings/update` - create or update a collection
3. `/app/v1/content/search` - search for relevant text fragments in the collection for the message

### Deploy of the embedding model
You can use any offline model compatible with [sentence_transformers](https://pypi.org/project/sentence-transformers/), for instance, [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small).

To deploy this model, do the following:
1. Install git-lfs (if you don't have it)
```bash
sudo apt update
sudo apt install git-lfs
git lfs install
```

2. Clone the model repository to the desired directory
```bash
git clone https://huggingface.co/intfloat/multilingual-e5-small
```

`Ru` Простой сервис для генерации эмбеддингов по переданному тексту. Хранит данные оффлайн в [ChromaDB](https://www.trychroma.com/) в разрезе топиков (sources). По переданной строке сервис вернет топ `n` ближайших совпадений. 
