.PHONY: install ingest api ui test eval clean

install:
	pip install -r requirements.txt

# Chunk + embed + store the corpus (downloads the embedding model once).
ingest:
	python -m scripts.ingest --reset

# FastAPI service on http://localhost:8000  (docs at /docs)
api:
	uvicorn api.main:app --reload --port 8000

# Streamlit UI on http://localhost:8501
ui:
	streamlit run app.py

# Full offline test suite (no API key needed).
test:
	pytest -q

# Retrieval quality report (Hit@k + MRR).
eval:
	python -m eval.run_eval

clean:
	rm -rf data/chroma __pycache__ */__pycache__ .pytest_cache
