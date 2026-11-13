# PhaseWise AI — one self-contained image for AWS deployment.
# The build creates the derived datasets, ML artifacts, vector index and SQLite schema
# so a clean checkout is runnable without a second provisioning step.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8081 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app/Project

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/Project/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . /app/Project/

# Prepare all generated runtime assets during the image build.
RUN python run_database_setup.py \
    && python run_data_pipeline.py \
    && python run_model_training.py \
    && python run_knowledge_indexing.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl --fail http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["sh", "-c", "python run_api_service.py & api_pid=$!; trap 'kill $api_pid' TERM INT; exec python -m streamlit run run_dashboard.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true"]
