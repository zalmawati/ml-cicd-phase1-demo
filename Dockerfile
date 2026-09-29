FROM python:3.11-slim

# CI passes the commit SHA in at build time so every MLflow run can record
# exactly which code produced it (there is no .git folder inside the image).
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY params.yaml .
COPY src/ ./src/

# Where the tracking server lives is decided at run time (MLFLOW_TRACKING_URI),
# never baked into the image.
ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--output-dir", "/app/artifacts"]
