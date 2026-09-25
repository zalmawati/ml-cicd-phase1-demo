FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Metrics + model land here. Mount a volume to this path if you want to
# persist artifacts outside the container (see README, Step 9).
ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--output-dir", "/app/artifacts"]
