FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OVERLAP_LANG="hi-IN" \
    OVERLAP_VOICE_ID="Nalini"

COPY pyproject.toml README.md ARCHITECTURE.md CLAUDE.md TASKS.md Makefile ./
COPY overlap ./overlap
COPY examples ./examples
COPY bench ./bench
COPY docs ./docs
COPY tools ./tools
COPY tests ./tests

RUN pip install --no-cache-dir -e .

EXPOSE 8765

CMD ["python", "-m", "overlap.demo", "--host", "0.0.0.0", "--no-browser"]
