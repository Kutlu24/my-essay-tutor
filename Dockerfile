FROM python:3.11-slim

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

RUN pip install --no-cache-dir --upgrade pip

COPY --chown=user pyproject.toml ./
COPY --chown=user src ./src
COPY --chown=user frontend ./frontend

RUN pip install --no-cache-dir --user -e .

EXPOSE 8000

# No persistent volume - every input here is a per-request upload, nothing
# written to disk survives (or needs to survive) a restart.
# GRAMMAR_CROSSCHECK_LOCAL_SERVER stays unset/false here too, same as
# Render: switching it on needs a resident LanguageTool JVM process this
# Dockerfile doesn't install - a real home-server win (real RAM headroom)
# but a separate change, not bundled into this migration.
CMD ["sh", "-c", "uvicorn my_essay_tutor.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
