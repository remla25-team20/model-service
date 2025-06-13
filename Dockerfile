FROM python:3.10

ARG MODEL_VERSION="v0.1.6-beta"

LABEL org.opencontainers.image.description="Built with multi-architecture support (amd64 + arm64). No code changes from 0.1.0."


WORKDIR /app
COPY . /app

ADD "https://github.com/remla25-team20/model-training/releases/download/${MODEL_VERSION}/Sentiment_Analysis_Model.joblib" /app/model/Sentiment_Analysis_Model.joblib
ADD "https://github.com/remla25-team20/model-training/releases/download/${MODEL_VERSION}/Sentiment_Analysis_Preprocessor.joblib" /app/model/Sentiment_Analysis_Preprocessor.joblib

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN poetry update
RUN poetry install --only=main && rm -rf $POETRY_CACHE_DIR
RUN poetry run python -m lib_ml.preprocessing

EXPOSE 8080

CMD ["poetry", "run", "python", "src/app.py"]
