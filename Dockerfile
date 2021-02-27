# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8.8-slim-buster

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    # deps for installing poetry
    curl \
    # deps for building python deps
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6
    
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app

COPY . /app


# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
RUN useradd appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "src.api:app"]
