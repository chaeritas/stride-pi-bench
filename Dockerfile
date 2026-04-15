FROM python:3.13-slim-bookworm

RUN adduser --disabled-password agent
USER agent
WORKDIR /home/agent

COPY --chown=agent pyproject.toml README.md ./
COPY --chown=agent src src

RUN pip install --no-cache-dir --user \
    "a2a-sdk[http-server]>=0.2" \
    "uvicorn[standard]>=0.20" \
    "litellm>=1.0"

RUN pip install --no-cache-dir --user curl_cffi 2>/dev/null || true

ENV PATH="/home/agent/.local/bin:${PATH}"

HEALTHCHECK --interval=2s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9009/.well-known/agent.json')" || exit 1

ENTRYPOINT ["python", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
