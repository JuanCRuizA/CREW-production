FROM python:3.12-slim
WORKDIR /app
COPY web/requirements.txt web/requirements.txt
RUN pip install --no-cache-dir -r web/requirements.txt
COPY web/api web/api
COPY deploy deploy
RUN useradd --create-home crew
USER crew
EXPOSE 8000
CMD ["python", "deploy/serve.py"]
