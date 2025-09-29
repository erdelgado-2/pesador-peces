FROM python:3.10-slim
WORKDIR /app
COPY ["requirements.txt", "/app/"]
RUN pip install --no-cache-dir -r requirements.txt
ENV ARTIFACTS_DIR=/app/artifacts
COPY ["src/", "/app/src/"] 
RUN python3 /app/src/train_model.py
CMD ["python3","src/app.py"]
