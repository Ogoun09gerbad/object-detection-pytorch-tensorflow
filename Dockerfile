FROM python:3.11-slim

WORKDIR /app

# Install of dependances 
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependance
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie of files 
COPY . .

# User non-root requiert by HuggingFace Spaces
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# Mandatory Port for HuggingFace Spaces
EXPOSE 7860

CMD ["python", "app.py"]