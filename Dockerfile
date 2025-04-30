FROM python:3.9-slim

WORKDIR /app

# Instala dependências do ML
COPY ml/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia códigos ML
COPY ml /app/ml

# Cria pasta para os artefatos
RUN mkdir -p /app/artifacts

# Comando para gerar dados + treinar
CMD ["sh", "-c", "python ml/data_generation.py && python ml/preprocess.py && python ml/train.py"]