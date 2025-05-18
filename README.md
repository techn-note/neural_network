# Neural Network API - Produção

Este repositório contém apenas a API Flask e os artefatos necessários para inferência de IA. Não há código de treino.

## Estrutura
- `api/`: Código da API Flask
- `artifacts/`: Modelos e scaler prontos para uso
- `estagios_config.json`: Configuração dos estágios (copie para a pasta `api/`)

## Requisitos
- Docker e Docker Compose instalados
- Ubuntu Server (ex: EC2 t3.micro)

## Deploy rápido
1. Suba os arquivos do projeto para o EC2 (ex: via SCP ou Git).
2. Copie o arquivo `estagios_config.json` para dentro da pasta `api/`.
3. Acesse a pasta `api/`:
   ```bash
   cd api
   ```
4. Construa a imagem:
   ```bash
   docker compose build
   ```
5. Suba a API:
   ```bash
   docker compose up -d
   ```
6. Acesse a API em `http://<IP-DO-EC2>:8000/healthcheck` para testar.

## Notas
- Não é necessário GPU.
- O container já expõe a porta 8000.
- Os artefatos devem estar em `artifacts/` na raiz do projeto.

---
Desenvolvido para produção.