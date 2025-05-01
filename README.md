# Neural Network Project

Este repositório contém a implementação de uma rede neural para aprendizado de máquina. O objetivo é explorar conceitos de inteligência artificial e aprendizado profundo.

## Funcionalidades

- Treinamento de redes neurais.
- Avaliação de desempenho.
- Visualização de resultados.

## Requisitos

- Docker instalado no sistema.
- No Windows, é necessário configurar o WSL2 e instalar o suporte ao Docker. Consulte a [documentação oficial do Docker](https://docs.docker.com/docker-for-windows/wsl/) para mais detalhes.
- Placa de vídeo Nvidia com drivers atualizados e suporte ao CUDA para treinamento.

## Como usar com Docker

1. Certifique-se de que o WSL2 está configurado (caso esteja no Windows) e que o Docker está instalado e funcionando corretamente.
2. Clone o repositório:
    ```bash
    git clone https://github.com/techn-note/neural_network.git
    ```
3. Navegue até o diretório:
    ```bash
    cd neural_network
    ```
4. Construa os serviços definidos no `docker-compose.yml`:
    ```bash
    docker compose build
    ```
5. Inicie os serviços (incluindo o treinamento, se configurado):
    ```bash
    docker compose up
    ```
6. Para testar a API com o modelo já treinado (exemplo disponível em `artifacts`):
    - Navegue até o diretório `api`:
        ```bash
        cd api
        ```
    - Construa a imagem Docker da API:
        ```bash
        docker build -t neural_network_api .
        ```
    - Execute o container da API:
        ```bash
        docker run -p 8000:8000 neural_network_api
        ```
7. Acesse a API em `http://localhost:8000`. Utilize o endpoint `/predict` para enviar dados e obter predições.

## Estrutura do Projeto

- `main.py`: Script principal para treinar e avaliar a rede neural.
- `api/`: Contém a implementação da API Flask para predições.
- `artifacts/`: Contém exemplos de artefatos prontos, como o modelo treinado.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

---
Desenvolvido por TechNote.