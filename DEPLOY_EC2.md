# Guia de Deploy na AWS EC2 - Aquasense + Neural Network API

Este guia mostra como rodar os dois projetos (Aquasense-API e Neural Network API) em um único ambiente Docker na EC2.

## Arquitetura

```
┌─────────────────────────────────────────┐
│          AWS EC2 Ubuntu                 │
│                                         │
│  ┌────────────────────────────────┐    │
│  │   Docker Network (bridge)      │    │
│  │                                 │    │
│  │  ┌──────────────────────────┐  │    │
│  │  │  aquasense-api:5000      │  │    │
│  │  │  (Flask + MongoDB)       │  │    │
│  │  └───────────┬──────────────┘  │    │
│  │              │                  │    │
│  │              │ HTTP request     │    │
│  │              ▼                  │    │
│  │  ┌──────────────────────────┐  │    │
│  │  │  neural-api:8000         │  │    │
│  │  │  (TensorFlow + SHAP)     │  │    │
│  │  └──────────────────────────┘  │    │
│  │                                 │    │
│  └────────────────────────────────┘    │
│                                         │
│  Portas expostas:                       │
│  - 5000 → Aquasense API                 │
│  - 8000 → Neural Network API            │
└─────────────────────────────────────────┘
```

## Pré-requisitos

### 1. Configurar a instância EC2

- **Tipo:** t2.micro (Free Tier - 1 vCPU, 1GB RAM)
  - ⚠️ **Atenção:** Com 1GB RAM, o TensorFlow pode ser lento. Para produção com muito tráfego, considere t3.small ou superior.
- **Storage:** 20GB+ (para imagens Docker e dependências)
- **OS:** Ubuntu 22.04 LTS (recomendado para t2.micro)
- **Security Group:** abrir portas:
  - 22 (SSH)
  - 5000 (Aquasense API)
  - 8000 (Neural Network API)

### 2. Instalar Docker e Docker Compose

Conecte-se via SSH à sua EC2 e execute:

```bash
# Atualizar pacotes
sudo apt-get update
sudo apt-get upgrade -y

# Instalar Docker
sudo apt-get install -y docker.io docker-compose

# Habilitar Docker no boot
sudo systemctl enable --now docker

# Adicionar usuário ao grupo docker (logout/login necessário depois)
sudo usermod -aG docker $USER

# Verificar instalação
docker --version
docker-compose --version
```

**Importante:** Após executar `usermod`, faça logout e login novamente para aplicar as permissões do grupo docker.

## Setup dos Projetos

### 1. Clonar os repositórios

Crie um diretório para os projetos:

```bash
mkdir -p ~/projects
cd ~/projects

# Clonar neural_network
git clone https://github.com/techn-note/neural_network.git

# Clonar Aquasense-API (ajuste a URL conforme seu repositório)
git clone https://github.com/SEU_USUARIO/Aquasense-API.git
```

**Estrutura esperada:**
```
~/projects/
├── neural_network/
│   ├── api/
│   ├── artifacts/
│   └── docker-compose-full.yml
└── Aquasense-API/
    ├── app.py
    ├── Dockerfile
    └── ...
```

### 2. Rodar com Docker Compose

⚠️ **Nota:** Como o Aquasense-API usa MongoDB, certifique-se de que você tem acesso a uma instância MongoDB (local, MongoDB Atlas ou container). Se precisar adicionar MongoDB ao Docker Compose, veja a seção de troubleshooting no final.

Volte para a pasta do `neural_network` onde está o `docker-compose-full.yml`:

```bash
cd ~/projects/neural_network

# Build das imagens (primeira vez ou após mudanças)
docker-compose -f docker-compose-full.yml build

# Subir os containers em modo daemon (background)
docker-compose -f docker-compose-full.yml up -d

# Verificar status dos containers
docker-compose -f docker-compose-full.yml ps

# Ver logs em tempo real
docker-compose -f docker-compose-full.yml logs -f

# Ver logs de um serviço específico
docker-compose -f docker-compose-full.yml logs -f aquasense-api
docker-compose -f docker-compose-full.yml logs -f neural-api
```

### 4. Testar as APIs

```bash
# Testar healthcheck da Neural API
curl http://localhost:8000/healthcheck
# Resposta esperada: {"status":"online"}

# Testar predict da Neural API diretamente
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperatura": 26.5,
    "ph": 7.2,
    "tds": 450,
    "fase": "crescimento"
  }'

# Testar Aquasense API (substituir com sua rota válida)
curl http://localhost:5000/
```

### 5. Testar comunicação entre APIs

```bash
# Testar rota /neural-predict da Aquasense que chama a Neural API
curl -X POST http://localhost:5000/neural-predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperatura": 26.5,
    "ph": 7.2,
    "tds": 450,
    "fase": "crescimento"
  }'
```

Você deve receber a resposta da Neural API através da Aquasense API.

## Comandos Úteis

### Gerenciar containers

```bash
cd ~/projects/neural_network

# Parar containers
docker-compose -f docker-compose-full.yml stop

# Iniciar containers parados
docker-compose -f docker-compose-full.yml start

# Reiniciar containers
docker-compose -f docker-compose-full.yml restart

# Parar e remover containers
docker-compose -f docker-compose-full.yml down

# Parar, remover containers e volumes
docker-compose -f docker-compose-full.yml down -v

# Rebuild forçado (após mudanças no código)
docker-compose -f docker-compose-full.yml build --no-cache
docker-compose -f docker-compose-full.yml up -d
```

### Debugging

```bash
# Entrar no container (shell interativo)
docker exec -it neural-api bash
docker exec -it aquasense-api bash

# Ver logs detalhados
docker-compose -f docker-compose-full.yml logs --tail=100 neural-api
docker-compose -f docker-compose-full.yml logs --tail=100 aquasense-api

# Inspecionar container
docker inspect neural-api
docker inspect aquasense-api

# Ver uso de recursos
docker stats

# Verificar rede
docker network ls
docker network inspect neural_network_aquasense-network
```

### Limpeza de espaço em disco

```bash
# Remover imagens não utilizadas
docker image prune -a

# Remover volumes não utilizados
docker volume prune

# Limpeza completa (cuidado!)
docker system prune -a --volumes
```

## Acessar APIs externamente

Para acessar as APIs do seu aplicativo/navegador:

```text
http://<IP_PUBLICO_EC2>:5000/neural-predict  → Endpoint para predição via Aquasense
http://<IP_PUBLICO_EC2>:8000/predict         → Endpoint direto da Neural API
http://<IP_PUBLICO_EC2>:5000                 → Outras rotas do Aquasense
```

**Importante:** Certifique-se de que as portas 5000 e 8000 estão abertas no Security Group da EC2.

**Uso da rota `/neural-predict`:**

```bash
curl -X POST http://<IP_PUBLICO_EC2>:5000/neural-predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperatura": 26.5,
    "ph": 7.2,
    "tds": 450,
    "fase": "crescimento"
  }'
```

## Otimizações para t2.micro (1GB RAM)

A t2.micro tem apenas 1GB de RAM, o que pode ser limitado para rodar TensorFlow. Aqui estão algumas dicas:

### 1. Habilitar swap (memória virtual)

```bash
# Criar arquivo swap de 2GB
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Tornar permanente (adiciona ao /etc/fstab)
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verificar swap ativo
free -h
```

### 2. Reduzir workers do Gunicorn

No `Dockerfile.api`, já está configurado com `--workers 1`. Mantenha assim para economizar RAM.

### 3. Monitorar uso de memória

```bash
# Ver memória em tempo real
watch -n 1 free -m

# Ver uso por container
docker stats
```

### 4. Se a memória for insuficiente

- **Opção 1:** Upgrade para t3.small (2GB RAM) - custa ~$15/mês
- **Opção 2:** Use a Neural API apenas quando necessário
- **Opção 3:** Hospede a Neural API em um serviço serverless (AWS Lambda)

## Troubleshooting

### Problema: Container não sobe

```bash
# Verificar logs
docker-compose -f docker-compose-full.yml logs neural-api
docker-compose -f docker-compose-full.yml logs aquasense-api

# Verificar se portas estão em uso
sudo lsof -iTCP -sTCP:LISTEN -P -n | grep 5000
sudo lsof -iTCP -sTCP:LISTEN -P -n | grep 8000
```

### Problema: Erro "No space left on device"

```bash
# Limpar imagens e containers antigos
docker system prune -a

# Se necessário, aumentar disco da EC2 pelo painel AWS
```

### Problema: Neural API não responde

```bash
# Verificar healthcheck
docker inspect neural-api | grep -A 10 Health

# Entrar no container e testar
docker exec -it neural-api bash
curl http://localhost:8000/healthcheck
```

### Problema: Aquasense não conecta à Neural API

```bash
# Verificar se estão na mesma rede
docker network inspect neural_network_aquasense-network

# Testar conectividade dentro do container
docker exec -it aquasense-api bash
curl http://neural-api:8000/healthcheck
```

## Manutenção

### Atualizar código

```bash
cd ~/projects/neural_network
git pull

cd ~/projects/Aquasense-API
git pull

# Rebuild e restart
cd ~/projects/neural_network
docker-compose -f docker-compose-full.yml build
docker-compose -f docker-compose-full.yml up -d
```

### Backup de volumes (se necessário)

```bash
# Listar volumes
docker volume ls

# Backup de volume (exemplo)
docker run --rm -v <volume_name>:/data -v $(pwd):/backup \
  ubuntu tar czf /backup/backup.tar.gz /data
```

## Monitoramento

Para produção, considere adicionar:

- **Logs centralizados:** ELK Stack ou AWS CloudWatch
- **Monitoring:** Prometheus + Grafana
- **Alertas:** AWS SNS ou similar
- **Auto-restart:** já configurado com `restart: unless-stopped`

## Próximos Passos

1. Configure HTTPS com Let's Encrypt (Certbot + Nginx reverse proxy)
2. Configure CI/CD para deploy automático
3. Adicione testes automatizados
4. Configure backup automático do MongoDB
5. Implemente rate limiting e autenticação

---

**Dúvidas ou problemas?** Consulte os logs detalhados e verifique a documentação de cada projeto.
