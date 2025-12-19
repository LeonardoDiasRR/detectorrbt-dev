#!/bin/bash

# Interrompe o script se qualquer comando falhar
set -e

# Caminho para o ambiente virtual
VENV_DIR="./venv"

echo "➡️ Criando ambiente virtual em $VENV_DIR..."
python3 -m venv "$VENV_DIR"

echo "✅ Ambiente virtual criado."

# Atualiza o pip
echo "➡️ Atualizando pip..."
"$VENV_DIR/bin/pip" install --upgrade pip

echo "✅ pip atualizado."

# Detecta GPU CUDA
echo "➡️ Detectando GPU CUDA..."
if "$VENV_DIR/bin/python" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✅ GPU CUDA detectada. Usando requirements_gpu.txt"
    REQUIREMENTS_FILE="requirements_gpu.txt"
elif command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU NVIDIA detectada via nvidia-smi. Usando requirements_gpu.txt"
    REQUIREMENTS_FILE="requirements_gpu.txt"
else
    echo "ℹ️ CPU detectada. Usando requirements_cpu.txt"
    REQUIREMENTS_FILE="requirements_cpu.txt"
fi

# Instala dependências
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "➡️ Instalando dependências de $REQUIREMENTS_FILE..."
    "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS_FILE"
    echo "✅ Dependências instaladas."
elif [ -f "requirements.txt" ]; then
    echo "⚠️ Arquivo $REQUIREMENTS_FILE não encontrado."
    echo "➡️ Usando requirements.txt como fallback..."
    "$VENV_DIR/bin/pip" install -r requirements.txt
    echo "✅ Dependências instaladas."
else
    echo "❌ Nenhum arquivo de requisitos encontrado."
    exit 1
fi

echo ""
echo "➡️ Copiando arquivos de configuração..."

# Copia config-sample.yaml para config.yaml se não existir
if [ ! -f "config.yaml" ]; then
    if [ -f "config-sample.yaml" ]; then
        cp config-sample.yaml config.yaml
        echo "✅ config.yaml criado a partir de config-sample.yaml"
    else
        echo "⚠️ Arquivo config-sample.yaml não encontrado"
    fi
else
    echo "ℹ️ config.yaml já existe - mantendo arquivo atual"
fi

# Detecta GPU CUDA e ajusta config.yaml se houver GPU disponível
echo ""
echo "➡️ Detectando GPU CUDA para ajuste automático de configuração..."
if "$VENV_DIR/bin/python" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_ID=$(python3 -c "import torch; print(0)" 2>/dev/null)
    echo "✅ GPU CUDA detectada (cuda:$GPU_ID). Ajustando config.yaml para usar GPU..."
    
    # Usa Python para modificar config.yaml de forma confiável
    "$VENV_DIR/bin/python" << 'EOF'
import yaml
import sys

try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Ajusta device para GPU em modelo_deteccao
    if 'modelo_deteccao' in config:
        config['modelo_deteccao']['device'] = 'cuda:0'
        print("  • modelo_deteccao.device → cuda:0")
    
    # Ajusta device para GPU em modelo_landmark
    if 'modelo_landmark' in config:
        config['modelo_landmark']['device'] = 'cuda:0'
        print("  • modelo_landmark.device → cuda:0")
    
    # Salva as alterações
    with open('config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print("✅ config.yaml atualizado para usar GPU (cuda:0)")
    
except Exception as e:
    print(f"⚠️ Erro ao modificar config.yaml: {e}")
    print("   Config permanecerá com configurações padrão (CPU)")
    sys.exit(0)
EOF
else
    echo "ℹ️ GPU CUDA não detectada. Mantendo config.yaml com configurações CPU padrão"
fi

# Copia .env-exemplo para .env se não existir
if [ ! -f ".env" ]; then
    if [ -f ".env-exemplo" ]; then
        cp .env-exemplo .env
        echo "✅ .env criado a partir de .env-exemplo"
    else
        echo "⚠️ Arquivo .env-exemplo não encontrado"
    fi
else
    echo "ℹ️ .env já existe - mantendo arquivo atual"
fi

echo ""
echo "➡️ Extraindo caminho do modelo de detecção..."

# Extrai model_path da seção modelo_deteccao do config.yaml
MODEL_PATH=$(grep -A 5 "^modelo_deteccao:" config.yaml | grep "model_path:" | sed 's/.*model_path:[[:space:]]*"\?\([^"]*\)"\?.*/\1/' | xargs)

if [ -z "$MODEL_PATH" ]; then
    echo "⚠️ Não foi possível extrair model_path do config.yaml"
    MODEL_PATH="yolo-models/yolov12n-face.pt"
    echo "   Usando valor padrão: $MODEL_PATH"
else
    echo "   Modelo de detecção: $MODEL_PATH"
fi

echo ""
echo "➡️ Configurando TensorRT (se GPU disponível)..."

if [ -f "$MODEL_PATH" ]; then
    "$VENV_DIR/bin/python" setup_tensorrt.py "$MODEL_PATH"
else
    echo "⚠️ Modelo não encontrado: $MODEL_PATH"
    echo "   TensorRT não será configurado - você pode executar manualmente:"
    echo "   $VENV_DIR/bin/python setup_tensorrt.py <caminho_modelo>"
fi

echo ""
echo "✅ Setup concluído com sucesso."
