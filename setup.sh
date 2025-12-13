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

echo "✅ Setup concluído com sucesso."
