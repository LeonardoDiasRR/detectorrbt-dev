@echo off
SETLOCAL

:: Caminho do ambiente virtual
SET VENV_DIR=venv

echo [1/4] Criando ambiente virtual em %VENV_DIR%...
python -m venv %VENV_DIR%
IF ERRORLEVEL 1 (
    echo Erro ao criar o ambiente virtual.
    EXIT /B 1
)

echo [2/5] Atualizando pip...
%VENV_DIR%\Scripts\python.exe -m pip install --upgrade pip
IF ERRORLEVEL 1 (
    echo Erro ao atualizar pip.
    EXIT /B 1
)

echo [3/5] Detectando GPU CUDA...
%VENV_DIR%\Scripts\python.exe -c "import torch; print('CUDA disponível' if torch.cuda.is_available() else 'CUDA não disponível')" 2>nul
IF ERRORLEVEL 1 (
    echo PyTorch não instalado ainda. Verificando GPU via nvidia-smi...
    nvidia-smi >nul 2>&1
    IF ERRORLEVEL 1 (
        echo CPU detectada. Usando requirements_cpu.txt
        SET REQUIREMENTS_FILE=requirements_cpu.txt
    ) ELSE (
        echo GPU CUDA detectada. Usando requirements_gpu.txt
        SET REQUIREMENTS_FILE=requirements_gpu.txt
    )
) ELSE (
    %VENV_DIR%\Scripts\python.exe -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
    IF ERRORLEVEL 1 (
        echo CPU detectada. Usando requirements_cpu.txt
        SET REQUIREMENTS_FILE=requirements_cpu.txt
    ) ELSE (
        echo GPU CUDA detectada. Usando requirements_gpu.txt
        SET REQUIREMENTS_FILE=requirements_gpu.txt
    )
)

echo [4/4] Instalando dependências de %REQUIREMENTS_FILE%...
IF EXIST %REQUIREMENTS_FILE% (
    %VENV_DIR%\Scripts\pip.exe install -r %REQUIREMENTS_FILE%
    IF ERRORLEVEL 1 (
        echo Erro ao instalar dependências.
        EXIT /B 1
    )
    echo Dependências instaladas com sucesso.
) ELSE (
    echo Arquivo %REQUIREMENTS_FILE% não encontrado.
    echo Tentando instalar requirements.txt como fallback...
    IF EXIST requirements.txt (
        %VENV_DIR%\Scripts\pip.exe install -r requirements.txt
        IF ERRORLEVEL 1 (
            echo Erro ao instalar dependências.
            EXIT /B 1
        )
    ) ELSE (
        echo Nenhum arquivo de requisitos encontrado.
        EXIT /B 1
    )
)

echo ✅ Setup concluído com sucesso.
ENDLOCAL
