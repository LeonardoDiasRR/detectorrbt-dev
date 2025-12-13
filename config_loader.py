# built-in
import os
from pathlib import Path
from typing import Dict, Any

# 3rd party
import yaml


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Carrega o arquivo de configuração YAML e retorna um dicionário com as configurações.
    
    :param config_path: Caminho para o arquivo config.yaml. Se None, procura no diretório do script.
    :return: Dicionário com as configurações carregadas.
    :raises FileNotFoundError: Se o arquivo de configuração não for encontrado.
    :raises yaml.YAMLError: Se houver erro ao fazer parsing do YAML.
    """
    if config_path is None:
        # Procura config.yaml no diretório do script
        config_path = Path(__file__).parent / "config.yaml"
        
        # Se não encontrar, tenta no diretório de trabalho atual
        if not config_path.exists():
            config_path = Path.cwd() / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Erro ao fazer parsing do arquivo de configuração: {e}")


# Carrega as configurações globalmente
CONFIG: Dict[str, Any] = load_config()


def get_config(key: str, default: Any = None) -> Any:
    """
    Retorna o valor de uma chave de configuração.
    
    :param key: Chave da configuração (suporta notação de ponto para chaves aninhadas, ex: 'qualidade_face.tamanho_bbox').
    :param default: Valor padrão caso a chave não exista.
    :return: Valor da configuração ou o valor padrão.
    """
    keys = key.split('.')
    value = CONFIG
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value


def reload_config(config_path: str = None) -> Dict[str, Any]:
    """
    Recarrega o arquivo de configuração.
    
    :param config_path: Caminho para o arquivo config.yaml.
    :return: Dicionário com as configurações recarregadas.
    """
    global CONFIG
    CONFIG = load_config(config_path)
    return CONFIG


if __name__ == "__main__":
    # Teste de carregamento de configuração
    print("=== Configurações Carregadas ===")
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
    
    print("\n=== Testando get_config ===")
    print(f"GPU Index: {get_config('gpu_index', 0)}")
    print(f"Verbose Log: {get_config('verbose_log', False)}")
    print(f"Tamanho BBox (aninhado): {get_config('qualidade_face.tamanho_bbox', 2)}")
    print(f"Chave inexistente: {get_config('chave_inexistente', 'valor_padrao')}")
