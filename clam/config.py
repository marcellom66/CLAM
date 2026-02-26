import os
import yaml
from pydantic import BaseModel

class LLMConfig(BaseModel):
    provider: str
    model: str
    base_url: str
    temperature: float

class ShortTermMemoryConfig(BaseModel):
    promotion_threshold: int
    decay_time_minutes: int
    min_score: int

class LongTermMemoryConfig(BaseModel):
    provider: str
    path: str
    semantic_collection: str
    episodic_collection: str
    graph_db: str

class MemoryConfig(BaseModel):
    short_term: ShortTermMemoryConfig
    long_term: LongTermMemoryConfig

class APIConfig(BaseModel):
    host: str
    port: int

class ClamConfig(BaseModel):
    llm: LLMConfig
    memory: MemoryConfig
    api: APIConfig
    # Language code for UI & LLM prompts. Default: 'en' so old config files still work.
    language: str = "en"

def load_config(config_path: str) -> ClamConfig:
    """
    Carica e valida il file YAML fornendo autocompletamento e type safety.
    (10-Year Rule: Niente costanti globali slegate)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Error: File '{config_path}' mancante. Assicurati che il config.yaml sia nella root.")
        
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    return ClamConfig(**data)

# Caricamento automatico path relativo alla root del progetto
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_config_path = os.path.join(_root_dir, "config.yaml")

CONFIG = load_config(_config_path)
