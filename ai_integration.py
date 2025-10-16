# ai_integration.py
import requests
import json
import logging
import os
from typing import Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройки API (лучше хранить в переменных окружения)
XAI_API_KEY = os.getenv('XAI_API_KEY', 'YOUR_XAI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YOUR_OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', 'YOUR_ANTHROPIC_API_KEY')  # Claude API

# URL эндпоинтов
XAI_API_URL = 'https://api.x.ai/v1/chat/completions'
OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions'
ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'

def ask_ai(query: str, provider: str = "openai") -> str:
    """
    Отправка запроса к выбранному провайдеру ИИ
    
    Args:
        query: Текст запроса
        provider: Провайдер ИИ ("openai", "xai", "anthropic")
    
    Returns:
        Ответ от ИИ
    """
    
    providers = {
        "openai": _ask_openai,
        "xai": _ask_xai,
        "anthropic": _ask_anthropic
    }
    
    if provider not in providers:
        logger.warning(f"Провайдер {provider} не поддерживается, используем OpenAI")
        provider = "openai"
    
    try:
        return providers[provider](query)
    except Exception as e:
        logger.error(f"Ошибка с провайдером {provider}: {e}")
        # Пробуем следующий провайдер
        for backup_provider in [p for p in providers.keys() if p != provider]:
            try:
                logger.info(f"Пробуем провайдер {backup_provider}")
                return providers[backup_provider](query)
            except Exception as backup_error:
                logger.error(f"Ошибка с провайдером {backup_provider}: {backup_error}")
                continue
        
        # Если все провайдеры недоступны
        raise Exception("Все ИИ провайдеры недоступны. Проверьте API ключи и подключение к интернету.")

def _ask_openai(query: str) -> str:
    """Запрос к OpenAI API"""
    if OPENAI_API_KEY == 'YOUR_OPENAI_API_KEY':
        raise Exception("OpenAI API ключ не настроен")
    
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'gpt-4',  # или 'gpt-3.5-turbo'
        'messages': [
            {
                'role': 'system', 
                'content': 'Ты - эксперт по анализу данных в области производства топливных пеллет. Отвечай на русском языке, будь точным и используй данные из контекста.'
            },
            {
                'role': 'user', 
                'content': query
            }
        ],
        'temperature': 0.3,
        'max_tokens': 2000
    }
    
    response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    
    result = response.json()
    return result['choices'][0]['message']['content']

def _ask_xai(query: str) -> str:
    """Запрос к xAI Grok API"""
    if XAI_API_KEY == 'YOUR_XAI_API_KEY':
        raise Exception("xAI API ключ не настроен")
    
    headers = {
        'Authorization': f'Bearer {XAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'grok-beta',
        'messages': [{'role': 'user', 'content': query}],
        'temperature': 0.3,
        'max_tokens': 2000
    }
    
    response = requests.post(XAI_API_URL, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    
    result = response.json()
    return result['choices'][0]['message']['content']

def _ask_anthropic(query: str) -> str:
    """Запрос к Anthropic Claude API"""
    if ANTHROPIC_API_KEY == 'YOUR_ANTHROPIC_API_KEY':
        raise Exception("Anthropic API ключ не настроен")
    
    headers = {
        'X-API-Key': ANTHROPIC_API_KEY,
        'Content-Type': 'application/json',
        'Anthropic-Version': '2023-06-01'
    }
    
    data = {
        'model': 'claude-3-sonnet-20240229',
        'max_tokens': 2000,
        'temperature': 0.3,
        'system': 'Ты - эксперт по анализу данных в области производства топливных пеллет. Отвечай на русском языке, будь точным и используй данные из контекста.',
        'messages': [
            {
                'role': 'user',
                'content': query
            }
        ]
    }
    
    response = requests.post(ANTHROPIC_API_URL, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    
    result = response.json()
    return result['content'][0]['text']

# Функция для проверки доступности провайдеров
def check_ai_providers() -> dict:
    """Проверяет доступность всех провайдеров ИИ"""
    providers_status = {}
    
    for provider in ['openai', 'xai', 'anthropic']:
        try:
            test_query = "Ответь одним словом: 'работает'"
            ask_ai(test_query, provider)
            providers_status[provider] = {'available': True, 'error': None}
        except Exception as e:
            providers_status[provider] = {'available': False, 'error': str(e)}
    
    return providers_status

# Функция для получения информации о стоимости API
def get_ai_cost_estimate(query: str, provider: str = "openai") -> dict:
    """Оценивает стоимость запроса к API"""
    # Примерная оценка токенов (1 токен ≈ 0.75 слова на русском)
    estimated_tokens = len(query.split()) * 1.3  # Примерная оценка
    
    cost_per_1k_tokens = {
        "openai": {"gpt-4": 0.03, "gpt-3.5-turbo": 0.0015},
        "xai": {"grok-beta": 0.02},
        "anthropic": {"claude-3-sonnet": 0.015}
    }
    
    provider_costs = cost_per_1k_tokens.get(provider, {})
    model = list(provider_costs.keys())[0] if provider_costs else "unknown"
    cost_per_token = provider_costs.get(model, 0) / 1000
    
    estimated_cost = estimated_tokens * cost_per_token
    
    return {
        'provider': provider,
        'model': model,
        'estimated_tokens': round(estimated_tokens),
        'estimated_cost_usd': round(estimated_cost, 4),
        'estimated_cost_rub': round(estimated_cost * 75, 2)  # Примерный курс
    }