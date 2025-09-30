import requests
import json

# Настройки для xAI Grok API
XAI_API_KEY = 'YOUR_XAI_API_KEY'  # Замени на реальный ключ с https://x.ai/api
XAI_API_URL = 'https://api.x.ai/v1/chat/completions'

# Альтернатива: OpenAI API
OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'  # Замени на реальный ключ с https://platform.openai.com
OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions'

def ask_ai(query, use_openai=False):
    """Отправка запроса к ИИ (xAI или OpenAI)."""
    if use_openai:
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': 'gpt-4',  # Или другой доступный OpenAI модель
            'messages': [{'role': 'user', 'content': query}]
        }
        url = OPENAI_API_URL
    else:
        headers = {
            'Authorization': f'Bearer {XAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': 'grok-beta',  # Или последняя модель xAI
            'messages': [{'role': 'user', 'content': query}]
        }
        url = XAI_API_URL

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        raise Exception(f"AI request failed: {e}. Try using a VPN or an alternative provider like OpenAI.")