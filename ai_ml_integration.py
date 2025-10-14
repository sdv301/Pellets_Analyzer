# ai_ml_integration.py
import pandas as pd
import numpy as np
import json
from ml_optimizer import ml_system
from database import query_db
from ai_integration import ask_ai

class AIMLAnalyzer:
    def __init__(self):
        self.ml_system = ml_system  # Обновляем здесь тоже
    
    def get_data_summary(self):
        """Возвращает сводку по данным"""
        try:
            # Используем обновленную систему
            status = self.ml_system.get_system_status()
            return {
                'total_samples': status.get('training_data_size', 0),
                'trained_models': len(status.get('trained_models', [])),
                'available_components': status.get('available_components', [])
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_ml_models_status(self):
        """Возвращает статус ML моделей"""
        try:
            status = self.ml_system.get_system_status()
            return {
                'is_trained': status.get('is_trained', False),
                'models_count': len(status.get('trained_models', [])),
                'models': status.get('trained_models', [])
            }
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_trends_with_ml(self):
        """Анализирует тенденции с помощью ML моделей"""
        trends = {}
        
        if not self.ml_optimizer.models:
            return {"error": "ML модели не обучены"}
        
        # Анализ влияния компонентов на свойства
        component_impact = {}
        for prop in self.ml_optimizer.models.keys():
            if 'feature_importance' in self.ml_optimizer.models[prop]:
                importance = self.ml_optimizer.models[prop]['feature_importance']
                # Сортируем по важности
                sorted_impact = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                component_impact[prop] = sorted_impact[:3]  # Топ-3 компонента
        
        trends['component_impact'] = component_impact
        
        # Поиск оптимальных составов
        optimal_compositions = {}
        for prop in ['q', 'kf']:  # Теплота сгорания и прочность
            if prop in self.ml_optimizer.models:
                try:
                    result = self.ml_optimizer.optimize_composition(prop, maximize=True)
                    if result['success']:
                        optimal_compositions[prop] = result['composition']
                except:
                    pass
        
        trends['optimal_compositions'] = optimal_compositions
        
        return trends
    
    def analyze_with_ai(self, user_query):
        """Интегрированный анализ с ИИ и ML"""
        
        # 1. Собираем данные и ML анализ
        data_summary = self.get_data_summary()
        ml_trends = self.analyze_trends_with_ml()
        
        # 2. Формируем контекст для ИИ
        context = f"""
        АНАЛИЗ ДАННЫХ ПЕЛЛЕТ:
        
        ДАННЫЕ:
        - Количество образцов: {data_summary.get('total_samples', 0)}
        - Исследуемые свойства: {', '.join(data_summary.get('trained_models', []))}
        - Компоненты: {', '.join(data_summary.get('available_components', []))}
        
        ML АНАЛИЗ:
        {json.dumps(ml_trends, indent=2, ensure_ascii=False)}
        
        ЗАПРОС ПОЛЬЗОВАТЕЛЯ: {user_query}
        
        ПРОАНИЛИЗИРУЙ данные и ML тенденции, ответь на запрос пользователя.
        Если в запросе просят найти оптимальный состав - используй данные ML моделей.
        Будь конкретен в рекомендациях и используй численные данные.
        """
        
        try:
            # 3. Отправляем запрос к ИИ
            ai_response = ask_ai(context)
            
            # 4. Извлекаем оптимальный состав из ML анализа если есть
            optimal_composition = {}
            if 'optimal_compositions' in ml_trends and 'q' in ml_trends['optimal_compositions']:
                optimal_composition = ml_trends['optimal_compositions']['q']
            
            return {
                'analysis': ai_response,
                'recommendations': "Рекомендации основаны на анализе ML моделей и данных",
                'optimal_composition': optimal_composition
            }
            
        except Exception as e:
            # Если ИИ недоступен, используем локальную логику
            return self._local_analysis(user_query, data_summary, ml_trends)
    
    def _local_analysis(self, user_query, data_summary, ml_trends):
        """Локальный анализ если ИИ недоступен"""
        
        analysis = "🔍 **Анализ на основе ML моделей:**\n\n"
        
        # Анализ влияния компонентов
        if 'component_impact' in ml_trends:
            analysis += "**Влияние компонентов на свойства:**\n"
            for prop, components in ml_trends['component_impact'].items():
                analysis += f"- {prop}: {', '.join([f'{comp}({imp:.2f})' for comp, imp in components])}\n"
            analysis += "\n"
        
        # Оптимальные составы
        if 'optimal_compositions' in ml_trends:
            analysis += "**Оптимальные составы:**\n"
            for prop, composition in ml_trends['optimal_compositions'].items():
                valid_components = {k: v for k, v in composition.items() if v > 1}
                if valid_components:
                    analysis += f"- Для {prop}: {', '.join([f'{k} {v}%' for k, v in valid_components.items()])}\n"
        
        return {
            'analysis': analysis,
            'recommendations': "Для более детального анализа подключите ИИ API",
            'optimal_composition': ml_trends.get('optimal_compositions', {}).get('q', {})
        }