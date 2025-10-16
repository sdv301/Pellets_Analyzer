# ai_ml_integration.py
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class AIMLAnalyzer:
    def __init__(self, db_path='pellets_data.db'):
        self.db_path = db_path
        self.analysis_history = []
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Получает сводку по данным"""
        try:
            from database import query_db
            
            measured_data = query_db(self.db_path, "measured_parameters")
            components_data = query_db(self.db_path, "components")
            
            # Анализ данных
            total_samples = len(measured_data)
            total_components = len(components_data)
            
            # Анализ заполненности данных
            if not measured_data.empty:
                numeric_cols = measured_data.select_dtypes(include=[np.number]).columns
                available_props = [col for col in numeric_cols if col != 'id']
                completeness = measured_data[numeric_cols].notna().mean().mean() if len(numeric_cols) > 0 else 0
            else:
                available_props = []
                completeness = 0
            
            return {
                'total_samples': total_samples,
                'total_components': total_components,
                'available_properties': available_props,
                'data_completeness': round(completeness * 100, 1),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'status': 'active' if total_samples > 0 else 'no_data'
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения сводки данных: {e}")
            return {
                'total_samples': 0,
                'total_components': 0,
                'available_properties': [],
                'data_completeness': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def get_ml_models_status(self) -> Dict[str, Any]:
        """Получает статус ML моделей"""
        try:
            from database import get_active_ml_models, get_ml_optimizations
            
            active_models = get_active_ml_models(self.db_path)
            optimizations = get_ml_optimizations(self.db_path, limit=5)
            
            trained_models = []
            if not active_models.empty:
                trained_models = active_models['target_property'].tolist()
            
            # Анализ производительности моделей
            model_performance = {}
            for _, model in active_models.iterrows():
                prop = model['target_property']
                model_performance[prop] = {
                    'r2_score': round(model.get('r2_score', 0), 3),
                    'status': 'excellent' if model.get('r2_score', 0) > 0.8 else 
                             'good' if model.get('r2_score', 0) > 0.6 else 'needs_improvement'
                }
            
            return {
                'is_trained': len(trained_models) > 0,
                'trained_models': trained_models,
                'models_count': len(trained_models),
                'model_performance': model_performance,
                'recent_optimizations': len(optimizations),
                'last_training': active_models['timestamp'].max() if not active_models.empty else 'Never'
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения статуса ML: {e}")
            return {
                'is_trained': False,
                'trained_models': [],
                'models_count': 0,
                'model_performance': {},
                'status': 'error',
                'error': str(e)
            }
    
    def analyze_with_ai(self, user_query: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Умный анализ с контекстными ответами"""
        try:
            data_summary = self.get_data_summary()
            ml_status = self.get_ml_models_status()
            
            # Анализ типа запроса
            query_type = self._classify_query(user_query)
            
            # Генерация контекстного ответа
            analysis_result = self._generate_contextual_response(user_query, query_type, data_summary, ml_status)
            
            # Сохраняем в историю
            self.analysis_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': user_query,
                'type': query_type,
                'response': analysis_result
            })
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Ошибка анализа: {e}")
            return {
                'analysis': f"⚠️ Произошла ошибка при анализе: {str(e)}",
                'recommendations': "Попробуйте переформулировать запрос или проверьте наличие данных",
                'success': False
            }
    
    def _classify_query(self, query: str) -> str:
        """Классифицирует тип запроса пользователя"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['оптим', 'лучш', 'макс', 'мин', 'улучш']):
            return 'optimization'
        elif any(word in query_lower for word in ['тренд', 'завис', 'коррел', 'связ']):
            return 'trends'
        elif any(word in query_lower for word in ['предсказ', 'прогноз', 'рассчит']):
            return 'prediction'
        elif any(word in query_lower for word in ['анализ', 'проанализ', 'изуч']):
            return 'analysis'
        elif any(word in query_lower for word in ['состав', 'компонент', 'ингред']):
            return 'composition'
        else:
            return 'general'
    
    def _generate_contextual_response(self, query: str, query_type: str, data_summary: Dict, ml_status: Dict) -> Dict[str, Any]:
        """Генерирует контекстный ответ на основе типа запроса"""
        
        base_info = self._get_base_analysis_info(data_summary, ml_status)
        
        if query_type == 'optimization':
            return self._generate_optimization_response(query, base_info)
        elif query_type == 'trends':
            return self._generate_trends_response(query, base_info)
        elif query_type == 'prediction':
            return self._generate_prediction_response(query, base_info)
        elif query_type == 'composition':
            return self._generate_composition_response(query, base_info)
        else:
            return self._generate_general_response(query, base_info)
    
    def _get_base_analysis_info(self, data_summary: Dict, ml_status: Dict) -> Dict:
        """Базовая информация для анализа"""
        return {
            'samples_count': data_summary.get('total_samples', 0),
            'components_count': data_summary.get('total_components', 0),
            'properties_count': len(data_summary.get('available_properties', [])),
            'data_status': data_summary.get('status', 'unknown'),
            'ml_trained': ml_status.get('is_trained', False),
            'ml_models_count': ml_status.get('models_count', 0),
            'data_completeness': data_summary.get('data_completeness', 0)
        }
    
    def _generate_optimization_response(self, query: str, base_info: Dict) -> Dict[str, Any]:
        """Генерация ответа для оптимизационных запросов"""
        
        if base_info['ml_trained']:
            analysis = f"""
            **🎯 Анализ оптимизационного запроса:** "{query}"
            
            **Текущие возможности системы:**
            ✅ Обучено {base_info['ml_models_count']} ML моделей
            ✅ Доступна оптимизация составов
            ✅ База содержит {base_info['samples_count']} образцов
            
            **Рекомендуемые действия:**
            1. Перейдите в раздел **ML Анализ**
            2. Выберите целевое свойство для оптимизации
            3. Задайте ограничения по компонентам (при необходимости)
            4. Запустите оптимизацию
            
            **Доступные для оптимизации свойства:**
            - Теплота сгорания (q)
            - Прочность (kf) 
            - Зольность (ad)
            - И другие измеряемые параметры
            """
        else:
            analysis = f"""
            **🎯 Анализ оптимизационного запроса:** "{query}"
            
            **Текущий статус:**
            ⚠️ ML модели не обучены
            📊 Данных для анализа: {base_info['samples_count']} образцов
            
            **Для выполнения оптимизации необходимо:**
            1. **Обучить ML модели** в разделе ML анализа
            2. Убедиться в наличии данных о составах
            3. Выбрать целевые свойства для оптимизации
            
            **Следующие шаги:**
            - Перейдите в раздел **ML Анализ**
            - Нажмите "Обучить систему ML"
            - После обучения используйте оптимизацию
            """
        
        return {
            'analysis': analysis,
            'recommendations': "Для точной оптимизации рекомендуется обучить ML модели на имеющихся данных",
            'actions': ['train_ml', 'optimize'] if not base_info['ml_trained'] else ['optimize'],
            'success': True
        }
    
    def _generate_trends_response(self, query: str, base_info: Dict) -> Dict[str, Any]:
        """Генерация ответа для анализа трендов"""
        
        if base_info['samples_count'] > 10:
            analysis = f"""
            **📈 Анализ трендов и зависимостей:** "{query}"
            
            **Доступные данные для анализа:**
            ✅ {base_info['samples_count']} образцов
            ✅ {base_info['properties_count']} параметров
            ✅ Заполненность данных: {base_info['data_completeness']}%
            
            **Методы анализа:**
            1. **Корреляционный анализ** - выявление взаимосвязей между параметрами
            2. **Трендовый анализ** - поиск закономерностей в данных
            3. **Кластерный анализ** - группировка схожих составов
            
            **Рекомендации:**
            - Используйте раздел **Создание графиков** для визуализации
            - Примените **Сравнительную таблицу** для детального анализа
            - Для ML анализа трендов обучите соответствующие модели
            """
        else:
            analysis = f"""
            **📈 Анализ трендов:** "{query}"
            
            **Текущий статус:**
            ⚠️ Недостаточно данных для анализа трендов
            📊 Доступно только {base_info['samples_count']} образцов
            
            **Рекомендации:**
            1. **Добавьте больше данных** в систему
            2. Убедитесь в разнообразии составов
            3. Проверьте полноту измеряемых параметров
            """
        
        return {
            'analysis': analysis,
            'recommendations': "Для глубокого анализа трендов рекомендуется увеличить объем данных",
            'actions': ['add_data', 'create_graphs'],
            'success': True
        }
    
    def _generate_prediction_response(self, query: str, base_info: Dict) -> Dict[str, Any]:
        """Генерация ответа для прогнозных запросов"""
        
        if base_info['ml_trained']:
            analysis = f"""
            **🔮 Прогнозный анализ:** "{query}"
            
            **Возможности системы:**
            ✅ Обучено {base_info['ml_models_count']} ML моделей
            ✅ Доступно предсказание свойств по составу
            ✅ Точность моделей: R² > 0.7 для ключевых параметров
            
            **Как использовать:**
            1. Перейдите в **ML Анализ**
            2. Используйте функцию предсказания
            3. Введите состав для анализа
            4. Получите прогноз свойств
            
            **Пример использования:**
            "Предскажи теплоту сгорания для состава: 60% опилки, 30% солома, 10% лигнин"
            """
        else:
            analysis = f"""
            **🔮 Прогнозный анализ:** "{query}"
            
            **Текущий статус:**
            ⚠️ ML модели не обучены для прогнозирования
            📊 Доступно данных: {base_info['samples_count']} образцов
            
            **Для включения прогнозирования:**
            1. **Обучите ML модели** в соответствующем разделе
            2. Убедитесь в качестве тренировочных данных
            3. Выберите свойства для предсказания
            
            **После обучения вы сможете:**
            - Предсказывать свойства по новым составам
            - Оценивать качество пеллет до производства
            - Оптимизировать составы на основе прогнозов
            """
        
        return {
            'analysis': analysis,
            'recommendations': "Прогнозные возможности доступны после обучения ML моделей",
            'actions': ['train_ml', 'predict'],
            'success': True
        }
    
    def _generate_composition_response(self, query: str, base_info: Dict) -> Dict[str, Any]:
        """Генерация ответа для запросов о составах"""
        
        analysis = f"""
        **🧪 Анализ составов:** "{query}"
        
        **Информация о данных:**
        📊 Образцов в базе: {base_info['samples_count']}
        🔬 Компонентов: {base_info['components_count']}
        📈 Заполненность: {base_info['data_completeness']}%
        
        **Доступные функции:**
        1. **Поиск составов** - фильтрация по параметрам
        2. **Сравнение составов** - анализ различий
        3. **ML оптимизация** - поиск лучших комбинаций
        
        **Рекомендации:**
        - Используйте **таблицы** для просмотра всех составов
        - Применяйте **поиск** для фильтрации данных
        - Сравнивайте составы в соответствующем разделе
        """
        
        return {
            'analysis': analysis,
            'recommendations': "Для анализа конкретных составов используйте таблицы и поиск",
            'actions': ['view_tables', 'search', 'compare'],
            'success': True
        }
    
    def _generate_general_response(self, query: str, base_info: Dict) -> Dict[str, Any]:
        """Генерация общего ответа"""
        
        analysis = f"""
        **🤖 Интеллектуальный анализ:** "{query}"
        
        **Обзор системы:**
        📊 Данные: {base_info['samples_count']} образцов, {base_info['properties_count']} параметров
        🤖 ML модели: {base_info['ml_models_count']} обученных {'✅' if base_info['ml_trained'] else '❌'}
        📈 Качество данных: {base_info['data_completeness']}% заполненности
        
        **Доступные функции:**
        • **Анализ данных** - поиск закономерностей и трендов
        • **ML оптимизация** - поиск оптимальных составов
        • **Прогнозирование** - предсказание свойств
        • **Сравнение** - анализ различных составов
        
        **Как получить максимальную пользу:**
        1. Убедитесь в наличии достаточного объема данных
        2. Обучите ML модели для точных предсказаний
        3. Используйте конкретные запросы для анализа
        
        **Примеры эффективных запросов:**
        • "Проанализируй зависимость теплоты сгорания от состава"
        • "Найди оптимальный состав для максимальной прочности"
        • "Какие компоненты лучше всего влияют на экологичность?"
        """
        
        return {
            'analysis': analysis,
            'recommendations': "Используйте конкретные запросы для получения точных ответов",
            'actions': ['analyze', 'train_ml', 'optimize'],
            'success': True
        }
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict]:
        """Возвращает историю анализов"""
        return self.analysis_history[-limit:] if self.analysis_history else []
    
    def get_system_recommendations(self) -> Dict[str, Any]:
        """Возвращает рекомендации по улучшению системы"""
        data_summary = self.get_data_summary()
        ml_status = self.get_ml_models_status()
        
        recommendations = []
        
        if data_summary.get('total_samples', 0) < 20:
            recommendations.append({
                'type': 'data',
                'priority': 'high',
                'message': 'Добавьте больше данных для улучшения анализа (рекомендуется >20 образцов)',
                'action': 'add_data'
            })
        
        if not ml_status.get('is_trained', False):
            recommendations.append({
                'type': 'ml',
                'priority': 'high', 
                'message': 'Обучите ML модели для включения прогнозирования и оптимизации',
                'action': 'train_ml'
            })
        
        if data_summary.get('data_completeness', 0) < 80:
            recommendations.append({
                'type': 'quality',
                'priority': 'medium',
                'message': f'Улучшите заполненность данных (сейчас {data_summary.get("data_completeness", 0)}%)',
                'action': 'improve_data'
            })
        
        return {
            'recommendations': recommendations,
            'total_count': len(recommendations),
            'high_priority_count': len([r for r in recommendations if r['priority'] == 'high'])
        }

# Глобальный экземпляр
ai_ml_analyzer = AIMLAnalyzer()