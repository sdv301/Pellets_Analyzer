# ml_optimizer.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import warnings
import re
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

class PelletPropertyPredictor:
    """
    Модель предсказания свойств пеллет на основе состава
    """
    def __init__(self):
        self.models = {}  # property_name -> trained_model
        self.scalers = {}  # property_name -> scaler
        self.feature_names = []  # названия компонентов
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Подготавливает признаки (компоненты) из данных
        Возвращает матрицу признаков и список названий компонентов
        """
        # Автоматически определяем компоненты из данных
        component_columns = []
        possible_components = [
            'Опилки', 'Солома', 'Картон', 'Подсолнечный_жмых', 
            'Рисовая_шелуха', 'Угольный_шлам', 'Торф', 'Бурый_уголь', 
            'СМС', 'Пластик', 'Древесная_мука', 'Щепа'
        ]
        
        # Ищем колонки с компонентами в данных
        for comp in possible_components:
            if comp in data.columns:
                component_columns.append(comp)
        
        if not component_columns:
            # Если нет готовых колонок, пытаемся извлечь из composition
            component_columns = self._extract_components_from_composition(data)
        
        self.feature_names = component_columns
        print(f"📋 Обнаружено компонентов: {len(component_columns)}")
        print(f"📋 Компоненты: {component_columns}")
        
        # Создаем матрицу признаков
        X = data[component_columns].fillna(0).values
        return X, component_columns
    
    def _extract_components_from_composition(self, data: pd.DataFrame) -> List[str]:
        """Извлекает компоненты из колонки composition"""
        all_components = set()
        
        for comp_str in data.get('composition', []):
            if pd.notna(comp_str):
                # Простой парсинг - ищем слова-компоненты
                words = re.findall(r'[А-Яа-яA-Za-z_]+', str(comp_str))
                for word in words:
                    if len(word) > 2:  # Игнорируем короткие слова
                        all_components.add(word)
        
        return list(all_components)[:15]  # Ограничиваем количество
    
    def train(self, data: pd.DataFrame, target_properties: List[str]) -> bool:
        """
        Обучает модели для предсказания свойств
        """
        try:
            # Подготавливаем признаки
            X, feature_names = self.prepare_features(data)
            self.feature_names = feature_names
            
            if X.shape[1] == 0:
                print("❌ Не найдено компонентов для обучения")
                return False
            
            trained_count = 0
            
            for target_property in target_properties:
                if target_property not in data.columns:
                    print(f"⚠️ Свойство {target_property} не найдено в данных")
                    continue
                
                # Подготавливаем целевую переменную
                y = data[target_property].values
                valid_mask = ~np.isnan(y)
                
                if np.sum(valid_mask) < 5:
                    print(f"⚠️ Недостаточно данных для {target_property}: {np.sum(valid_mask)} samples")
                    continue
                
                X_clean = X[valid_mask]
                y_clean = y[valid_mask]
                
                # Масштабируем признаки
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_clean)
                
                # Разделяем на train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_clean, test_size=0.2, random_state=42, 
                    shuffle=True
                )
                
                # Обучаем модель
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train)
                
                # Оцениваем качество
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Сохраняем модель
                self.models[target_property] = model
                self.scalers[target_property] = scaler
                
                trained_count += 1
                print(f"✅ Модель {target_property}: R²={r2:.3f}, MAE={mae:.3f}")
            
            self.is_trained = trained_count > 0
            print(f"🎯 Обучено моделей: {trained_count}")
            return self.is_trained
            
        except Exception as e:
            print(f"❌ Ошибка обучения: {e}")
            return False
    
    def predict(self, composition: Dict[str, float], target_property: str) -> Optional[float]:
        """
        Предсказывает значение свойства для заданного состава
        """
        if not self.is_trained or target_property not in self.models:
            return None
        
        try:
            # Создаем вектор признаков
            features = np.array([[composition.get(comp, 0.0) for comp in self.feature_names]])
            
            # Масштабируем
            scaler = self.scalers[target_property]
            features_scaled = scaler.transform(features)
            
            # Предсказываем
            prediction = self.models[target_property].predict(features_scaled)[0]
            return prediction
            
        except Exception as e:
            print(f"❌ Ошибка предсказания: {e}")
            return None

class CompositionOptimizer:
    """
    AI Agent для оптимизации состава пеллет
    """
    def __init__(self, predictor: PelletPropertyPredictor):
        self.predictor = predictor
        self.available_components = predictor.feature_names if predictor else []
    
    def optimize(self, 
                target_property: str, 
                maximize: bool = True,
                constraints: Optional[Dict] = None,
                max_iterations: int = 1000) -> Dict:
        """
        Находит оптимальный состав для максимизации/минимизации целевого свойства
        
        Args:
            target_property: Свойство для оптимизации
            maximize: True - максимизировать, False - минимизировать
            constraints: Ограничения на компоненты {'component': (min, max)}
            max_iterations: Максимальное количество итераций оптимизации
        """
        if not self.predictor.is_trained or target_property not in self.predictor.models:
            return {
                'success': False,
                'error': f'Модель для свойства {target_property} не обучена'
            }
        
        # Настройки по умолчанию
        if constraints is None:
            constraints = {}
        
        n_components = len(self.available_components)
        
        def objective_function(x):
            """Целевая функция для оптимизации"""
            composition = dict(zip(self.available_components, x))
            prediction = self.predictor.predict(composition, target_property)
            
            if prediction is None:
                return 1e6  # Большая штрафная функция
            
            return -prediction if maximize else prediction
        
        def sum_constraint(x):
            """Ограничение: сумма компонентов = 100%"""
            return np.sum(x) - 100
        
        def component_constraints(x):
            """Ограничения на отдельные компоненты"""
            constraints_list = []
            
            for i, comp in enumerate(self.available_components):
                if comp in constraints:
                    min_val, max_val = constraints[comp]
                    # Ограничение снизу
                    constraints_list.append(x[i] - min_val)
                    # Ограничение сверху  
                    constraints_list.append(max_val - x[i])
                else:
                    # Стандартные ограничения 0-100%
                    constraints_list.append(x[i])  # >= 0
                    constraints_list.append(100 - x[i])  # <= 100
            
            return constraints_list
        
        # Начальное приближение (равномерное распределение)
        x0 = np.ones(n_components) * (100 / n_components)
        
        # Границы переменных
        bounds = [(0, 100) for _ in range(n_components)]
        
        # Ограничения
        constraints_optim = [
            {'type': 'eq', 'fun': sum_constraint},
            {'type': 'ineq', 'fun': component_constraints}
        ]
        
        # Оптимизация
        try:
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_optim,
                options={'maxiter': max_iterations, 'disp': False}
            )
            
            if result.success:
                optimal_composition = dict(zip(self.available_components, result.x))
                optimal_value = -result.fun if maximize else result.fun
                
                # Предсказываем все свойства для оптимального состава
                all_predictions = {}
                for prop in self.predictor.models.keys():
                    all_predictions[prop] = self.predictor.predict(optimal_composition, prop)
                
                return {
                    'success': True,
                    'optimal_composition': optimal_composition,
                    'optimal_value': optimal_value,
                    'target_property': target_property,
                    'all_predictions': all_predictions,
                    'iterations': result.nit,
                    'message': 'Оптимизация завершена успешно'
                }
            else:
                return {
                    'success': False,
                    'error': f'Оптимизация не сошлась: {result.message}',
                    'optimal_composition': dict(zip(self.available_components, result.x))
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Ошибка оптимизации: {str(e)}'
            }
    
    def find_best_existing(self, data: pd.DataFrame, target_property: str, maximize: bool = True) -> Dict:
        """
        Находит лучший существующий состав из данных
        Полезно для валидации и сравнения
        """
        if target_property not in data.columns:
            return {'success': False, 'error': f'Свойство {target_property} не найдено в данных'}
        
        valid_data = data.dropna(subset=[target_property])
        if valid_data.empty:
            return {'success': False, 'error': 'Нет данных для анализа'}
        
        if maximize:
            best_idx = valid_data[target_property].idxmax()
        else:
            best_idx = valid_data[target_property].idxmin()
        
        best_row = valid_data.loc[best_idx]
        best_composition = {}
        
        # Извлекаем состав
        for comp in self.available_components:
            if comp in valid_data.columns:
                best_composition[comp] = best_row[comp]
        
        return {
            'success': True,
            'composition': best_composition,
            'value': best_row[target_property],
            'source': 'existing_data',
            'message': f'Лучший существующий состав ({"максимум" if maximize else "минимум"})'
        }

class PelletMLSystem:
    """
    Главная система ML анализа пеллет
    """
    def __init__(self, db_path: str = 'pellets_data.db'):
        self.db_path = db_path
        self.predictor = PelletPropertyPredictor()
        self.optimizer = CompositionOptimizer(self.predictor)
        self.training_data = None
    
    def load_training_data(self) -> pd.DataFrame:
        """Загружает данные для обучения из базы"""
        try:
            from database import query_db
            data = query_db(self.db_path, "measured_parameters")
            print(f"📊 Загружено данных для обучения: {len(data)} записей")
            self.training_data = data
            return data
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            return pd.DataFrame()
    
    def train_models(self, target_properties: List[str] = None) -> bool:
        """Обучает модели предсказания свойств"""
        if target_properties is None:
            target_properties = ['q', 'density', 'ad', 'kf']
        
        data = self.load_training_data()
        if data.empty:
            print("❌ Нет данных для обучения")
            return False
        
        print(f"🔬 Обучение моделей для свойств: {target_properties}")
        success = self.predictor.train(data, target_properties)
        
        if success:
            print("✅ Система ML готова к работе!")
        else:
            print("❌ Обучение моделей не удалось")
        
        return success
    
    def optimize_composition(self, target_property: str, **kwargs) -> Dict:
        """Оптимизирует состав для целевого свойства"""
        if not self.predictor.is_trained:
            return {'success': False, 'error': 'Модели не обучены'}
        
        print(f"🎯 Оптимизация состава для свойства: {target_property}")
        return self.optimizer.optimize(target_property, **kwargs)
    
    def get_system_status(self) -> Dict:
        """Возвращает статус системы"""
        status = {
            'is_trained': self.predictor.is_trained,
            'trained_models': list(self.predictor.models.keys()),
            'available_components': self.predictor.feature_names,
            'training_data_size': len(self.training_data) if self.training_data is not None else 0
        }
        
        # Добавляем метрики моделей
        model_metrics = {}
        for prop, model in self.predictor.models.items():
            # Здесь можно добавить больше метрик
            model_metrics[prop] = {
                'feature_importance': dict(zip(
                    self.predictor.feature_names, 
                    model.feature_importances_
                )) if hasattr(model, 'feature_importances_') else {}
            }
        
        status['model_metrics'] = model_metrics
        return status

# Глобальный экземпляр системы
ml_system = PelletMLSystem()