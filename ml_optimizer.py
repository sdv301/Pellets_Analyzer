# ml_optimizer.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import warnings
import re
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')


class CompositionParser:
    def __init__(self):
        self.component_patterns = {
            'Опилки': [r'опилки?', r'древесн\w*\s*опилки?'],
            'Солома': [r'солом[ауы]?', r'пшеничн\w*\s*солом'],
            'Картон': [r'картон?'],
            'Подсолнечный_жмых': [r'подсолнечн\w*\s*жмых', r'жмых\s*подсолнечн\w*'],
            'Рисовая_шелуха': [r'рисов\w*\s*шелух', r'шелух\w*\s*рисов\w*'],
            'Угольный_шлам': [r'угольн\w*\s*шлам', r'шлам\s*угольн\w*'],
            'Торф': [r'торф'],
            'Бурый_уголь': [r'бурый\s*уголь', r'уголь\s*бурый'],
            'СМС': [r'смс', r'cmc', r'с\.?м\.?с'],  # Добавлены латинские варианты
            'Пластик': [r'пластик'],
            'Древесная_мука': [r'древесн\w*\s*мук', r'мук\w*\s*древесн\w*'],
            'Щепа': [r'щеп']
        }

    def parse_composition(self, composition_text: str) -> Dict[str, float]:
        """Исправленный парсер с диагностикой"""
        if pd.isna(composition_text) or not composition_text:
            return {}
        
        original_text = str(composition_text)
        text = original_text.lower()
        
        composition_dict = {}
        found_matches = []
        
        # Шаг 1: Находим все процент-компонент пары
        main_pattern = r'(\d+(?:\.\d+)?)%\s*([^%,+]+?)(?=\s*[,+%]|$)'
        matches = re.findall(main_pattern, text)
        
        for percentage_str, comp_text in matches:
            percentage = float(percentage_str)
            comp_text = comp_text.strip()
            
            # Ищем соответствующий компонент
            matched_component = None
            for comp_name, patterns in self.component_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, comp_text):
                        matched_component = comp_name
                        found_matches.append(f"{comp_name}: {percentage}%")
                        break
                if matched_component:
                    break
            
            if matched_component:
                composition_dict[matched_component] = composition_dict.get(matched_component, 0) + percentage
        
        # Шаг 2: Обрабатываем компоненты после + (без указания процента)
        if '+' in text and composition_dict:
            plus_components = re.findall(r'\+\s*([^%+,]+)', text)
            
            if plus_components:
                # Берем процент последнего найденного компонента и делим его
                last_component = list(composition_dict.keys())[-1]
                last_percentage = composition_dict[last_component]
                shared_percentage = last_percentage / (len(plus_components) + 1)
                
                # Перераспределяем
                composition_dict[last_component] = shared_percentage
                
                for comp_text in plus_components:
                    comp_text = comp_text.strip()
                    matched_component = None
                    for comp_name, patterns in self.component_patterns.items():
                        for pattern in patterns:
                            if re.search(pattern, comp_text):
                                matched_component = comp_name
                                found_matches.append(f"{comp_name}: +{shared_percentage:.1f}%")
                                break
                        if matched_component:
                            break
                    
                    if matched_component:
                        composition_dict[matched_component] = composition_dict.get(matched_component, 0) + shared_percentage
        
        # Шаг 3: Нормализация к 100%
        total = sum(composition_dict.values())
        if total > 0:
            if abs(total - 100) > 1.0:
                for comp in composition_dict:
                    composition_dict[comp] = (composition_dict[comp] / total) * 100
            composition_dict = {k: round(v, 2) for k, v in composition_dict.items()}
        
        return composition_dict

class PelletPropertyPredictor:
    """
    ML модель предсказания свойств пеллет на основе состава
    """
    def __init__(self, ml_system=None):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_trained = False
        self.training_metrics = {}
        self.parser = CompositionParser()
        self.ml_system = ml_system  # Для доступа к linear_predict
        
        # Определяем целевые свойства из базы данных (новые колонки)
        self.target_properties_mapping = {
            'war': 'Влажность на аналитическую массу',
            'ad': 'Зольность на сухую массу', 
            'vd': 'Содержание летучих на сухую массу',
            'q': 'Теплота сгорания',
            'cd': 'Содержание углерода на сухую массу',
            'hd': 'Содержание водорода на сухую массу',
            'nd': 'Содержание азота на сухую массу',
            'sd': 'Содержание серы на сухую массу',
            'od': 'Содержание кислорода на сухую массу'
        }
        
        # Основные целевые свойства для оптимизации
        self.main_target_properties = list(self.target_properties_mapping.keys())  # ['war', 'ad', ...]
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[int]]:
        """Упрощенная версия с базовой диагностикой"""
        if 'composition' not in data.columns:
            print("❌ Колонка 'composition' не найдена")
            return np.array([]), [], []
        
        print(f"🔍 Анализ {len(data)} составов...")
        
        all_components = set()
        composition_data = []
        valid_indices = []
        
        for idx, row in data.iterrows():
            composition_dict = self.parser.parse_composition(row['composition'])
            if composition_dict:
                all_components.update(composition_dict.keys())
                composition_data.append(composition_dict)
                valid_indices.append(idx)
        
        component_list = sorted(list(all_components))
        
        if not component_list:
            return np.array([]), [], []
        
        print(f"📋 Найдено {len(component_list)} компонентов: {component_list}")
        
        # Создаем матрицу признаков
        X = []
        final_valid_indices = []
        
        for i, comp_dict in enumerate(composition_data):
            row = [comp_dict.get(comp, 0.0) for comp in component_list]
            total = sum(row)
            if total > 10:  # Минимум 10%
                # Нормализуем к 100%
                row = [(val / total) * 100 for val in row]
                X.append(row)
                final_valid_indices.append(valid_indices[i])
        
        if not X:
            return np.array([]), [], []
        
        X = np.array(X)
        print(f"📊 Финальная матрица: {X.shape}")
        
        return X, component_list, final_valid_indices
    
    def train(self, data: pd.DataFrame, target_properties: List[str], algorithm: str = 'gradient_boosting') -> bool:
        """Исправленная версия с совместимостью для фронтенда"""
        X, feature_names, valid_indices = self.prepare_features(data)
        if len(X) == 0:
            return False
        
        self.feature_names = feature_names
        trained_count = 0
        
        print(f"📊 Обучение моделей для {len(X)} samples, {len(feature_names)} features")
        
        for prop in target_properties:
            y = data[prop].iloc[valid_indices]
            
            # Проверяем достаточно ли данных
            valid_y = y.dropna()
            if len(valid_y) < 8:
                print(f"⚠️ Пропуск {prop}: недостаточно данных ({len(valid_y)} < 8)")
                continue
            
            print(f"🎯 Обучение модели для {prop} ({len(valid_y)} samples)")
            
            # Удаляем NaN
            valid_mask = ~y.isna()
            X_prop = X[valid_mask]
            y_prop = y[valid_mask]
            
            # Для совместимости с фронтендом используем простой подход
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_prop)
            
            # Простая модель для избежания переобучения
            if algorithm == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
            else:  # gradient_boosting
                model = GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42
                )
            
            # Обучаем на всех данных (как было раньше)
            model.fit(X_scaled, y_prop)
            
            # Предсказания для расчета метрик
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y_prop, y_pred)
            mae = mean_absolute_error(y_prop, y_pred)
            
            # Кросс-валидация для оценки
            cv_scores = cross_val_score(model, X_scaled, y_prop, cv=min(5, len(y_prop)), scoring='r2')
            avg_cv_r2 = np.mean(cv_scores)
            
            self.models[prop] = model
            self.scalers[prop] = scaler
            
            # ВОЗВРАЩАЕМ СТАРУЮ СТРУКТУРУ ДЛЯ СОВМЕСТИМОСТИ
            self.training_metrics[prop] = {
                'r2_score': r2,
                'mae': mae,
                'cv_r2': avg_cv_r2,
                # Добавляем feature_importance в корень для совместимости
                'feature_importance': {}
            }
            
            # Feature importance (отдельно для совместимости)
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                total = sum(feature_importance)
                normalized = feature_importance / total if total != 0 else feature_importance
                # Сохраняем в двух местах для совместимости
                self.training_metrics[prop]['feature_importance'] = dict(zip(feature_names, normalized))
            
            print(f"   ✅ {prop}: R²={r2:.3f}, MAE={mae:.3f}, CV R²={avg_cv_r2:.3f}")
            trained_count += 1
        
        self.is_trained = trained_count > 0
        
        if self.is_trained:
            print(f"✅ Обучено {trained_count} моделей")
        else:
            print("❌ Не удалось обучить ни одной модели")
        
        return self.is_trained
    
    def predict(self, composition: Dict[str, float], target_property: str) -> Optional[float]:
        """Предсказывает свойство: ML если обучено, иначе линейное из компонентов"""
        if target_property not in self.models:
            if self.ml_system:
                return self.ml_system.linear_predict(composition, target_property)
            return None
        
        X = self.prepare_composition_for_prediction(composition)
        scaler = self.scalers[target_property]
        X_scaled = scaler.transform([X])
        model = self.models[target_property]
        return model.predict(X_scaled)[0]
    
    def prepare_composition_for_prediction(self, composition: Dict[str, float]) -> np.ndarray:
        """Подготавливает состав для предсказания"""
        total = sum(composition.values())
        if total != 100:
            composition = {k: (v / total) * 100 for k, v in composition.items()}
        
        X = [composition.get(feature, 0.0) for feature in self.feature_names]
        return np.array(X)
    
    def get_feature_importance(self, target_property: str) -> Dict[str, float]:
        """Возвращает важность признаков для свойства"""
        if target_property not in self.training_metrics:
            return {}
        return self.training_metrics[target_property].get('feature_importance', {})

class MLCompositionOptimizer:
    """Оптимизатор составов на основе ML предсказаний"""
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.optimization_history = []
    
    def optimize_composition(self, target_property: str, maximize: bool = True, constraints: Dict[str, Tuple[float, float]] = None) -> Dict:
        """Оптимизирует состав с проверкой совместимости ограничений"""
        if target_property not in self.predictor.models:
            return {'success': False, 'error': f'Модель для {target_property} не обучена'}
        
        feature_names = self.predictor.feature_names
        n_features = len(feature_names)
        
        # Проверяем совместимость ограничений
        if constraints:
            min_total = 0.0
            max_total = 0.0
            
            for comp, (min_val, max_val) in constraints.items():
                if comp in feature_names:
                    min_total += min_val
                    max_total += max_val
            
            print(f"🔍 Проверка ограничений: min_total={min_total:.1f}%, max_total={max_total:.1f}%")
            
            # Если минимальная сумма > 100% - ограничения несовместимы
            if min_total > 100.0:
                return {
                    'success': False, 
                    'error': f'Ограничения несовместимы: минимальная сумма {min_total:.1f}% > 100%'
                }
            
            # Если максимальная сумма < 100% - тоже несовместимы
            if max_total < 100.0:
                return {
                    'success': False, 
                    'error': f'Ограничения несовместимы: максимальная сумма {max_total:.1f}% < 100%'
                }
        
        # Начальный состав - равномерное распределение
        initial_composition = np.full(n_features, 1.0 / n_features)
        
        def objective(composition):
            try:
                comp_dict = dict(zip(feature_names, composition * 100))
                pred = self.predictor.predict(comp_dict, target_property)
                if pred is None:
                    return 1e6 if maximize else -1e6
                return -pred if maximize else pred
            except:
                return 1e6 if maximize else -1e6
        
        # Ограничение: сумма = 100%
        def sum_constraint(composition):
            return sum(composition) - 1.0
        
        cons = [{'type': 'eq', 'fun': sum_constraint}]
        
        # Ограничения на компоненты
        bounds = [(0.0, 1.0) for _ in range(n_features)]
        if constraints:
            for comp, (min_val, max_val) in constraints.items():
                if comp in feature_names:
                    idx = feature_names.index(comp)
                    # Преобразуем проценты в доли
                    bounds[idx] = (max(min_val / 100.0, 0.0), min(max_val / 100.0, 1.0))
        
        # Пробуем разные методы оптимизации
        methods = ['SLSQP', 'trust-constr']
        best_result = None
        best_value = -np.inf if maximize else np.inf
        
        for method in methods:
            try:
                result = minimize(
                    objective, 
                    initial_composition, 
                    method=method, 
                    bounds=bounds, 
                    constraints=cons, 
                    options={'maxiter': 500, 'disp': False}
                )
                
                if result.success:
                    current_value = -result.fun if maximize else result.fun
                    if (maximize and current_value > best_value) or (not maximize and current_value < best_value):
                        best_result = result
                        best_value = current_value
            except Exception as e:
                print(f"⚠️ Метод {method} не сработал: {e}")
                continue
        
        # Если ни один метод не сработал, пробуем без ограничений
        if best_result is None:
            print("🔄 Пробую оптимизацию без ограничений...")
            try:
                result = minimize(
                    objective, 
                    initial_composition, 
                    method='SLSQP', 
                    constraints=cons, 
                    options={'maxiter': 500, 'disp': False}
                )
                if result.success:
                    best_result = result
            except:
                pass
        
        if best_result is None:
            return {
                'success': False, 
                'error': 'Не удалось найти решение. Попробуйте ослабить ограничения.'
            }
        
        # Формируем результат
        optimal_composition = dict(zip(feature_names, best_result.x * 100))
        
        # Фильтруем нулевые и нормализуем
        optimal_composition = {k: round(v, 2) for k, v in optimal_composition.items() if v > 0.1}
        total = sum(optimal_composition.values())
        
        if total > 0 and abs(total - 100) > 0.1:
            optimal_composition = {k: round((v / total) * 100, 2) for k, v in optimal_composition.items()}
        
        optimal_value = -best_result.fun if maximize else best_result.fun
        
        # Проверяем соблюдение ограничений
        if constraints:
            violations = []
            for comp, (min_val, max_val) in constraints.items():
                if comp in optimal_composition:
                    value = optimal_composition[comp]
                    if value < min_val - 0.1 or value > max_val + 0.1:
                        violations.append(f"{comp}: {value:.1f}% (требуется {min_val:.1f}-{max_val:.1f}%)")
            
            if violations:
                return {
                    'success': False,
                    'error': f'Не удалось соблюсти ограничения: {", ".join(violations)}'
                }
        
        # Генерация сообщения
        display_name = self.predictor.target_properties_mapping.get(target_property, target_property)
        direction = "максимизации" if maximize else "минимизации"
        comp_text = ", ".join([f"{k}: {v:.1f}%" for k, v in optimal_composition.items()])
        
        message = (f"Оптимальный состав для {direction} {display_name}: {comp_text}. "
                f"Ожидаемое значение: {optimal_value:.2f}.")
        
        print(f"✅ {message}")
        
        self.optimization_history.append({
            'target_property': target_property,
            'optimal_composition': optimal_composition,
            'optimal_value': optimal_value,
            'message': message
        })
        
        return {
            'success': True,
            'optimal_composition': optimal_composition,
            'optimal_value': optimal_value,
            'message': message
        }

    def validate_constraints(self, constraints: Dict[str, Tuple[float, float]], feature_names: List[str]) -> Tuple[bool, str]:
        """Проверяет совместимость ограничений"""
        if not constraints:
            return True, ""
        
        min_total = 0.0
        max_total = 0.0
        invalid_components = []
        
        for comp, (min_val, max_val) in constraints.items():
            if comp not in feature_names:
                invalid_components.append(comp)
                continue
                
            if min_val < 0 or max_val > 100 or min_val > max_val:
                return False, f"Некорректные ограничения для {comp}: {min_val}-{max_val}%"
                
            min_total += min_val
            max_total += max_val
        
        if invalid_components:
            return False, f"Неизвестные компоненты: {', '.join(invalid_components)}"
        
        if min_total > 100.0:
            return False, f"Минимальная сумма {min_total:.1f}% > 100%"
            
        if max_total < 100.0:
            return False, f"Максимальная сумма {max_total:.1f}% < 100%"
        
        return True, ""

class PelletMLSystem:
    """
    Главная система ML анализа и оптимизации пеллет
    """
    def __init__(self, db_path: str = 'pellets_data.db'):
        self.db_path = db_path
        self.predictor = PelletPropertyPredictor(self)
        self.ml_optimizer = MLCompositionOptimizer(self.predictor)
        self.training_data = self.load_training_data()
        self.components = self.load_components()
    
    def load_components(self) -> pd.DataFrame:
        """Загружает свойства компонентов из БД"""
        try:
            from database import query_db
            components = query_db(self.db_path, "components")
            if components.empty:
                print("⚠️ Таблица components пуста")
            else:
                print(f"📊 Загружено компонентов: {len(components)}")
            return components
        except Exception as e:
            print(f"❌ Ошибка загрузки компонентов: {e}")
            return pd.DataFrame()
    
    def linear_predict(self, composition: Dict[str, float], target_property: str) -> Optional[float]:
        """Линейное предсказание свойства как взвешенной суммы свойств компонентов"""
        if self.components.empty:
            return None
        value = 0.0
        total_weight = 0.0
        for comp, percent in composition.items():
            if comp in self.components['component'].values:
                row = self.components[self.components['component'] == comp]
                if target_property in row.columns and not pd.isna(row[target_property].iloc[0]):
                    value += (percent / 100.0) * row[target_property].iloc[0]
                    total_weight += percent
        if total_weight > 0:
            return value * (100.0 / total_weight)  # Нормализуем если total !=100
        return None
    
    def load_training_data(self) -> pd.DataFrame:
        """Загружает тренировочные данные из БД"""
        try:
            from database import query_db
            training_data = query_db(self.db_path, "measured_parameters")
            if training_data.empty:
                print("⚠️ Таблица measured_parameters пуста")
            else:
                print(f"📊 Загружено тренировочных данных: {len(training_data)} записей")
            return training_data
        except Exception as e:
            print(f"❌ Ошибка загрузки тренировочных данных: {e}")
            return pd.DataFrame()
    
    def train_models(self, target_properties: List[str] = None, algorithm: str = 'random_forest') -> Dict:
        """Обучает ML модели для предсказания свойств"""
        if self.training_data.empty:
            print("❌ Нет данных для ML обучения")
            return {'success': False, 'error': 'Нет данных для обучения'}
        
        if target_properties is None:
            target_properties = self.predictor.main_target_properties
        
        success = self.predictor.train(self.training_data, target_properties, algorithm)
        
        if success:
            print("✅ ML система готова к работе!")
            print("🤖 ML Agent может создавать новые оптимальные составы")
            status = self.get_ml_system_status()
            return {
                'success': True,
                'message': 'ML система успешно обучена!',
                'status': status,
                'trained_count': len(status['trained_models']),
                'metrics': {prop: status['model_metrics'][prop] for prop in status['trained_models']}
            }
        else:
            print("❌ Обучение ML моделей не удалось")
            return {'success': False, 'error': 'Обучение не удалось'}
    
    def optimize_composition(self, target_property: str, maximize: bool = True, constraints: Dict[str, Tuple[float, float]] = None) -> Dict:
        return self.ml_optimizer.optimize_composition(target_property, maximize, constraints)
    
    def get_ml_system_status(self) -> Dict:
        status = {
            'is_trained': self.predictor.is_trained,
            'trained_models': list(self.predictor.models.keys()),
            'available_components': self.predictor.feature_names,
            'training_data_size': len(self.training_data) if not self.training_data.empty else 0,
            'ml_optimizations_count': len(self.ml_optimizer.optimization_history)
        }
        
        model_metrics = {}
        for prop in self.predictor.models.keys():
            # СОВМЕСТИМАЯ СТРУКТУРА ДАННЫХ
            model_metrics[prop] = {
                'feature_importance': self.predictor.training_metrics.get(prop, {}).get('feature_importance', {}),
                'training_metrics': {
                    'r2_score': self.predictor.training_metrics.get(prop, {}).get('r2_score', 0),
                    'mae': self.predictor.training_metrics.get(prop, {}).get('mae', 0),
                    'cv_r2': self.predictor.training_metrics.get(prop, {}).get('cv_r2', 0)
                },
                'display_name': self.predictor.target_properties_mapping.get(prop, prop)
            }
        
        status['model_metrics'] = model_metrics
        status['target_properties_mapping'] = self.predictor.target_properties_mapping
        
        return status

# Глобальный экземпляр ML системы
ml_system = PelletMLSystem()

def get_ml_system():
    return ml_system