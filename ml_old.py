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
import os
import joblib
from concurrent.futures import ThreadPoolExecutor
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
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
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
    
    def _train_single_property(self, prop: str, X_prop: np.ndarray, y_prop: pd.Series, algorithm: str, feature_names: List[str]) -> Tuple[str, Dict]:
        """Обучает модель для одного свойства с использованием GridSearchCV"""
        from sklearn.linear_model import Ridge # <-- Импортируем линейную модель

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_prop)
        
        # Свойства, которые подчиняются линейным законам смешивания
        additive_properties = ['q', 'ad', 'war', 'density', 'vd', 'cd', 'hd', 'nd', 'sd', 'od']
        
        if prop in additive_properties:
            # Для тепла и плотности принудительно используем Линейную Регрессию (исключает "парадоксы")
            base_model = Ridge(alpha=1.0)
            param_grid = {'alpha': [0.1, 1.0, 10.0]}
        elif algorithm == 'random_forest':
            base_model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5]
            }
        else:  # gradient_boosting
            base_model = GradientBoostingRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
            
        # Подбор параметров
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=min(3, len(y_prop)), 
            scoring='r2', 
            n_jobs=-1
        )
        grid_search.fit(X_scaled, y_prop)
        
        model = grid_search.best_estimator_
        
        # Предсказания для расчета метрик
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y_prop, y_pred)
        mae = mean_absolute_error(y_prop, y_pred)
        
        cv_scores = cross_val_score(model, X_scaled, y_prop, cv=min(3, len(y_prop)), scoring='r2')
        avg_cv_r2 = np.mean(cv_scores)
        
        metrics = {
            'model': model,
            'scaler': scaler,
            'r2_score': float(r2),
            'mae': float(mae),
            'cv_r2': float(avg_cv_r2),
            'feature_importance': {}
        }
        
        # Сохранение важности компонентов в зависимости от типа модели
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            total = sum(importances)
            normalized = importances / total if total != 0 else importances
            metrics['feature_importance'] = {k: float(v) for k, v in zip(feature_names, normalized)}
        elif hasattr(model, 'coef_'):
            # Для линейной регрессии берем коэффициенты
            importances = np.abs(model.coef_)
            total = sum(importances)
            normalized = importances / total if total != 0 else importances
            metrics['feature_importance'] = {k: float(v) for k, v in zip(feature_names, normalized)}
            
        return prop, metrics

    def train(self, data: pd.DataFrame, target_properties: List[str], algorithm: str = 'gradient_boosting', selected_features: List[str] = None) -> Dict:
        """Многопоточное обучение моделей с AutoML тюнингом"""
        X, feature_names, valid_indices = self.prepare_features(data)
        if len(X) == 0:
            return {'success': False, 'trained_count': 0, 'skipped': [], 'error': 'Нет данных для формирования признаков'}
        
        # Если переданы конкретные фичи, фильтруем
        if selected_features and len(selected_features) > 0:
            valid_selected = [f for f in selected_features if f in feature_names]
            if not valid_selected:
                 return {'success': False, 'trained_count': 0, 'skipped': [], 'error': 'Ни один из выбранных компонентов не найден в данных'}
            
            # Находим индексы выбранных фич в исходной матрице
            feat_indices = [feature_names.index(f) for f in valid_selected]
            X = X[:, feat_indices]
            current_feature_names = valid_selected
        else:
            current_feature_names = feature_names

        self.feature_names = current_feature_names
        trained_count = 0
        skipped_props = []
        
        print(f"📊 Запуск многопоточного обучения ({len(target_properties)} свойств)...")
        
        # Подготовка задач для ThreadPoolExecutor
        tasks = []
        for prop in target_properties:
            y = data[prop].iloc[valid_indices]
            valid_y = y.dropna()
            
            if len(valid_y) < 8:
                reason = f"недостаточно данных ({len(valid_y)} из 8)"
                skipped_props.append({'property': prop, 'reason': reason})
                continue
                
            valid_mask = ~y.isna()
            X_prop = X[valid_mask]
            y_prop = y[valid_mask]
            tasks.append((prop, X_prop, y_prop, algorithm, current_feature_names))

        # Выполнение обучения в пуле потоков
        with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as executor:
            future_to_prop = {executor.submit(self._train_single_property, *task): task[0] for task in tasks}
            for future in future_to_prop:
                prop = future_to_prop[future]
                try:
                    prop, metrics = future.result()
                    self.models[prop] = metrics['model']
                    self.scalers[prop] = metrics['scaler']
                    self.training_metrics[prop] = {
                        'r2_score': metrics['r2_score'],
                        'mae': metrics['mae'],
                        'cv_r2': metrics['cv_r2'],
                        'feature_importance': metrics['feature_importance']
                    }
                    trained_count += 1
                    print(f"   ✅ {prop}: R²={metrics['r2_score']:.3f}, CV R²={metrics['cv_r2']:.3f}")
                except Exception as e:
                    print(f"   ❌ Ошибка обучения {prop}: {e}")
                    skipped_props.append({'property': prop, 'reason': str(e)})

        self.is_trained = trained_count > 0
        if self.is_trained:
            self.save_models() # Сохраняем на диск после обучения
            
        return {
            'success': self.is_trained,
            'trained_count': trained_count,
            'skipped': skipped_props
        }

    def save_models(self):
        """Сохранение моделей на диск"""
        try:
            data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_names': self.feature_names,
                'training_metrics': self.training_metrics
            }
            file_path = os.path.join(self.models_dir, 'pellet_models.joblib')
            joblib.dump(data, file_path)
            print(f"💾 Модели сохранены в {file_path}")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения моделей: {e}")

    def load_models(self) -> bool:
        """Загрузка моделей с диска"""
        try:
            file_path = os.path.join(self.models_dir, 'pellet_models.joblib')
            if os.path.exists(file_path):
                data = joblib.load(file_path)
                self.models = data['models']
                self.scalers = data['scalers']
                self.feature_names = data['feature_names']
                self.training_metrics = data['training_metrics']
                self.is_trained = True
                print(f"🧠 Модели загружены из {file_path}")
                return True
        except Exception as e:
            print(f"⚠️ Ошибка загрузки моделей: {e}")
        return False
    
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
    
    def optimize_composition(self, target_property: str, maximize: bool = True, constraints: Dict[str, Tuple[float, float]] = None, available_components: List[str] = None) -> Dict:
        """Оптимизирует состав с проверкой совместимости ограничений и выбором лучших компонентов"""
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
        
        # Фильтрация по доступным компонентам
        available_indices = []
        if available_components:
            for i, feat in enumerate(feature_names):
                if feat in available_components:
                    available_indices.append(i)
        else:
            available_indices = list(range(n_features))

        if not available_indices:
             return {
                 'success': False,
                 'error': 'Ни один из доступных компонентов не распознан моделью'
             }
             
        from scipy.optimize import differential_evolution

        # Начальный состав - равномерное распределение по *доступным*
        initial_composition = np.zeros(n_features)
        for idx in available_indices:
            initial_composition[idx] = 1.0 / len(available_indices)
            
        # Ограничения на компоненты (для генерации)
        bounds_percent = [(0.0, 100.0) for _ in range(n_features)]
        
        # Если заданы доступные компоненты, недоступные жестко нулируем
        if available_components:
             for i in range(n_features):
                 if i not in available_indices:
                     bounds_percent[i] = (0.0, 0.0)
                     
        if constraints:
            for comp, (min_val, max_val) in constraints.items():
                if comp in feature_names:
                    idx = feature_names.index(comp)
                    new_min = max(min_val, bounds_percent[idx][0])
                    new_max = min(max_val, bounds_percent[idx][1])
                    bounds_percent[idx] = (new_min, new_max)
        
        def objective(composition):
            try:
                total = sum(composition)
                if total <= 1e-6: return 1e6 if maximize else -1e6
                normalized = (composition / total) * 100.0
                
                comp_dict = {f_name: normalized[i] for i, f_name in enumerate(feature_names)}
                pred = self.predictor.predict(comp_dict, target_property)
                
                if pred is None: return 1e6 if maximize else -1e6
                
                # --- НОВЫЙ БЛОК: ФИЗИЧЕСКИЙ ОГРАНИЧИТЕЛЬ ---
                # Если мы предсказываем аддитивное свойство (например, теплоту q)
                if target_property in ['q', 'ad', 'war', 'density']:
                    max_possible = max([self.predictor.predict({f: 100}, target_property) for f in comp_dict.keys() if comp_dict[f] > 0])
                    # Если ML "нафантазировал", что смесь лучше чистого лучшего сырья - жестоко штрафуем
                    if maximize and pred > max_possible:
                        pred = max_possible - 0.1 # Срезаем пик
                # -------------------------------------------

                score = -pred if maximize else pred
                
                penalty = 0.0
                for i, val in enumerate(normalized):
                    b_min, b_max = bounds_percent[i]
                    if val < b_min - 0.1: penalty += (b_min - val) * 1000
                    elif val > b_max + 0.1: penalty += (val - b_max) * 1000
                        
                return score + penalty
            except:
                return 1e6 if maximize else -1e6
        
        print("🔄 Пробую глобальную оптимизацию Дифференциальной Эволюцией (Genetic Algorithm)...")
        # Для DE границы поиска можно задать как 0..1 для доступных
        de_bounds = [(0.0, 1.0) if i in available_indices else (0.0, 0.0) for i in range(n_features)]
        
        try:
            result = differential_evolution(
                objective,
                de_bounds,
                maxiter=50,
                popsize=15,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                disp=False
            )
        except Exception as e:
            print(f"❌ Ошибка в differential_evolution: {e}")
            return {
                'success': False, 
                'error': f'Ошибка оптимизатора: {str(e)}'
            }
        
        if not result.success:
            return {
                'success': False, 
                'error': 'Не удалось найти решение. Попробуйте ослабить ограничения.'
            }
        
        # Формируем результат
        best_comp = result.x
        total_w = sum(best_comp)
        if total_w > 0:
            final_comp = (best_comp / total_w) * 100.0
        else:
            final_comp = best_comp
            
        optimal_composition = dict(zip(feature_names, final_comp))
        
        # Фильтруем нулевые и нормализуем
        optimal_composition = {k: round(v, 2) for k, v in optimal_composition.items() if v > 0.1}
        total = sum(optimal_composition.values())
        
        if total > 0 and abs(total - 100) > 0.1:
            optimal_composition = {k: round(float(v / total) * 100, 2) for k, v in optimal_composition.items()}
        optimal_value = float(-result.fun) if maximize else float(result.fun)
        
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
        comp_text = ", ".join([f"{v:.1f}% {k}" for k, v in optimal_composition.items()])
        
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

    def compare_compositions(self, baseline_composition: Dict[str, float], optimized_composition: Dict[str, float], target_property: str, maximize: bool = True) -> Dict:
        """Сравнивает два состава по целевому свойству и возвращает разницу"""
        if target_property not in self.predictor.models:
            return {'success': False, 'error': f'Модель для {target_property} не обучена'}
            
        base_val = self.predictor.predict(baseline_composition, target_property)
        opt_val = self.predictor.predict(optimized_composition, target_property)
        
        if base_val is None or opt_val is None:
            return {'success': False, 'error': 'Не удалось предсказать свойства для одного из составов'}
            
        diff_abs = opt_val - base_val
        
        # Процентное улучшение (относительно базового)
        if base_val != 0:
            diff_pct = (diff_abs / abs(base_val)) * 100
            
            # Если мы минимизируем, то отрицательная разница - это улучшение (положительный процент)
            if not maximize:
                diff_pct = -diff_pct
        else:
            diff_pct = float('inf') if (maximize and diff_abs > 0) or (not maximize and diff_abs < 0) else 0.0
            
        is_better = (maximize and opt_val > base_val) or (not maximize and opt_val < base_val)
        
        direction_word = "лучше" if is_better else "хуже"
        if diff_pct == 0:
            message = "Оптимизированный состав показывает такой же результат как базовый."
        else:
            message = f"Оптимизированный состав на {abs(diff_pct):.1f}% {direction_word} базового (База: {base_val:.2f} -> Оптим: {opt_val:.2f})"
            
        return {
            'success': True,
            'baseline_value': base_val,
            'optimized_value': opt_val,
            'difference_absolute': diff_abs,
            'improvement_percent': diff_pct,
            'is_better': is_better,
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
        
        # Сначала пробуем загрузить готовые модели с диска
        models_loaded = self.predictor.load_models()
        
        self.components = self.load_components()
        self.training_data = self.load_training_data()
        
        if not models_loaded:
            print("💡 Модели не найдены на диске, требуется обучение.")

    def reload_data(self):
        """Принудительная перезагрузка данных из БД, чтобы учесть новые загрузки Excel"""
        try:
            self.components = self.load_components()
            self.training_data = self.load_training_data()
            print(f"🔄 Данные ML системы принудительно обновлены (Компонентов: {len(self.components)}, Образцов: {len(self.training_data)})")
        except Exception as e:
            print(f"❌ Ошибка обновления данных ML системы: {e}")

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
        """Загружает тренировочные данные из БД, включая ML оптимизации"""
        try:
            from database import query_db, get_ml_optimizations
            
            # Основные данные
            training_data = query_db(self.db_path, "measured_parameters")
            
            # Добавляем успешные ML оптимизации
            ml_optimizations = get_ml_optimizations(self.db_path, limit=100)
            
            if not ml_optimizations.empty:
                print(f"📊 Добавляю {len(ml_optimizations)} ML оптимизаций к тренировочным данным")
                
                # Все целевые свойства для маппинга
                all_target_props = list(self.predictor.target_properties_mapping.keys())
                
                for _, opt in ml_optimizations.iterrows():
                    composition_text = ", ".join([f"{v}% {k}" for k, v in opt['optimal_composition'].items()])
                    
                    # Создаем строку со всеми свойствами = None, кроме целевого
                    new_row = {'composition': composition_text}
                    for prop in all_target_props:
                        new_row[prop] = opt['optimal_value'] if opt['target_property'] == prop else None
                    
                    # Дополняем линейными предсказаниями из компонентов
                    if not self.components.empty:
                        comp_dict = opt['optimal_composition']
                        for prop in all_target_props:
                            if new_row.get(prop) is None:
                                linear_val = self.linear_predict(comp_dict, prop)
                                if linear_val is not None:
                                    new_row[prop] = linear_val
                    
                    training_data = pd.concat([training_data, pd.DataFrame([new_row])], ignore_index=True)
            
            print(f"📊 Итоговый размер тренировочных данных: {len(training_data)} записей")
            return training_data
            
        except Exception as e:
            print(f"❌ Ошибка загрузки тренировочных данных: {e}")
            return pd.DataFrame()
 
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
    def load_saved_models(self):
        """Прокси-метод для загрузки моделей через предиктор"""
        return self.predictor.load_models()

    def train_models(self, target_properties: List[str] = None, algorithm: str = 'gradient_boosting', selected_features: List[str] = None) -> Dict:
        """Обучает ML модели и сохраняет результаты в базу"""
        if self.training_data.empty:
            print("❌ Нет данных для ML обучения")
            return {'success': False, 'error': 'Нет данных для обучения'}
        
        if target_properties is None:
            target_properties = self.predictor.main_target_properties
        
        train_result = self.predictor.train(self.training_data, target_properties, algorithm, selected_features)
        success = train_result.get('success', False)
        skipped = train_result.get('skipped', [])
        
        if success:
            print("✅ ML система готова к работе!")
            
            # СОХРАНЯЕМ МЕТРИКИ МОДЕЛЕЙ В БАЗУ
            try:
                from database import insert_ml_model_metrics
                
                for prop in self.predictor.models.keys():
                    metrics = self.predictor.training_metrics.get(prop, {})
                    metrics_data = {
                        'target_property': prop,
                        'algorithm': algorithm,
                        'r2_score': metrics.get('r2_score'),
                        'mae': metrics.get('mae'),
                        'cv_r2': metrics.get('cv_r2'),
                        'feature_importance': metrics.get('feature_importance', {}),
                        'training_data_size': len(self.training_data)
                    }
                    insert_ml_model_metrics(self.db_path, metrics_data)
                    print(f"💾 Сохранены метрики для {prop} в базу")
                    
            except Exception as e:
                print(f"⚠️ Не удалось сохранить метрики в базу: {e}")
            
            status = self.get_ml_system_status()
            return {
                'success': True,
                'message': 'ML система успешно обучена!',
                'status': status,
                'trained_count': len(status['trained_models']),
                'metrics': {prop: status['model_metrics'].get(prop) for prop in status['trained_models']},
                'skipped': skipped
            }
        else:
            print("❌ Обучение ML моделей не удалось")
            return {
                'success': False, 
                'error': train_result.get('error', 'Обучение не удалось из-за нехватки данных'),
                'skipped': skipped
            }

    def optimize_composition(self, target_property: str, maximize: bool = True, constraints: Dict[str, Tuple[float, float]] = None, available_components: List[str] = None) -> Dict:
        """Оптимизирует состав и сохраняет результат в базу.
        Использует данные компонентов для стартовой точки оптимизации."""
        
        # Если ML модель не обучена, пробуем оптимизацию на основе компонентов
        if target_property not in self.predictor.models:
            if self.components.empty:
                return {'success': False, 'error': f'Модель для {target_property} не обучена и нет данных компонентов'}
            # Пробуем линейную оптимизацию через компоненты
            linear_result = self._optimize_from_components(target_property, maximize, constraints, available_components)
            if linear_result.get('success'):
                linear_result['message'] = '(линейная оптимизация по компонентам) ' + linear_result.get('message', '')
            return linear_result
        
        result = self.ml_optimizer.optimize_composition(target_property, maximize, constraints, available_components)
        
        if result.get('success'):
            try:
                from database import insert_ml_optimization, add_ml_optimization_to_training_data
                
                optimization_data = {
                    'target_property': target_property,
                    'maximize': maximize,
                    'optimal_composition': result['optimal_composition'],
                    'optimal_value': result['optimal_value'],
                    'constraints': constraints or {},
                    'algorithm': 'gradient_boosting',
                    'model_metrics': self.predictor.training_metrics.get(target_property, {})
                }
                
                insert_ml_optimization(self.db_path, optimization_data)
                add_ml_optimization_to_training_data(self.db_path, optimization_data)
                
                # Перезагружаем тренировочные данные
                self.training_data = self.load_training_data()
                
            except Exception as e:
                print(f"⚠️ Не удалось сохранить оптимизацию в базу: {e}")
        
        return result
    
    def _optimize_from_components(self, target_property: str, maximize: bool = True, constraints: Dict[str, Tuple[float, float]] = None, available_components: List[str] = None) -> Dict:
        """Оптимизация состава через линейное предсказание на основе данных компонентов."""
        if self.components.empty or target_property not in self.components.columns:
            return {'success': False, 'error': f'Нет данных компонентов для свойства {target_property}'}
        
        # Получаем компоненты с известными значениями свойства
        valid_comps = self.components.dropna(subset=[target_property])
        if valid_comps.empty:
            return {'success': False, 'error': f'Нет данных о {target_property} для компонентов'}
        
        component_names = valid_comps['component'].tolist()
        component_values = valid_comps[target_property].tolist()
        n = len(component_names)
        
        from scipy.optimize import minimize as scipy_minimize
        
        def objective(fractions):
            val = sum(f * v for f, v in zip(fractions, component_values))
            return -val if maximize else val
        
        cons = [{'type': 'eq', 'fun': lambda x: sum(x) - 1.0}]
        bounds = [(0.0, 1.0)] * n
        
        if available_components:
            for i, comp_name in enumerate(component_names):
                if comp_name not in available_components:
                    bounds[i] = (0.0, 0.0)
                    
        if constraints:
            for comp, (min_val, max_val) in constraints.items():
                if comp in component_names:
                    idx = component_names.index(comp)
                    new_min = max(min_val / 100.0, bounds[idx][0])
                    new_max = min(max_val / 100.0, bounds[idx][1])
                    bounds[idx] = (new_min, new_max)
        
        # Начальная точка с учетом доступности
        if available_components:
            available_indices = [i for i, name in enumerate(component_names) if name in available_components]
            if not available_indices:
                return {'success': False, 'error': 'Нет данных ни по одному из доступных компонентов'}
            x0 = np.zeros(n)
            for idx in available_indices:
                x0[idx] = 1.0 / len(available_indices)
        else:
            x0 = np.full(n, 1.0 / n)
        
        try:
            result = scipy_minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
            if result.success:
                optimal_comp = {name: round(frac * 100, 2) for name, frac in zip(component_names, result.x) if frac > 0.001}
                total = sum(optimal_comp.values())
                if total > 0 and abs(total - 100) > 0.1:
                    optimal_comp = {k: round((v / total) * 100, 2) for k, v in optimal_comp.items()}
                
                optimal_value = -result.fun if maximize else result.fun
                display_name = self.predictor.target_properties_mapping.get(target_property, target_property)
                direction = "максимизации" if maximize else "минимизации"
                comp_text = ", ".join([f"{v:.1f}% {k}" for k, v in optimal_comp.items()])
                message = f"Оптимальный состав для {direction} {display_name}: {comp_text}. Ожидаемое значение: {optimal_value:.2f}."
                
                return {
                    'success': True,
                    'optimal_composition': optimal_comp,
                    'optimal_value': optimal_value,
                    'message': message
                }
        except Exception as e:
            pass
        
        return {'success': False, 'error': 'Не удалось оптимизировать состав по данным компонентов'}

    def augment_database(self, variations_count: int = 3, confidence_interval: float = 5.0) -> Dict:
        """
        Масштабирует экспериментальную базу данных за счет добавления синтетических 
        образцов в пределах доверительного интервала.
        """
        try:
            from database import query_db, insert_data
            
            # Получаем текущие данные
            measured_data = query_db(self.db_path, "measured_parameters")
            if measured_data.empty:
                return {'success': False, 'error': 'База данных пуста, нет данных для аугментации'}
                
            print(f"🔄 Запуск аугментации данных: {len(measured_data)} базовых образцов, {variations_count} вариаций, интервал {confidence_interval}%")
            
            import random
            ci = confidence_interval / 100.0  # Например 0.05
            prop_ci = max(0.01, ci / 2)  # Погрешность для свойств делаем чуть меньше (например 2.5%)
            
            synthetic_rows = []
            
            # Список всех возможных свойств в measured_parameters (исключая composition)
            all_props = ['density', 'kf', 'kt', 'h', 'mass_loss', 'tign', 'tb', 'tau_d1', 'tau_d2', 'tau_b', 'co2', 'co', 'so2', 'nox', 'q', 'ad']
            
            for _, row in measured_data.iterrows():
                comp_text = row.get('composition')
                if not comp_text or pd.isna(comp_text):
                    continue
                    
                comp_dict = self.predictor.parser.parse_composition(comp_text)
                if not comp_dict:
                    continue
                    
                # Генерация вариаций
                for _ in range(variations_count):
                    new_comp = {}
                    # Вносим шум в компоненты
                    for comp, val in comp_dict.items():
                        noise = random.uniform(-ci, ci)
                        new_comp[comp] = max(0.1, val * (1 + noise))
                    
                    # Нормализуем обратно до 100
                    total = sum(new_comp.values())
                    if total > 0:
                        new_comp = {k: round((v / total) * 100, 2) for k, v in new_comp.items()}
                        
                    new_comp_text = ", ".join([f"{v}% {k}" for k, v in new_comp.items()])
                    
                    new_row = {'composition': new_comp_text}
                    
                    # Вносим шум в целевые свойства
                    for prop in all_props:
                        val = row.get(prop)
                        if pd.notna(val) and val is not None:
                            # Добавляем небольшой шум к известным свойствам
                            noise = random.uniform(-prop_ci, prop_ci)
                            new_row[prop] = val * (1 + noise)
                        else:
                            new_row[prop] = None
                            
                    synthetic_rows.append(new_row)
            
            if not synthetic_rows:
                return {'success': False, 'error': 'Не удалось сгенерировать новые данные'}
                
            # Сохраняем в БД
            synthetic_df = pd.DataFrame(synthetic_rows)
            insert_data(self.db_path, "measured_parameters", synthetic_df)
            
            print(f"✅ Успешно добавлено {len(synthetic_df)} синтетических образцов")
            
            # Перезагружаем данные и переобучаем модели
            self.training_data = self.load_training_data()
            retrain_result = self.retrain_on_new_data()
            
            return {
                'success': True,
                'message': f'База успешно масштабирована (+{len(synthetic_df)} образцов) и модели переобучены',
                'added_count': len(synthetic_df),
                'retrain_status': retrain_result
            }
            
        except Exception as e:
            print(f"❌ Ошибка при аугментации: {e}")
            return {'success': False, 'error': f'Ошибка аугментации: {str(e)}'}

    def retrain_on_new_data(self, target_properties: List[str] = None, algorithm: str = 'gradient_boosting') -> Dict:
        """Переобучает модели на обновленных данных"""
        print("🔄 Переобучение моделей на новых данных...")
        return self.train_models(target_properties, algorithm)

# Глобальный экземпляр ML системы (ленивая инициализация)
_ml_system = None

def get_ml_system():
    global _ml_system
    if _ml_system is None:
        print("🧠 Инициализация ML системы (PelletMLSystem)...")
        _ml_system = PelletMLSystem()
    return _ml_system