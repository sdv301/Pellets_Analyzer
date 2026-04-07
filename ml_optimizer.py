# ml_optimizer.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize, differential_evolution
import warnings
import re
import os
import joblib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional
import logging

# Настройка логирования
logger = logging.getLogger(__name__)
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
            'СМС': [r'смс', r'cmc', r'с\.?м\.?с'],
            'Пластик': [r'пластик'],
            'Древесная_мука': [r'древесн\w*\s*мук', r'мук\w*\s*древесн\w*'],
            'Щепа': [r'щеп']
        }

    def parse_composition(self, composition_text: str) -> Dict[str, float]:
        if pd.isna(composition_text) or not composition_text:
            return {}
        
        original_text = str(composition_text)
        text = original_text.lower()
        
        composition_dict = {}
        found_matches = []
        
        main_pattern = r'(\d+(?:\.\d+)?)%\s*([^%,+]+?)(?=\s*[,+%]|$)'
        matches = re.findall(main_pattern, text)
        
        for percentage_str, comp_text in matches:
            percentage = float(percentage_str)
            comp_text = comp_text.strip()
            
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
        
        if '+' in text and composition_dict:
            plus_components = re.findall(r'\+\s*([^%+,]+)', text)
            
            if plus_components:
                last_component = list(composition_dict.keys())[-1]
                last_percentage = composition_dict[last_component]
                shared_percentage = last_percentage / (len(plus_components) + 1)
                
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
        
        total = sum(composition_dict.values())
        if total > 0:
            if abs(total - 100) > 1.0:
                for comp in composition_dict:
                    composition_dict[comp] = (composition_dict[comp] / total) * 100
            composition_dict = {k: round(v, 2) for k, v in composition_dict.items()}
        
        return composition_dict

    def validate_and_normalize(self, composition: Dict[str, float],
                                valid_components: List[str] = None,
                                auto_normalize: bool = True) -> Tuple[Dict[str, float], List[str]]:
        """
        Валидирует и нормализует состав.
        
        Args:
            composition: Словарь {компонент: процент}
            valid_components: Список допустимых компонентов (если None — пропускается)
            auto_normalize: Автоматически нормализовать до 100%
            
        Returns:
            Tuple[нормализованный_состав, список_предупреждений]
        """
        warnings = []
        
        if not composition:
            return {}, ['Состав пуст']
        
        # Фильтрация неизвестных компонентов
        if valid_components:
            valid_set = set(valid_components)
            unknown = [k for k in composition.keys() if k not in valid_set]
            if unknown:
                warnings.append(f'Неизвестные компоненты удалены: {", ".join(unknown)}')
                composition = {k: v for k, v in composition.items() if k in valid_set}
        
        if not composition:
            return {}, ['Нет допустимых компонентов после фильтрации']
        
        # Удаление нулевых/отрицательных значений
        composition = {k: v for k, v in composition.items() if v > 0}
        if not composition:
            return {}, ['Все компоненты имеют нулевое или отрицательное значение']
        
        total = sum(composition.values())
        
        if abs(total - 100) > 0.1:
            if auto_normalize:
                warnings.append(f'Сумма компонентов {total:.1f}% нормализована до 100%')
                composition = {k: round((v / total) * 100, 2) for k, v in composition.items()}
            else:
                warnings.append(f'Сумма компонентов {total:.1f}% отличается от 100%')
        
        return composition, warnings

    def create_composition(self, components: Dict[str, float],
                           valid_components: List[str] = None,
                           auto_normalize: bool = True) -> Dict:
        """
        Создает валидный состав из компонентов.
        
        Args:
            components: Словарь {имя_компонента: процент_или_доля}
            valid_components: Список допустимых компонентов
            auto_normalize: Автоматически нормализовать до 100%
            
        Returns:
            Dict с ключами:
                - success: bool
                - composition: Dict[str, float] (нормализованный состав)
                - warnings: List[str] (предупреждения)
                - error: str (ошибка, если есть)
        """
        if not components:
            return {'success': False, 'error': 'Компоненты не указаны'}
        
        normalized, warnings = self.validate_and_normalize(
            components, valid_components, auto_normalize
        )
        
        if not normalized:
            return {'success': False, 'error': warnings[0] if warnings else 'Не удалось создать состав'}
        
        return {
            'success': True,
            'composition': normalized,
            'warnings': warnings
        }

    def composition_to_text(self, composition: Dict[str, float]) -> str:
        """Преобразует словарь состава в текстовый формат для БД."""
        if not composition:
            return ""
        sorted_comp = sorted(composition.items(), key=lambda x: x[1], reverse=True)
        return ", ".join([f"{v}% {k}" for k, v in sorted_comp])

    def composition_from_text(self, text: str) -> Dict[str, float]:
        """Парсит текстовый формат состава в словарь (алиас для parse_composition)."""
        return self.parse_composition(text)


class PelletPropertyPredictor:
    def __init__(self, ml_system=None):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_trained = False
        self.training_metrics = {}
        self.parser = CompositionParser()
        self.ml_system = ml_system
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        self.target_properties_mapping = {
            'war': 'Влажность на аналитическую массу',
            'ad': 'Зольность на сухую массу', 
            'vd': 'Содержание летучих на сухую массу',
            'q': 'Теплота сгорания',
            'cd': 'Содержание углерода на сухую массу',
            'hd': 'Содержание водорода на сухую массу',
            'nd': 'Содержание азота на сухую массу',
            'sd': 'Содержание серы на сухую массу',
            'od': 'Содержание кислорода на сухую массу',
            'density': 'Плотность',
            'kf': 'Ударопрочность',
            'kt': 'Устойчивость к вибрациям'
        }
        self.main_target_properties = list(self.target_properties_mapping.keys())
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[int]]:
        if 'composition' not in data.columns:
            return np.array([]), [], []
        
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
        
        X = []
        final_valid_indices = []
        
        for i, comp_dict in enumerate(composition_data):
            row = [comp_dict.get(comp, 0.0) for comp in component_list]
            total = sum(row)
            if total > 10:
                row = [(val / total) * 100 for val in row]
                X.append(row)
                final_valid_indices.append(valid_indices[i])
        
        if not X:
            return np.array([]), [], []
        
        return np.array(X), component_list, final_valid_indices
    
    def _train_single_property(self, prop: str, X_prop: np.ndarray, y_prop: pd.Series, algorithm: str, feature_names: List[str]) -> Tuple[str, Dict]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_prop)
        
        # ЛОГИКА ВЫБОРА АЛГОРИТМА (С учетом выбора пользователя)
        additive_properties = ['q', 'ad', 'war', 'density', 'vd', 'cd', 'hd', 'nd', 'sd', 'od']
        
        if algorithm == 'ridge':
            base_model = Ridge(alpha=1.0)
            param_grid = {'alpha': [0.1, 1.0, 10.0]}
            used_algo_name = "Линейная регрессия (Ridge)"
            algo_reason = "Принудительно выбрано пользователем."
        elif algorithm == 'xgboost':
            base_model = XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            used_algo_name = "XGBoost"
            algo_reason = "Градиентный бустинг с высокой точностью и защитой от переобучения."
        elif algorithm == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            base_model = RandomForestRegressor(random_state=42)
            param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
            used_algo_name = "Случайный лес (Random Forest)"
            algo_reason = "Принудительно выбрано пользователем."
        elif algorithm == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingRegressor
            base_model = GradientBoostingRegressor(random_state=42)
            param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
            used_algo_name = "Градиентный бустинг (GBM)"
            algo_reason = "Принудительно выбрано пользователем."
        else:
            # УМНЫЙ (АВТОМАТИЧЕСКИЙ) ВЫБОР
            if prop in additive_properties:
                base_model = Ridge(alpha=1.0)
                param_grid = {'alpha': [0.1, 1.0, 10.0]}
                used_algo_name = "Линейная регрессия [Авто]"
                algo_reason = "Умный выбор: исключает физические парадоксы смешивания."
            else:
                base_model = XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0
                )
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                }
                used_algo_name = "XGBoost [Авто]"
                algo_reason = "Умный выбор: высокая точность, работа со сложными зависимостями."
        
        n_cv = min(3, len(y_prop))
        if n_cv < 2:
            n_cv = 2
            
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=n_cv, 
            scoring='r2', 
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_scaled, y_prop)
        
        model = grid_search.best_estimator_
        y_pred = model.predict(X_scaled)
        r2 = float(r2_score(y_prop, y_pred))
        mae = float(mean_absolute_error(y_prop, y_pred))
        
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            importances = np.ones(len(feature_names))
            
        total = sum(importances)
        if total != 0:
            normalized = importances / total
            feature_importance = {k: float(v) for k, v in zip(feature_names, normalized)}
            
        metrics = {
            'model': model, 'scaler': scaler, 'r2_score': r2, 'mae': mae, 'cv_r2': r2,
            'feature_importance': feature_importance,
            'algorithm_used': used_algo_name, 'algorithm_reason': algo_reason
        }
        return prop, metrics

    def train(self, data: pd.DataFrame, target_properties: List[str], algorithm: str = 'xgboost', selected_features: List[str] = None) -> Dict:
        X, feature_names, valid_indices = self.prepare_features(data)
        if len(X) == 0:
            return {'success': False, 'trained_count': 0, 'skipped': [], 'error': 'Нет данных для формирования признаков'}
        
        if selected_features and len(selected_features) > 0:
            valid_selected = [f for f in selected_features if f in feature_names]
            if not valid_selected:
                return {'success': False, 'trained_count': 0, 'skipped': [], 'error': 'Ни один из выбранных компонентов не найден в данных'}
            feat_indices = [feature_names.index(f) for f in valid_selected]
            X = X[:, feat_indices]
            current_feature_names = valid_selected
        else:
            current_feature_names = feature_names

        self.feature_names = current_feature_names
        trained_count = 0
        skipped_props = []
        
        tasks = []
        for prop in target_properties:
            if prop not in data.columns:
                continue
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
                        'feature_importance': metrics['feature_importance'],
                        'algorithm_used': metrics['algorithm_used'],
                        'algorithm_reason': metrics['algorithm_reason']
                    }
                    trained_count += 1
                except Exception as e:
                    skipped_props.append({'property': prop, 'reason': str(e)})

        self.is_trained = trained_count > 0
        if self.is_trained:
            self.save_models()
            
        return {
            'success': self.is_trained,
            'trained_count': trained_count,
            'skipped': skipped_props
        }

    def save_models(self):
        """Сохраняет обученные модели на диск с логированием ошибок"""
        try:
            data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_names': self.feature_names,
                'training_metrics': self.training_metrics
            }
            file_path = os.path.join(self.models_dir, 'pellet_models.joblib')
            joblib.dump(data, file_path)
            logger.info(f"Модели успешно сохранены в {file_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения моделей: {e}")
            raise

    def load_models(self) -> bool:
        """Загружает обученные модели с диска с логированием ошибок"""
        try:
            file_path = os.path.join(self.models_dir, 'pellet_models.joblib')
            if os.path.exists(file_path):
                data = joblib.load(file_path)
                self.models = data['models']
                self.scalers = data['scalers']
                self.feature_names = data['feature_names']
                self.training_metrics = data['training_metrics']
                self.is_trained = True
                logger.info(f"Модели успешно загружены из {file_path}")
                return True
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {e}")
        return False
    
    def predict(self, composition: Dict[str, float], target_property: str) -> Optional[float]:
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
        total = sum(composition.values())
        if total != 100:
            composition = {k: (v / total) * 100 for k, v in composition.items()}
        X = [composition.get(feature, 0.0) for feature in self.feature_names]
        return np.array(X)
    
    def get_feature_importance(self, target_property: str) -> Dict[str, float]:
        if target_property not in self.training_metrics:
            return {}
        return self.training_metrics[target_property].get('feature_importance', {})

    def create_composition(self, components: Dict[str, float],
                           valid_components: List[str] = None,
                           auto_normalize: bool = True) -> Dict:
        """
        Создаёт валидный состав из компонентов через парсер.
        """
        return self.parser.create_composition(components, valid_components, auto_normalize)

    def validate_composition(self, composition: Dict[str, float],
                             valid_components: List[str] = None) -> Tuple[Dict[str, float], List[str]]:
        """
        Валидирует и нормализует состав.
        """
        return self.parser.validate_and_normalize(composition, valid_components)

    def composition_to_text(self, composition: Dict[str, float]) -> str:
        """Преобразует словарь состава в текстовый формат для БД."""
        return self.parser.composition_to_text(composition)

    def composition_from_text(self, text: str) -> Dict[str, float]:
        """Парсит текстовый формат состава в словарь."""
        return self.parser.parse_composition(text)


class MLCompositionOptimizer:
    def __init__(self, predictor):
        self.predictor = predictor
        self.optimization_history = []
    
    def optimize_composition(self, target_property: str, maximize: bool = True, constraints: Dict[str, Tuple[float, float]] = None, available_components: List[str] = None) -> Dict:
        if target_property not in self.predictor.models:
            return {'success': False, 'error': f'Модель для {target_property} не обучена'}
        
        feature_names = self.predictor.feature_names
        n_features = len(feature_names)
        
        if constraints:
            min_total = 0.0
            max_total = 0.0
            for comp, (min_val, max_val) in constraints.items():
                if comp in feature_names:
                    min_total += min_val
                    max_total += max_val
            if min_total > 100.0:
                return {'success': False, 'error': f'Ограничения несовместимы: минимальная сумма {min_total:.1f}% > 100%'}
            if max_total < 100.0:
                return {'success': False, 'error': f'Ограничения несовместимы: максимальная сумма {max_total:.1f}% < 100%'}
        
        available_indices = []
        if available_components:
            for i, feat in enumerate(feature_names):
                if feat in available_components:
                    available_indices.append(i)
        else:
            available_indices = list(range(n_features))

        if not available_indices:
            return {'success': False, 'error': 'Ни один из доступных компонентов не распознан моделью'}

        bounds_percent = [(0.0, 100.0) for _ in range(n_features)]
        
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
                if total <= 1e-6:
                    return 1e6 if maximize else -1e6
                normalized = (composition / total) * 100.0
                
                comp_dict = {f_name: normalized[i] for i, f_name in enumerate(feature_names)}
                pred = self.predictor.predict(comp_dict, target_property)
                
                if pred is None:
                    return 1e6 if maximize else -1e6
                
                # ЗАЩИТА ОТ ФИЗИЧЕСКИХ ПАРАДОКСОВ
                if target_property in ['q', 'ad', 'war', 'density']:
                    max_possible = max([self.predictor.predict({f: 100}, target_property) for f in comp_dict.keys() if comp_dict[f] > 0])
                    if maximize and pred > max_possible:
                        pred = max_possible - 0.01 

                score = -pred if maximize else pred
                
                penalty = 0.0
                for i, val in enumerate(normalized):
                    b_min, b_max = bounds_percent[i]
                    if val < b_min - 0.1:
                        penalty += (b_min - val) * 1000
                    elif val > b_max + 0.1:
                        penalty += (val - b_max) * 1000
                        
                return score + penalty
            except:
                return 1e6 if maximize else -1e6
        
        de_bounds = [(0.0, 1.0) if i in available_indices else (0.0, 0.0) for i in range(n_features)]
        
        try:
            result = differential_evolution(objective, de_bounds, maxiter=50, popsize=15, mutation=(0.5, 1.0), recombination=0.7, seed=42, disp=False)
        except Exception as e:
            return {'success': False, 'error': f'Ошибка оптимизатора: {str(e)}'}
        
        if not result.success:
            return {'success': False, 'error': 'Не удалось найти решение.'}
        
        best_comp = result.x
        total_w = sum(best_comp)
        if total_w > 0:
            final_comp = (best_comp / total_w) * 100.0
        else:
            final_comp = best_comp
            
        optimal_composition = dict(zip(feature_names, final_comp))
        optimal_composition = {k: round(v, 2) for k, v in optimal_composition.items() if v > 0.1}
        total = sum(optimal_composition.values())
        
        if total > 0 and abs(total - 100) > 0.1:
            optimal_composition = {k: round(float(v / total) * 100, 2) for k, v in optimal_composition.items()}
        optimal_value = float(-result.fun) if maximize else float(result.fun)
        
        display_name = self.predictor.target_properties_mapping.get(target_property, target_property)
        direction = "максимизации" if maximize else "минимизации"
        comp_text = ", ".join([f"{v:.1f}% {k}" for k, v in optimal_composition.items()])
        
        message = (f"Оптимальный состав для {direction} {display_name}: {comp_text}. "
                f"Ожидаемое значение: {optimal_value:.2f}.")
        
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


class PelletMLSystem:
    def __init__(self, db_path: str = 'pellets_data.db', use_remote: bool = False, **ssh_kwargs):
        """
        Инициализация ML-системы с поддержкой удалённой БД.
        
        Args:
            db_path: Путь к локальной БД (или кэш для удалённой)
            use_remote: Использовать удалённую БД через SSH
            **ssh_kwargs: Параметры для SSH-подключения:
                - ssh_host: Адрес SSH-сервера
                - ssh_port: Порт SSH (22)
                - ssh_user: Имя пользователя
                - ssh_password: Пароль
                - ssh_key_path: Путь к ключу
                - remote_db_path: Путь к удалённой БД
                - auto_sync: Автосинхронизация
        """
        self.db_path = db_path
        self.use_remote = use_remote
        self.ssh_kwargs = ssh_kwargs
        self._db_connection = None
        
        # Подключение к БД
        if use_remote:
            self._connect_remote_db()
        
        self.predictor = PelletPropertyPredictor(self)
        self.ml_optimizer = MLCompositionOptimizer(self.predictor)
        models_loaded = self.predictor.load_models()
        self.components = self.load_components()
        self.training_data = self.load_training_data()
        
        # АВТОМАТИЧЕСКАЯ ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ ПРИ ПЕРВОМ ЗАПУСКЕ
        if not models_loaded or not self.predictor.is_trained:
            self._initialize_default_models()
    
    def _connect_remote_db(self):
        """Подключается к удалённой БД через SSH-туннель."""
        try:
            from database import DatabaseConnection
            self._db_connection = DatabaseConnection(
                db_path=self.db_path,
                use_remote=True,
                **self.ssh_kwargs
            )
            result = self._db_connection.connect()
            if result['success']:
                logger.info(f"✅ Подключено к удалённой БД: {result.get('remote_path', 'unknown')}")
            else:
                logger.warning(f"⚠️ Не удалось подключиться к удалённой БД: {result.get('error')}. Использую локальную.")
                self.use_remote = False
                self._db_connection = None
        except Exception as e:
            logger.error(f"Ошибка подключения к удалённой БД: {e}")
            self.use_remote = False
            self._db_connection = None
    
    def sync_to_remote(self) -> bool:
        """Синхронизирует локальную БД с удалённой."""
        if self._db_connection and self.use_remote:
            try:
                self._db_connection.close()  # Это вызовет upload
                self._db_connection.connect()  # Переподключение
                return True
            except Exception as e:
                logger.error(f"Ошибка синхронизации: {e}")
                return False
        return False
    
    def close(self):
        """Закрывает подключение к БД."""
        if self._db_connection:
            self._db_connection.close()
            self._db_connection = None

    def _initialize_default_models(self):
        """
        Автоматически создает начальные ML модели на основе данных из БД.
        Использует готовые составы из measured_parameters для обучения.
        """
        logger.info("🔄 Попытка автоматической инициализации ML моделей...")
        
        # Проверяем есть ли данные для обучения
        if self.training_data.empty or len(self.training_data) < 3:
            logger.warning("⚠️ Недостаточно данных для автоматической инициализации моделей")
            return
        
        # Проверяем сколько записей имеют составы
        compositions_with_data = 0
        for _, row in self.training_data.iterrows():
            if 'composition' in row and not pd.isna(row['composition']):
                comp_dict = self.predictor.parser.parse_composition(row['composition'])
                if comp_dict:
                    compositions_with_data += 1
        
        if compositions_with_data < 3:
            logger.warning(f"⚠️ Найдено только {compositions_with_data} составов для обучения")
            return
        
        # Обучаем модели автоматически на всех доступных данных
        logger.info(f"✅ Найдено {compositions_with_data} составов. Запускаю обучение...")
        
        # Обучаем на основных свойствах
        target_props = ['q', 'density', 'kf', 'ad']
        result = self.train_models(target_properties=target_props, algorithm='auto')
        
        if result.get('success'):
            logger.info(f"✅ ML модели успешно инициализированы! Обучено: {result.get('trained_count', 0)} моделей")
        else:
            logger.warning(f"⚠️ Не удалось инициализировать модели: {result.get('error')}")

    def reload_data(self):
        try:
            self.components = self.load_components()
            self.training_data = self.load_training_data()
        except Exception:
            pass

    def load_components(self) -> pd.DataFrame:
        try:
            from database import query_db
            components = query_db(self.db_path, "components")
            return components
        except Exception:
            return pd.DataFrame()
    
    def linear_predict(self, composition: Dict[str, float], target_property: str) -> Optional[float]:
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
            return value * (100.0 / total_weight)
        return None
    
    def load_training_data(self) -> pd.DataFrame:
        try:
            from database import query_db, get_ml_optimizations
            training_data = query_db(self.db_path, "measured_parameters")
            ml_optimizations = get_ml_optimizations(self.db_path, limit=100)
            
            if not ml_optimizations.empty:
                all_target_props = list(self.predictor.target_properties_mapping.keys())
                for _, opt in ml_optimizations.iterrows():
                    composition_text = ", ".join([f"{v}% {k}" for k, v in opt['optimal_composition'].items()])
                    new_row = {'composition': composition_text}
                    for prop in all_target_props:
                        new_row[prop] = opt['optimal_value'] if opt['target_property'] == prop else None
                    if not self.components.empty:
                        comp_dict = opt['optimal_composition']
                        for prop in all_target_props:
                            if new_row.get(prop) is None:
                                linear_val = self.linear_predict(comp_dict, prop)
                                if linear_val is not None:
                                    new_row[prop] = linear_val
                    training_data = pd.concat([training_data, pd.DataFrame([new_row])], ignore_index=True)
            return training_data
        except Exception:
            return pd.DataFrame()
 
    def get_ml_system_status(self) -> Dict:
        all_comps = set()
        if not self.training_data.empty and 'composition' in self.training_data.columns:
            for comp_text in self.training_data['composition'].dropna():
                parsed = self.predictor.parser.parse_composition(comp_text)
                if parsed:
                    all_comps.update(parsed.keys())
        if self.predictor.feature_names:
            all_comps.update(self.predictor.feature_names)
            
        status = {
            'is_trained': self.predictor.is_trained,
            'trained_models': list(self.predictor.models.keys()),
            'available_components': sorted(list(all_comps)),
            'training_data_size': len(self.training_data) if not self.training_data.empty else 0,
            'ml_optimizations_count': len(self.ml_optimizer.optimization_history)
        }
        
        model_metrics = {}
        for prop in self.predictor.models.keys():
            model_metrics[prop] = {
                'feature_importance': self.predictor.training_metrics.get(prop, {}).get('feature_importance', {}),
                'training_metrics': {
                    'r2_score': self.predictor.training_metrics.get(prop, {}).get('r2_score', 0),
                    'mae': self.predictor.training_metrics.get(prop, {}).get('mae', 0),
                    'cv_r2': self.predictor.training_metrics.get(prop, {}).get('cv_r2', 0)
                },
                'display_name': self.predictor.target_properties_mapping.get(prop, prop),
                'algorithm_used': self.predictor.training_metrics.get(prop, {}).get('algorithm_used', 'Модель не определена'),
                'algorithm_reason': self.predictor.training_metrics.get(prop, {}).get('algorithm_reason', '')
            }
        
        status['model_metrics'] = model_metrics
        status['target_properties_mapping'] = self.predictor.target_properties_mapping
        return status

    def train_models(self, target_properties: List[str] = None, algorithm: str = 'xgboost', selected_features: List[str] = None) -> Dict:
        """Обучает ML модели с улучшенной обработкой ошибок"""
        if self.training_data.empty:
            return {'success': False, 'error': 'Нет данных для обучения'}
        if target_properties is None:
            target_properties = self.predictor.main_target_properties
        
        logger.info(f"Начало обучения моделей. Алгоритм: {algorithm}, свойств: {len(target_properties)}")
        
        train_result = self.predictor.train(self.training_data, target_properties, algorithm, selected_features)
        if train_result.get('success', False):
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
                logger.info("Метрики моделей сохранены в базу данных")
            except Exception as e:
                logger.warning(f"Не удалось сохранить метрики в БД: {e}")
            
            status = self.get_ml_system_status()
            logger.info(f"Обучение завершено. Обучено моделей: {len(status['trained_models'])}")
            return {
                'success': True,
                'message': 'ML система успешно обучена!',
                'status': status,
                'trained_count': len(status['trained_models']),
                'metrics': {prop: status['model_metrics'].get(prop) for prop in status['trained_models']},
                'skipped': train_result.get('skipped', [])
            }
        else:
            logger.error(f"Обучение не удалось: {train_result.get('error')}")
            return {
                'success': False, 
                'error': train_result.get('error', 'Обучение не удалось из-за нехватки данных'),
                'skipped': train_result.get('skipped', [])
            }

    def optimize_composition(self, target_property: str, maximize: bool = True, constraints: Dict[str, Tuple[float, float]] = None, available_components: List[str] = None) -> Dict:
        """Оптимизирует состав с логированием и улучшенной обработкой ошибок"""
        logger.info(f"Оптимизация для {target_property}, maximize={maximize}")
        
        if target_property not in self.predictor.models:
            if self.components.empty:
                return {'success': False, 'error': f'Модель для {target_property} не обучена и нет данных компонентов'}
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
                    'algorithm': 'xgboost',
                    'model_metrics': self.predictor.training_metrics.get(target_property, {})
                }
                insert_ml_optimization(self.db_path, optimization_data)
                add_ml_optimization_to_training_data(self.db_path, optimization_data)
                self.training_data = self.load_training_data()
                logger.info(f"Оптимизация успешна. Значение: {result['optimal_value']:.2f}")
            except Exception as e:
                logger.warning(f"Не удалось сохранить оптимизацию в БД: {e}")
        else:
            logger.warning(f"Оптимизация не удалась: {result.get('error')}")
            
        return result
    
    def _optimize_from_components(self, target_property: str, maximize: bool = True, constraints: Dict[str, Tuple[float, float]] = None, available_components: List[str] = None) -> Dict:
        """
        Оптимизация на основе линейных свойств компонентов из БД.
        Используется как fallback когда ML модель не обучена.
        """
        if self.components.empty:
            return {'success': False, 'error': 'Нет данных о компонентах для оптимизации'}
        
        if target_property not in self.components.columns:
            return {'success': False, 'error': f'Свойство {target_property} отсутствует в данных компонентов'}
        
        # Получаем доступные компоненты и их значения целевого свойства
        comp_values = []
        for _, row in self.components.iterrows():
            comp_name = row['component']
            if available_components and comp_name not in available_components:
                continue
            if pd.notna(row.get(target_property)):
                comp_values.append((comp_name, float(row[target_property])))
        
        if not comp_values:
            return {'success': False, 'error': f'Нет данных о свойстве {target_property} для доступных компонентов'}
        
        # Для максимизации выбираем компонент с наибольшим значением
        # Для минимизации - с наименьшим
        if maximize:
            best_comp = max(comp_values, key=lambda x: x[1])
        else:
            best_comp = min(comp_values, key=lambda x: x[1])
        
        optimal_composition = {best_comp[0]: 100.0}
        optimal_value = best_comp[1]
        
        display_name = self.predictor.target_properties_mapping.get(target_property, target_property)
        direction = "максимизации" if maximize else "минимизации"
        
        message = f"Оптимальный состав для {direction} {display_name}: 100% {best_comp[0]}. Ожидаемое значение: {optimal_value:.2f}."
        
        return {
            'success': True,
            'optimal_composition': optimal_composition,
            'optimal_value': optimal_value,
            'message': message,
            'is_linear': True
        }

    def create_composition(self, components: Dict[str, float],
                           auto_normalize: bool = True,
                           check_db: bool = True) -> Dict:
        """
        Создаёт валидный состав из компонентов с проверкой по БД.
        
        Args:
            components: Словарь {имя_компонента: процент}
            auto_normalize: Автоматически нормализовать до 100%
            check_db: Проверять наличие компонентов в БД
            
        Returns:
            Dict с ключами:
                - success: bool
                - composition: Dict[str, float] (нормализованный состав)
                - text: str (текстовое представление для БД)
                - warnings: List[str]
                - error: str (если ошибка)
        """
        if not components:
            return {'success': False, 'error': 'Компоненты не указаны'}
        
        # Определяем допустимые компоненты
        valid_components = None
        if check_db and not self.components.empty:
            valid_components = list(self.components['component'].values)
            # Также добавляем компоненты из ML модели
            if self.predictor.feature_names:
                valid_components = list(set(valid_components + self.predictor.feature_names))
        
        # Создаём и валидируем состав
        result = self.predictor.create_composition(components, valid_components, auto_normalize)
        
        if not result.get('success'):
            return result
        
        composition = result['composition']
        warnings = result.get('warnings', [])
        
        # Проверяем предсказуемость свойств
        predictions = {}
        if self.predictor.is_trained:
            for prop in self.predictor.models.keys():
                try:
                    pred = self.predictor.predict(composition, prop)
                    if pred is not None:
                        display_name = self.predictor.target_properties_mapping.get(prop, prop)
                        predictions[display_name] = round(float(pred), 2)
                except Exception as e:
                    logger.warning(f"Не удалось предсказать {prop}: {e}")
        
        return {
            'success': True,
            'composition': composition,
            'text': self.predictor.composition_to_text(composition),
            'warnings': warnings,
            'predictions': predictions,
            'total': round(sum(composition.values()), 2)
        }

    def augment_database(self, variations_count: int = 3, confidence_interval: float = 5.0) -> Dict:
        try:
            from database import query_db, insert_data
            measured_data = query_db(self.db_path, "measured_parameters")
            if measured_data.empty:
                return {'success': False, 'error': 'База данных пуста'}
            import random
            ci = confidence_interval / 100.0
            prop_ci = max(0.01, ci / 2)
            synthetic_rows = []
            all_props = ['density', 'kf', 'kt', 'h', 'mass_loss', 'tign', 'tb', 'tau_d1', 'tau_d2', 'tau_b', 'co2', 'co', 'so2', 'nox', 'q', 'ad']
            
            for _, row in measured_data.iterrows():
                comp_text = row.get('composition')
                if not comp_text or pd.isna(comp_text):
                    continue
                comp_dict = self.predictor.parser.parse_composition(comp_text)
                if not comp_dict:
                    continue
                for _ in range(variations_count):
                    new_comp = {}
                    for comp, val in comp_dict.items():
                        noise = random.uniform(-ci, ci)
                        new_comp[comp] = max(0.1, val * (1 + noise))
                    total = sum(new_comp.values())
                    if total > 0:
                        new_comp = {k: round((v / total) * 100, 2) for k, v in new_comp.items()}
                    new_row = {'composition': ", ".join([f"{v}% {k}" for k, v in new_comp.items()])}
                    for prop in all_props:
                        val = row.get(prop)
                        if pd.notna(val) and val is not None:
                            new_row[prop] = val * (1 + random.uniform(-prop_ci, prop_ci))
                        else:
                            new_row[prop] = None
                    synthetic_rows.append(new_row)
            
            if not synthetic_rows:
                return {'success': False, 'error': 'Не удалось сгенерировать новые данные'}
            synthetic_df = pd.DataFrame(synthetic_rows)
            insert_data(self.db_path, "measured_parameters", synthetic_df)
            self.training_data = self.load_training_data()
            retrain_result = self.train_models()
            return {'success': True, 'message': f'База успешно масштабирована (+{len(synthetic_df)} образцов)', 'added_count': len(synthetic_df), 'retrain_status': retrain_result}
        except Exception as e:
            return {'success': False, 'error': f'Ошибка аугментации: {str(e)}'}


_ml_system = None
def get_ml_system(use_remote: bool = False, **ssh_kwargs):
    """
    Получение ML-системы с поддержкой удалённой БД.
    
    Args:
        use_remote: Использовать удалённую БД через SSH
        **ssh_kwargs: Параметры SSH-подключения
            - ssh_host: Адрес SSH-сервера
            - ssh_user: Имя пользователя
            - ssh_password: Пароль (или ssh_key_path)
            - ssh_key_path: Путь к SSH-ключу
            - remote_db_path: Путь к удалённой БД
    
    Пример:
        # Локальная БД
        ml = get_ml_system()
        
        # Удалённая БД
        ml = get_ml_system(
            use_remote=True,
            ssh_host='server.com',
            ssh_user='user',
            ssh_key_path='~/.ssh/id_rsa',
            remote_db_path='/path/to/pellets_data.db'
        )
    """
    global _ml_system
    
    # Если параметры изменились — пересоздаём
    if _ml_system is not None:
        if not use_remote and _ml_system.use_remote:
            _ml_system.close()
            _ml_system = None
        elif use_remote and _ml_system.use_remote:
            return _ml_system  # Уже подключено
        else:
            _ml_system.close()
            _ml_system = None
    
    if _ml_system is None:
        if use_remote:
            _ml_system = PelletMLSystem(use_remote=True, **ssh_kwargs)
        else:
            _ml_system = PelletMLSystem()
    
    return _ml_system


def reload_ml_system():
    """Перезагружает ML-систему (полезно после изменения данных)."""
    global _ml_system
    if _ml_system:
        _ml_system.close()
        _ml_system = None
