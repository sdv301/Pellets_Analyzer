"""Статистические вычисления и фильтрация параметров для графиков."""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from app.services.charts.palette import PARAM_NAMES


def get_data_statistics(data: pd.DataFrame) -> Dict[str, Any]:
    """Возвращает статистику по числовым колонкам датафрейма."""
    if data.empty:
        return {}
    
    stats = {}
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        stats[col] = {
            'mean': float(data[col].mean()) if not pd.isna(data[col].mean()) else None,
            'std': float(data[col].std()) if not pd.isna(data[col].std()) else None,
            'min': float(data[col].min()) if not pd.isna(data[col].min()) else None,
            'max': float(data[col].max()) if not pd.isna(data[col].max()) else None,
            'count': int(data[col].count())
        }
    
    return stats


def simple_filter(df: pd.DataFrame, column: str, operator: str, value: Any) -> pd.DataFrame:
    """Фильтрация датафрейма по условию."""
    if column not in df.columns:
        return pd.DataFrame()
    
    try:
        if column == 'composition':
            return df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
        
        try:
            if operator == 'BETWEEN':
                min_val, max_val = value
                num_min = float(min_val)
                num_max = float(max_val)
                return df[(df[column] >= num_min) & (df[column] <= num_max)]
                
            num_value = float(value)
            
            if operator == '=':
                return df[df[column] == num_value]
            elif operator == '>':
                return df[df[column] > num_value]
            elif operator == '>=':
                return df[df[column] >= num_value]
            elif operator == '<':
                return df[df[column] < num_value]
            elif operator == '<=':
                return df[df[column] <= num_value]
            elif operator == '!=':
                return df[df[column] != num_value]
            elif operator == 'LIKE':
                return df[df[column] == num_value]
                
        except ValueError:
            if operator in ('LIKE', '='):
                return df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
            return pd.DataFrame()
                
    except Exception:
        pass
    
    return pd.DataFrame()


def filter_parameters(data: pd.DataFrame, param_group: str, show_diff_only: bool = False) -> pd.DataFrame:
    """Фильтрует параметры по смысловым группам."""
    param_groups = {
        'thermal': ['q', 'tign', 'tb', 'tau_b', 'tau_d1', 'tau_d2'],
        'mechanical': ['density', 'kf', 'kt', 'h', 'mass_loss'],
        'chemical': ['ad', 'cd', 'hd', 'nd', 'sd', 'od', 'vd', 'war'],
        'emissions': ['co2', 'co', 'so2', 'nox'],
        'combustion': ['mass_loss', 'tau_b', 'tau_d1', 'tau_d2', 'tign', 'tb']
    }
    
    if param_group != 'all' and param_group in param_groups:
        selected_params = ['composition'] + param_groups[param_group]
        existing_params = [col for col in selected_params if col in data.columns]
        if not existing_params:
            return pd.DataFrame()
        filtered_data = data[existing_params]
    else:
        filtered_data = data.copy()
    
    if show_diff_only and len(data['composition'].unique()) >= 2:
        try:
            numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
            significant_cols = ['composition']
            
            for col in numeric_cols:
                if col != 'composition':
                    composition_stats = filtered_data.groupby('composition')[col].mean()
                    if len(composition_stats) >= 2:
                        max_val = composition_stats.max()
                        min_val = composition_stats.min()
                        if max_val > 0:
                            difference_pct = ((max_val - min_val) / max_val) * 100
                            if difference_pct >= 10:
                                significant_cols.append(col)
            
            if len(significant_cols) > 1:
                filtered_data = filtered_data[significant_cols]
        except Exception:
            pass
    
    return filtered_data


def get_comparison_stats(data: pd.DataFrame) -> Dict[str, Any]:
    """Возвращает статистику для блока сравнения."""
    if data.empty:
        return {}
    
    composition_column = 'composition' if 'composition' in data.columns else 'Состав'
    if composition_column not in data.columns:
        return {
            'compositions_count': 0,
            'parameters_count': len(data.select_dtypes(include=[np.number]).columns),
            'total_rows': len(data)
        }
    
    numeric_data = data.select_dtypes(include=[np.number])
    return {
        'compositions_count': len(data[composition_column].unique()),
        'parameters_count': len(numeric_data.columns),
        'total_rows': len(data)
    }


def get_key_params_summary(comp_data: pd.DataFrame, max_params: int = 3) -> str:
    """Возвращает ключевые параметры для состава."""
    numeric_cols = comp_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return "Нет числовых данных"
    
    summary = []
    for col in numeric_cols[:max_params]:
        mean_val = comp_data[col].mean()
        if not pd.isna(mean_val):
            summary.append(f"{PARAM_NAMES.get(col, col)}: {mean_val:.2f}")
    
    return "; ".join(summary) if summary else "Нет данных"


def get_compact_composition_table(data: pd.DataFrame, max_compositions: int = 10) -> Optional[pd.DataFrame]:
    """Создает компактную таблицу составов для легенды."""
    if data.empty or 'composition' not in data.columns:
        return None
    
    compositions = data['composition'].unique()
    if len(compositions) > max_compositions:
        main_compositions = compositions[:max_compositions-1]
        other_count = len(compositions) - len(main_compositions)
        compositions = list(main_compositions) + [f'Другие ({other_count} составов)']
    
    compact_data = []
    for comp in compositions:
        comp_data = data[data['composition'] == comp]
        if not comp_data.empty:
            compact_data.append({
                'composition': comp,
                'samples': len(comp_data),
                'key_params': get_key_params_summary(comp_data)
            })
    
    return pd.DataFrame(compact_data)
