import os
from ml_optimizer import get_ml_system

def main():
    print("--- Тестирование улучшений ML-Оптимизатора ---")
    ml = get_ml_system()
    
    # Печатаем обученные модели
    status = ml.get_ml_system_status()
    print("\n[Статус ML системы]")
    print(f"Обучено: {status['is_trained']}")
    print(f"Доступные компоненты: {status['available_components']}")
    print(f"Модели: {status['trained_models']}")
    
    if not status['trained_models']:
        print("Обучаю модели для тестов...")
        ml.train_models(target_properties=['q', 'ad'])
    
    target_prop = 'q'  # Теплота сгорания - хотим максимизировать
    if target_prop not in ml.predictor.models:
        print(f"Внимание: Модель для {target_prop} не обучена. Использую перую доступную.")
        if status['trained_models']:
            target_prop = status['trained_models'][0]
        else:
            print("Нет моделей. Тест завершен.")
            return

    maximize = True
    print(f"\n[Тест 1] Оптимизация {target_prop} с ограниченными компонентами")
    available = ["Опилки", "Пластик", "Картон"]
    
    print(f"Доступны только: {available}")
    result = ml.optimize_composition(target_property=target_prop, maximize=maximize, available_components=available)
    
    if result.get('success'):
        print(f"Успех! Состав: {result['optimal_composition']}")
        print(f"Значение свойства: {result['optimal_value']:.2f}")
        
        # Проверка, что нет "лишних" компонентов
        for comp in result['optimal_composition']:
            if comp not in available:
                print(f"ОШИБКА: Компонент {comp} использован, но его нет в списке доступных!")
    else:
        print(f"Ошибка оптимизации: {result.get('error')}")

    print(f"\n[Тест 2] Сравнение составов")
    baseline = {"Опилки": 70.0, "Солома": 30.0}
    
    if result.get('success'):
        optimized = result['optimal_composition']
        print(f"Базовый состав: {baseline}")
        print(f"Оптимизированный: {optimized}")
        
        comp_result = ml.ml_optimizer.compare_compositions(baseline, optimized, target_prop, maximize=maximize)
        if comp_result.get('success'):
            print(f"Базовое значение: {comp_result['baseline_value']:.2f}")
            print(f"Оптимизированное значение: {comp_result['optimized_value']:.2f}")
            print(f"Улучшение: {comp_result['improvement_percent']:.1f}%")
            print(f"Вывод: {comp_result['message']}")
        else:
            print(f"Ошибка сравнения: {comp_result.get('error')}")

if __name__ == "__main__":
    main()
