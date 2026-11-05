/**
 * static/js/app_tour.js — Интерактивный тур онбординга по системе Pellets Analyzer.
 * Поддерживает ролевые туры (оператор / лаборант / админ), контекстные шаги для активной страницы
 * и защиту от автозапуска на тяжёлых графических страницах.
 */
(function() {
    'use strict';

    function buildTourSteps(userRole) {
        const isAdmin = userRole === 'admin';
        const isOperator = userRole === 'operator' || userRole === 'user' || userRole === 'guest';

        const steps = [
            {
                element: '#navbar',
                popover: {
                    title: isAdmin
                        ? '👋 Pellets Analyzer: Режим Администратора'
                        : '👋 Добро пожаловать в Pellets Analyzer!',
                    description: isAdmin
                        ? 'Полный доступ к управлению ML-моделями (v2 / legacy), ролями пользователей, системными метриками и базой данных.'
                        : 'Аналитическая платформа для оперативного расчёта теплофизических свойств, подбора рецептур и экономической оценки пеллет.',
                    side: 'bottom',
                    align: 'start'
                }
            },
            {
                element: '#globalSearchInput',
                popover: {
                    title: '🔍 Быстрый поиск составов',
                    description: 'Мгновенный поиск рецептур по сырью и породе с отображением высшей теплоты сгорания и зольности прямо в подсказках.',
                    side: 'bottom',
                    align: 'center'
                }
            },
            {
                element: '#notificationsDropdownWrapper',
                popover: {
                    title: '🔔 Центр статуса и уведомлений',
                    description: 'Индикатор актуальной версии ML-модели (pinned lab v2 / legacy), объёма базы и системных оповещений.',
                    side: 'bottom',
                    align: 'end'
                }
            },
            {
                element: '#themeToggleBtn',
                popover: {
                    title: '🌓 Переключатель темы',
                    description: 'Комфортное переключение между контрастной тёмной и светлой темами AstroWind.',
                    side: 'bottom',
                    align: 'end'
                }
            },
            {
                element: '#sidenav-main',
                popover: {
                    title: '🧭 Навигация по модулям',
                    description: isAdmin
                        ? 'Переходите между Базой данных, Лабораторией ИИ, Экономикой, Сравнением, Графиками и закрытой Админ-панелью.'
                        : 'Быстрый переход между Базой данных, Лабораторией ИИ, Калькулятором себестоимости и Сравнением рецептур.',
                    side: 'right',
                    align: 'start'
                }
            }
        ];

        // Контекстный шаг: Пресеты на странице таблиц
        if (document.getElementById('presetGroup')) {
            steps.push({
                element: '#presetGroup',
                popover: {
                    title: '⚡ Пресеты фильтрации',
                    description: 'Быстрое переключение срезов базы (сырые пеллеты, гидротермальная обработка, торф) с возможностью сохранения собственных пресетов.',
                    side: 'bottom',
                    align: 'start'
                }
            });
        }

        // Контекстный шаг: Песочница на странице ML
        if (document.getElementById('step4-sandbox')) {
            steps.push({
                element: '#step4-sandbox',
                popover: {
                    title: '🧪 ML-Песочница рецептур',
                    description: 'Интерактивный ввод процентных долей компонентов, прогнозирование Q с доверительным интервалом и мгновенный расчёт невязки (Residual) с лабораторным фактом.',
                    side: 'top',
                    align: 'center'
                }
            });
        }

        // Контекстный шаг: Быстрые параметры графиков
        if (document.getElementById('quickParamButtons')) {
            steps.push({
                element: '#quickParamButtons',
                popover: {
                    title: '📊 Быстрый выбор параметров',
                    description: 'В один клик назначайте зольность, теплоту, влагу или углерод на активную ось X/Y/Z.',
                    side: 'top',
                    align: 'start'
                }
            });
        }

        // Контекстный шаг: Экономический калькулятор
        if (document.getElementById('costCalculatorForm')) {
            steps.push({
                element: '#costCalculatorForm',
                popover: {
                    title: '💰 Калькулятор себестоимости',
                    description: 'Укажите цены компонентов, энергозатраты и логистику для мгновенного расчёта маржи и чистой прибыли партии.',
                    side: 'top',
                    align: 'start'
                }
            });
        }

        // Контекстный шаг: Сравнение составов
        if (document.querySelector('.compare-matrix') || document.getElementById('compositionsTable')) {
            steps.push({
                element: document.querySelector('.compare-matrix') || document.getElementById('compositionsTable'),
                popover: {
                    title: '⚖️ Матрица сравнения',
                    description: 'Наглядное сопоставление физико-химических свойств нескольких выбранных составов в единой таблице.',
                    side: 'top',
                    align: 'center'
                }
            });
        }

        // Фильтруем шаги: оставляем только те, элементы которых реально существуют в DOM
        return steps.filter(step => {
            if (typeof step.element === 'string') {
                return !!document.querySelector(step.element);
            }
            return !!step.element;
        });
    }

    function initTour() {
        if (typeof window.driver === 'undefined' || typeof window.driver.js === 'undefined') {
            console.warn('Driver.js not loaded yet');
            return;
        }

        const userRole = document.body.dataset.userRole || 'operator';
        const steps = buildTourSteps(userRole);

        if (!steps.length) return;

        const driverObj = window.driver.js.driver({
            showProgress: true,
            animate: true,
            allowClose: true,
            doneBtnText: 'Завершить',
            closeBtnText: 'Закрыть',
            nextBtnText: 'Далее →',
            prevBtnText: '← Назад',
            popoverClass: 'pellet-tour-popover',
            steps: steps,
            onDestroyStarted: () => {
                localStorage.setItem('pellet_tour_completed', 'true');
                driverObj.destroy();
            }
        });

        // Запуск по кнопке
        const startBtn = document.getElementById('startTourBtn');
        if (startBtn) {
            startBtn.addEventListener('click', function(e) {
                e.preventDefault();
                driverObj.drive();
            });
        }

        // Автозапуск при первом посещении:
        // НЕ запускаем автоматически на страницах с тяжёлыми графиками Plotly,
        // чтобы не прерывать рендеринг WebGL/SVG.
        const isPlotlyHeavyPage = !!document.getElementById('dynamicGraphContainer') || !!document.querySelector('.js-plotly-plot');
        const hasSeenTour = localStorage.getItem('pellet_tour_completed');

        if (!hasSeenTour && !isPlotlyHeavyPage) {
            setTimeout(() => {
                driverObj.drive();
            }, 1200);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initTour);
    } else {
        initTour();
    }
})();
