/**
 * static/js/app_tour.js — Интерактивный тур онбординга по системе Pellets Analyzer.
 * Использует вендоренную библиотеку driver.js без зависимостей.
 */
(function() {
    'use strict';

    function initTour() {
        if (typeof window.driver === 'undefined' || typeof window.driver.js === 'undefined') {
            console.warn('Driver.js not loaded yet');
            return;
        }

        const driverObj = window.driver.js.driver({
            showProgress: true,
            animate: true,
            allowClose: true,
            doneBtnText: 'Завершить',
            closeBtnText: 'Закрыть',
            nextBtnText: 'Далее →',
            prevBtnText: '← Назад',
            popoverClass: 'pellet-tour-popover',
            steps: [
                {
                    element: '#navbar',
                    popover: {
                        title: '👋 Добро пожаловать в Pellets Analyzer!',
                        description: 'Аналитическая платформа для расчёта теплофизических свойств, подбора оптимальных рецептур и экономической оценки топливных гранул.',
                        side: 'bottom',
                        align: 'start'
                    }
                },
                {
                    element: '#globalSearchInput',
                    popover: {
                        title: '🔍 Быстрый поиск составов',
                        description: 'Мгновенный поиск рецептур по названию сырья прямо из верхней панели с подсказками теплоты и зольности.',
                        side: 'bottom',
                        align: 'center'
                    }
                },
                {
                    element: '#notificationsDropdownWrapper',
                    popover: {
                        title: '🔔 Центр уведомлений',
                        description: 'Здесь отображаются статус активной модели ML (v2 / legacy), актуальный объём базы данных и ход фоновых расчётов.',
                        side: 'bottom',
                        align: 'end'
                    }
                },
                {
                    element: '#themeToggleBtn',
                    popover: {
                        title: '🌓 Переключатель темы',
                        description: 'Быстрое переключение между комфортной ночной темой и контрастной дневной темой AstroWind.',
                        side: 'bottom',
                        align: 'end'
                    }
                },
                {
                    element: '#sidenav-main',
                    popover: {
                        title: '🧭 Навигация по модулям',
                        description: 'Переходите между Базой данных, Лабораторией ИИ, Калькулятором экономики, Сравнением составов и Графиками.',
                        side: 'right',
                        align: 'start'
                    }
                }
            ],
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

        // Автозапуск при первом посещении
        const hasSeenTour = localStorage.getItem('pellet_tour_completed');
        if (!hasSeenTour) {
            // Небольшая задержка после первой загрузки страницы
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
