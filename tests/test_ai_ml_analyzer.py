import pytest
from unittest.mock import patch

from app.services.ai_ml_analyzer import AIMLAnalyzer


@pytest.fixture
def analyzer(test_db_path, sample_measured_data, sample_components_data):
    from app.database.database import init_db, insert_data
    init_db(test_db_path)
    insert_data(test_db_path, "measured_parameters", sample_measured_data)
    insert_data(test_db_path, "components", sample_components_data)
    return AIMLAnalyzer(test_db_path)


@pytest.fixture
def empty_analyzer(test_db_path):
    from app.database.database import init_db
    init_db(test_db_path)
    return AIMLAnalyzer(test_db_path)


class TestAIMLAnalyzer:
    def test_get_data_summary(self, analyzer):
        summary = analyzer.get_data_summary()
        assert summary["total_samples"] == 3
        assert summary["total_components"] == 3
        assert summary["status"] == "active"
        assert len(summary["available_properties"]) > 0
        assert summary["data_completeness"] > 0

    def test_get_data_summary_empty(self, empty_analyzer):
        summary = empty_analyzer.get_data_summary()
        assert summary["total_samples"] == 0
        assert summary["total_components"] == 0
        assert summary["status"] == "no_data"

    def test_get_ml_models_status_empty(self, empty_analyzer):
        status = empty_analyzer.get_ml_models_status()
        assert status["is_trained"] is False
        assert status["models_count"] == 0


class TestQueryClassification:
    def test_classify_query_optimization(self, empty_analyzer):
        assert empty_analyzer._classify_query("Как оптимизировать состав?") == "optimization"
        assert empty_analyzer._classify_query("Какой лучше") == "optimization"

    def test_classify_query_trends(self, empty_analyzer):
        assert empty_analyzer._classify_query("Покажи тренды") == "trends"
        assert empty_analyzer._classify_query("Есть ли зависимость") == "trends"

    def test_classify_query_prediction(self, empty_analyzer):
        assert empty_analyzer._classify_query("Нужно предсказать теплоту") == "prediction"
        assert empty_analyzer._classify_query("Сделай прогноз") == "prediction"

    def test_classify_query_composition(self, empty_analyzer):
        assert empty_analyzer._classify_query("Какой состав у пеллет") == "composition"
        assert empty_analyzer._classify_query("Какие компоненты нужны") == "composition"

    def test_classify_query_general(self, empty_analyzer):
        assert empty_analyzer._classify_query("Привет") == "general"


class TestAnalysis:
    def test_analyze_optimization(self, analyzer):
        result = analyzer.analyze_with_ai("Оптимизируй состав")
        assert result["success"] is True
        assert "analysis" in result
        assert "recommendations" in result
        assert "actions" in result

    def test_analyze_trends(self, analyzer):
        result = analyzer.analyze_with_ai("Покажи зависимости")
        assert result["success"] is True
        assert "analysis" in result

    def test_analyze_prediction(self, analyzer):
        result = analyzer.analyze_with_ai("Предскажи параметры")
        assert result["success"] is True
        assert "analysis" in result

    def test_analyze_composition(self, analyzer):
        result = analyzer.analyze_with_ai("Анализ составов")
        assert result["success"] is True
        assert "analysis" in result

    def test_analyze_general(self, analyzer):
        result = analyzer.analyze_with_ai("Общий вопрос")
        assert result["success"] is True
        assert "analysis" in result


class TestRecommendations:
    def test_get_system_recommendations_empty(self, empty_analyzer):
        recs = empty_analyzer.get_system_recommendations()
        assert recs["total_count"] > 0
        assert any(r["type"] == "data" for r in recs["recommendations"])
        assert any(r["type"] == "ml" for r in recs["recommendations"])


class TestAnalysisHistory:
    def test_get_analysis_history(self, analyzer):
        analyzer.analyze_with_ai("Первый вопрос")
        analyzer.analyze_with_ai("Второй вопрос")
        
        history = analyzer.get_analysis_history(limit=10)
        assert len(history) == 2
        assert history[0]["query"] == "Первый вопрос"
        assert history[1]["query"] == "Второй вопрос"
