"""
Simple Tests for ADK Agents

Tests for:
1. DataFetcher / Researcher Agent
2. Analyzer / Evaluator Agent
3. Summarizer / Reporter Agent
4. Pipeline orchestration

Run with: python -m pytest test_agents.py -v
"""

import pytest
from simple_adk_example import (
    DataFetcherAgent,
    AnalyzerAgent,
    SummarizerAgent,
    MultiAgentPipeline
)

from adk_real_example import (
    ResearcherAgent,
    EvaluatorAgent,
    ReporterAgent,
    AgentOrchestrator
)


# ============================================================================
# TESTS FOR SIMPLE ADK EXAMPLE
# ============================================================================

class TestDataFetcherAgent:
    """Test DataFetcher Agent"""

    def test_agent_creation(self):
        """Test agent can be created"""
        agent = DataFetcherAgent()
        assert agent.name == "DataFetcher"
        assert agent.role == "Gathers information and data from sources"

    def test_agent_model_default(self):
        """Test default model is set"""
        agent = DataFetcherAgent()
        assert agent.model == "gemini-1.5-flash"

    def test_agent_model_custom(self):
        """Test custom model can be set"""
        agent = DataFetcherAgent(model="gemini-2.0-pro")
        assert agent.model == "gemini-2.0-pro"

    def test_fetch_data_python(self):
        """Test fetching data for Python"""
        agent = DataFetcherAgent()
        result = agent.fetch_data("python")
        assert "python" in result.lower()
        assert len(result) > 0

    def test_fetch_data_javascript(self):
        """Test fetching data for JavaScript"""
        agent = DataFetcherAgent()
        result = agent.fetch_data("javascript")
        assert "javascript" in result.lower()

    def test_fetch_data_unknown(self):
        """Test fetching data for unknown topic"""
        agent = DataFetcherAgent()
        result = agent.fetch_data("unknown_topic_xyz")
        assert len(result) > 0


class TestAnalyzerAgent:
    """Test Analyzer Agent"""

    def test_agent_creation(self):
        """Test analyzer agent can be created"""
        agent = AnalyzerAgent()
        assert agent.name == "Analyzer"
        assert agent.role == "Analyzes data and provides insights"

    def test_agent_model_default(self):
        """Test default model"""
        agent = AnalyzerAgent()
        assert agent.model == "gemini-2.0-flash"

    def test_analyze_returns_string(self):
        """Test analyze method returns string"""
        agent = AnalyzerAgent()
        result = agent.analyze("Sample data", aspect="test")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_analyze_with_different_aspects(self):
        """Test analyze with different aspects"""
        agent = AnalyzerAgent()
        result1 = agent.analyze("Data", aspect="performance")
        result2 = agent.analyze("Data", aspect="usability")
        assert isinstance(result1, str)
        assert isinstance(result2, str)


class TestSummarizerAgent:
    """Test Summarizer Agent"""

    def test_agent_creation(self):
        """Test summarizer agent creation"""
        agent = SummarizerAgent()
        assert agent.name == "Summarizer"

    def test_summarize_short(self):
        """Test short summary"""
        agent = SummarizerAgent()
        analysis = "This is a test analysis"
        result = agent.summarize(analysis, length="short")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_long(self):
        """Test long summary"""
        agent = SummarizerAgent()
        analysis = "This is a test analysis"
        result = agent.summarize(analysis, length="long")
        assert isinstance(result, str)
        assert len(result) > 0


class TestMultiAgentPipeline:
    """Test the pipeline execution"""

    def test_pipeline_creation(self):
        """Test pipeline can be created"""
        pipeline = MultiAgentPipeline()
        assert pipeline.data_fetcher is not None
        assert pipeline.analyzer is not None
        assert pipeline.summarizer is not None

    def test_pipeline_execute(self):
        """Test pipeline execution"""
        pipeline = MultiAgentPipeline()
        result = pipeline.execute(topic="python", summary_length="short")

        assert isinstance(result, dict)
        assert "topic" in result
        assert "data" in result
        assert "analysis" in result
        assert "summary" in result
        assert result["topic"] == "python"

    def test_pipeline_output_quality(self):
        """Test pipeline outputs have content"""
        pipeline = MultiAgentPipeline()
        result = pipeline.execute(topic="python")

        assert len(result["data"]) > 0
        assert len(result["analysis"]) > 0
        assert len(result["summary"]) > 0

    def test_pipeline_multiple_topics(self):
        """Test pipeline with different topics"""
        pipeline = MultiAgentPipeline()

        for topic in ["python", "javascript", "go"]:
            result = pipeline.execute(topic=topic)
            assert result["topic"] == topic


# ============================================================================
# TESTS FOR REAL ADK EXAMPLE
# ============================================================================

class TestResearcherAgent:
    """Test Researcher Agent"""

    def test_agent_creation(self):
        """Test researcher agent creation"""
        agent = ResearcherAgent()
        assert agent.name == "Researcher"
        assert agent.role == "Investigates and gathers information"

    def test_agent_model(self):
        """Test model is set correctly"""
        agent = ResearcherAgent(model="gemini-2.0-flash")
        assert agent.model == "gemini-2.0-flash"

    def test_research_returns_string(self):
        """Test research method returns string"""
        agent = ResearcherAgent()
        result = agent.research("Machine Learning")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_agent_representation(self):
        """Test agent string representation"""
        agent = ResearcherAgent()
        repr_str = repr(agent)
        assert "Researcher" in repr_str


class TestEvaluatorAgent:
    """Test Evaluator Agent"""

    def test_agent_creation(self):
        """Test evaluator agent creation"""
        agent = EvaluatorAgent()
        assert agent.name == "Evaluator"

    def test_evaluate_returns_string(self):
        """Test evaluate method returns string"""
        agent = EvaluatorAgent()
        result = agent.evaluate("Sample research findings")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_evaluate_with_criteria(self):
        """Test evaluate with different criteria"""
        agent = EvaluatorAgent()
        result1 = agent.evaluate("Data", criteria="impact")
        result2 = agent.evaluate("Data", criteria="feasibility")
        assert isinstance(result1, str)
        assert isinstance(result2, str)


class TestReporterAgent:
    """Test Reporter Agent"""

    def test_agent_creation(self):
        """Test reporter agent creation"""
        agent = ReporterAgent()
        assert agent.name == "Reporter"

    def test_create_report_executive(self):
        """Test executive report creation"""
        agent = ReporterAgent()
        result = agent.create_report(
            topic="AI",
            evaluation="Evaluation data",
            format="executive"
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_create_report_detailed(self):
        """Test detailed report creation"""
        agent = ReporterAgent()
        result = agent.create_report(
            topic="AI",
            evaluation="Evaluation data",
            format="detailed"
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestAgentOrchestrator:
    """Test the orchestrator"""

    def test_orchestrator_creation(self):
        """Test orchestrator creation"""
        orchestrator = AgentOrchestrator()
        assert orchestrator.researcher is not None
        assert orchestrator.evaluator is not None
        assert orchestrator.reporter is not None

    def test_execute_research_pipeline(self):
        """Test research pipeline execution"""
        orchestrator = AgentOrchestrator()
        result = orchestrator.execute_research_pipeline(
            topic="AI",
            evaluation_criteria="impact"
        )

        assert isinstance(result, dict)
        assert "topic" in result
        assert "research" in result
        assert "evaluation" in result
        assert "report" in result
        assert "agents_used" in result
        assert result["topic"] == "AI"

    def test_pipeline_output_structure(self):
        """Test pipeline output has all components"""
        orchestrator = AgentOrchestrator()
        result = orchestrator.execute_research_pipeline(topic="Test")

        assert len(result["research"]) > 0
        assert len(result["evaluation"]) > 0
        assert len(result["report"]) > 0
        assert len(result["agents_used"]) == 3

    def test_print_result(self, capsys):
        """Test print result doesn't crash"""
        orchestrator = AgentOrchestrator()
        result = orchestrator.execute_research_pipeline(topic="Test")

        # Should not raise exception
        orchestrator.print_result(result)

        # Check output was printed
        captured = capsys.readouterr()
        assert "RESEARCH RESULTS" in captured.out


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete pipelines"""

    def test_simple_pipeline_full_execution(self):
        """Test full simple pipeline execution"""
        pipeline = MultiAgentPipeline()
        result = pipeline.execute(topic="python", summary_length="short")

        assert result is not None
        assert all(key in result for key in ["topic", "data", "analysis", "summary"])

    def test_real_pipeline_full_execution(self):
        """Test full real pipeline execution"""
        orchestrator = AgentOrchestrator()
        result = orchestrator.execute_research_pipeline(
            topic="Quantum Computing",
            evaluation_criteria="feasibility",
            report_format="executive"
        )

        assert result is not None
        assert all(key in result for key in [
            "topic", "research", "evaluation", "report", "agents_used"
        ])

    def test_agent_count_consistency(self):
        """Test correct number of agents in results"""
        pipeline1 = MultiAgentPipeline()
        result1 = pipeline1.execute("python")
        # Simple pipeline uses 3 agents internally

        orchestrator = AgentOrchestrator()
        result2 = orchestrator.execute_research_pipeline("python")
        assert len(result2["agents_used"]) == 3


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance and efficiency tests"""

    def test_agent_execution_speed(self):
        """Test agents execute quickly (mocked, no API calls)"""
        import time

        agent = DataFetcherAgent()
        start = time.time()
        agent.fetch_data("python")
        elapsed = time.time() - start

        # Should be very fast since mocked (no API calls)
        assert elapsed < 1.0, f"Agent took {elapsed}s, should be < 1s"

    def test_pipeline_execution_speed(self):
        """Test pipeline executes efficiently"""
        import time

        pipeline = MultiAgentPipeline()
        start = time.time()
        pipeline.execute("python")
        elapsed = time.time() - start

        # Should complete quickly (mocked, no API calls)
        assert elapsed < 5.0, f"Pipeline took {elapsed}s, should be < 5s"


if __name__ == "__main__":
    # Run tests with: python -m pytest test_agents.py -v
    print("Run tests with: pytest test_agents.py -v")
