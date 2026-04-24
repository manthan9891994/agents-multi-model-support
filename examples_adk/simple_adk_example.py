"""
Simple Google ADK Multi-Agent Example

This example demonstrates 3 simple agents working together:
1. DataFetcherAgent - Gathers information
2. AnalyzerAgent - Analyzes the data
3. SummarizerAgent - Creates a summary

Keep it simple - no complex orchestration, just straightforward agent execution.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# For this example, we'll use a simple simulation since ADK requires specific setup
# In production, you would use: from google.generativeai import adk

class SimpleAgent:
    """Simple agent wrapper for ADK-style agents"""

    def __init__(self, name: str, role: str, model: str = "gemini-2.0-flash"):
        self.name = name
        self.role = role
        self.model = model

    def execute(self, task: str) -> str:
        """Execute the agent with a task"""
        print(f"\n{'='*60}")
        print(f"Agent: {self.name}")
        print(f"Role: {self.role}")
        print(f"Task: {task}")
        print(f"Model: {self.model}")
        print(f"{'='*60}")
        return f"Result from {self.name}: {task}"


class DataFetcherAgent(SimpleAgent):
    """
    Agent responsible for fetching/gathering information
    Simple task: gather data about a topic
    """

    def __init__(self, model: str = "gemini-1.5-flash"):
        super().__init__(
            name="DataFetcher",
            role="Gathers information and data from sources",
            model=model
        )

    def fetch_data(self, query: str) -> str:
        """Fetch data for a given query"""
        print(f"\n[FETCH] DataFetcher fetching information about: {query}")

        sample_data = {
            "python": {
                "description": "Python is a high-level programming language",
                "year_created": 1991,
                "creator": "Guido van Rossum",
                "popularity": "Very High",
                "use_cases": ["Web Development", "Data Science", "AI/ML", "Automation"]
            },
            "javascript": {
                "description": "JavaScript is a web scripting language",
                "year_created": 1995,
                "creator": "Brendan Eich",
                "popularity": "Very High",
                "use_cases": ["Web Development", "Frontend", "Node.js", "Mobile Apps"]
            },
            "go": {
                "description": "Go is a statically typed compiled language",
                "year_created": 2009,
                "creator": "Google",
                "popularity": "High",
                "use_cases": ["Microservices", "Cloud", "DevOps", "System Programming"]
            }
        }

        data = sample_data.get(query.lower(), {
            "description": "Unknown topic",
            "data": "No data available"
        })

        result = f"Data for '{query}':\n"
        for key, value in data.items():
            result += f"  - {key}: {value}\n"

        print(f"[OK] Data fetched successfully")
        return result


class AnalyzerAgent(SimpleAgent):
    """
    Agent responsible for analyzing information
    Simple task: analyze the fetched data
    """

    def __init__(self, model: str = "gemini-2.0-flash"):
        super().__init__(
            name="Analyzer",
            role="Analyzes data and provides insights",
            model=model
        )

    def analyze(self, data: str, aspect: str = "general") -> str:
        """Analyze the provided data"""
        print(f"\n[ANALYZE] Analyzer analyzing data for: {aspect}")

        analysis = f"""
Analysis Report - {aspect}:
{'='*50}

Key Insights:
1. Popularity & Adoption: High level of industry adoption
2. Use Cases: Multiple domain applications
3. Community: Strong community support
4. Learning Curve: Varies by language
5. Performance: Trade-offs between readability and speed

Recommendation: Choose based on specific use case needs

Detailed Analysis:
- Development Speed: Important for rapid prototyping
- Performance: Critical for high-scale systems
- Ecosystem: Mature with extensive libraries
- Market Demand: All three are highly sought after

Conclusion:
Each language excels in different domains. The choice
depends on project requirements and team expertise.
"""

        print(f"[OK] Analysis complete")
        return analysis


class SummarizerAgent(SimpleAgent):
    """
    Agent responsible for creating summaries
    Simple task: summarize the analysis
    """

    def __init__(self, model: str = "gemini-1.5-flash"):
        super().__init__(
            name="Summarizer",
            role="Creates concise summaries of analysis",
            model=model
        )

    def summarize(self, analysis: str, length: str = "short") -> str:
        """Summarize the analysis"""
        print(f"\n[WRITE] Summarizer creating {length} summary")

        if length == "short":
            summary = """
Executive Summary
===============
All three languages (Python, JavaScript, Go) are powerful
and widely used. Python is best for data science and AI,
JavaScript dominates web development, and Go excels in
microservices and cloud infrastructure.

Quick Recommendation: Choose based on your project domain.
"""
        else:
            summary = """
Comprehensive Summary
====================
After analyzing Python, JavaScript, and Go:

Python: Best for AI/ML, Data Science, rapid development
- Strengths: Easy to learn, rich ecosystem, great for data
- Use when: Building ML models, data analysis, automation

JavaScript: Dominates web and full-stack development
- Strengths: Runs everywhere, large ecosystem, fast to develop
- Use when: Building web applications, frontend/backend

Go: Perfect for systems and cloud infrastructure
- Strengths: High performance, built for concurrency, simple
- Use when: Building microservices, cloud tools, DevOps

Final Verdict: No single "best" language. Match language
to project requirements and team expertise.
"""

        print(f"[OK] Summary created")
        return summary


class MultiAgentPipeline:
    """
    Simple pipeline that chains agents together
    Keep it straightforward - one agent output feeds to next
    """

    def __init__(self):
        self.data_fetcher = DataFetcherAgent()
        self.analyzer = AnalyzerAgent()
        self.summarizer = SummarizerAgent()

    def execute(self, topic: str, summary_length: str = "short") -> dict:
        """
        Execute the pipeline: Fetch -> Analyze -> Summarize

        Args:
            topic: Topic to research (e.g., "python", "javascript")
            summary_length: "short" or "long"

        Returns:
            Dictionary with results from each agent
        """

        print(f"\n[RUN] Starting Multi-Agent Pipeline for topic: {topic}")
        print(f"{'='*70}")

        print(f"\n[STEP 1] Data Fetching")
        data = self.data_fetcher.fetch_data(topic)

        print(f"\n[STEP 2] Data Analysis")
        analysis = self.analyzer.analyze(data, aspect=f"{topic} comparison")

        print(f"\n[STEP 3] Summary Creation")
        summary = self.summarizer.summarize(analysis, length=summary_length)

        print(f"\n{'='*70}")
        print(f"[OK] Pipeline Complete!")

        return {
            "topic": topic,
            "data": data,
            "analysis": analysis,
            "summary": summary
        }


def main():
    """Run the simple ADK example"""

    print("\n" + "="*70)
    print("SIMPLE GOOGLE ADK MULTI-AGENT EXAMPLE")
    print("="*70)
    print("\nDemonstration of 3 simple agents:")
    print("1. DataFetcherAgent - Gathers information")
    print("2. AnalyzerAgent - Analyzes the data")
    print("3. SummarizerAgent - Summarizes findings")

    # Create pipeline
    pipeline = MultiAgentPipeline()

    # Execute for Python
    result = pipeline.execute(topic="python", summary_length="short")

    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(result["summary"])

    print("\nTo test with other topics, uncomment the code at the bottom of main()")


if __name__ == "__main__":
    main()
