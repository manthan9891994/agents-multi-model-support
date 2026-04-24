# Multi-Layer Task Classification Framework
## Comprehensive Implementation Plan

**Project**: Dynamic Model Selection for Agentic Frameworks  
**Version**: 1.0  
**Last Updated**: 2026-04-24  
**Status**: Ready for Implementation  

---

## Table of Contents

1. [Executive Overview](#executive-overview)
2. [Architecture Design](#architecture-design)
3. [Phase-by-Phase Roadmap](#phase-by-phase-roadmap)
4. [Component Specifications](#component-specifications)
5. [Integration Points](#integration-points)
6. [Testing Strategy](#testing-strategy)
7. [Deployment & Operations](#deployment--operations)

---

## Executive Overview

### Problem Statement
Current agentic frameworks (CrewAI, Google ADK) lack intelligent model selection based on task complexity. This causes:
- Wasted API spend on simple tasks using expensive models
- Potential failures on complex tasks using cheap models
- Inability to dynamically adapt to changing request patterns

### Solution: Multi-Layer Classification Framework
A **4-layer classification system** that progressively evaluates task complexity using:
1. **Layer 1**: Fast Python heuristics (regex, keyword matching, token counting)
2. **Layer 2**: On-premise ML model (Gemma-2/4 local inference)
3. **Layer 3**: Lightweight API classifier (Claude Haiku / GPT-4o-Mini)
4. **Layer 4**: NLP + ML ensemble (scikit-learn + transformers)

Each layer has different speed/accuracy/cost trade-offs. The framework uses them intelligently based on:
- Request characteristics (complexity signals)
- System load
- Cost budget
- Latency requirements

### Expected Outcomes
- **Cost Reduction**: 40-60% lower API spend
- **Performance**: Sub-100ms classification for 90% of requests
- **Accuracy**: >95% correct model selection
- **Extensibility**: Easy to add new models/classifiers

---

## Architecture Design

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER REQUEST                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│         REQUEST PREPROCESSING & FEATURE EXTRACTION               │
│  ├─ Token counting                                              │
│  ├─ Language detection                                          │
│  ├─ Data volume estimation                                      │
│  └─ Context enrichment                                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│         MULTI-LAYER CLASSIFICATION PIPELINE                      │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ LAYER 1: Python Heuristics (< 10ms)                     │   │
│  │ IF confidence > 0.95: RETURN (SIMPLE/STANDARD/etc)      │   │
│  └──────────────────────┬────────────────────────────────┘    │
│                         │ confidence < 0.95                     │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ LAYER 2: On-Prem Model (Gemma-2/4) (50-200ms)          │   │
│  │ IF confidence > 0.90: RETURN                             │   │
│  │ IF cache hit: SKIP to Layer 3                            │   │
│  └──────────────────────┬────────────────────────────────┘    │
│                         │ confidence < 0.90                     │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ LAYER 3: API Classifier (Claude/GPT-4o-Mini) (300-800ms)│   │
│  │ Cost: ~$0.0002 per request (negligible)                 │   │
│  │ IF confidence > 0.85: RETURN                             │   │
│  └──────────────────────┬────────────────────────────────┘    │
│                         │ confidence < 0.85                     │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ LAYER 4: ML Ensemble (300-500ms) [ASYNC, optional]     │   │
│  │ Return Layer 3 result + trigger async ML training       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└──────────────────────┬─────────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  ROUTING DECISION + LOGGING  │
         │  {model, complexity, reason} │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  AGENT EXECUTION WITH MODEL │
         │  (CrewAI / Google ADK)       │
         └─────────────────────────────┘
```

### Component Interaction Flow

```python
# Pseudocode: How layers interact

def classify(request, context):
    """Main classification pipeline"""
    
    # Step 1: Extract features
    features = extract_features(request, context)
    
    # Step 2: Layer 1 - Fast heuristics
    layer1_result = heuristic_classifier.classify(features)
    if layer1_result.confidence > 0.95:
        return layer1_result  # Early exit, < 10ms
    
    # Step 3: Layer 2 - On-prem model
    layer2_result = onprem_classifier.classify(request)
    if layer2_result.confidence > 0.90:
        return layer2_result  # 50-200ms
    
    # Step 4: Layer 3 - API classifier
    layer3_result = api_classifier.classify(request)
    
    # Step 5: Layer 4 - Async ML ensemble (background)
    async_task = background.enqueue(
        ml_ensemble.train_and_evaluate,
        request, layer3_result
    )
    
    return layer3_result  # Return Layer 3, Layer 4 improves async
```

### Data Flow

```
Request Input
    ↓
Feature Extractor
    ├─ token_count
    ├─ keyword_signals
    ├─ data_volume
    ├─ nesting_depth
    ├─ tool_requirements
    ├─ language
    └─ domain_tags
    ↓
Classification Layers (parallel/serial)
    ├─ Heuristic Engine
    ├─ Gemma-2/4 Local
    ├─ Claude Haiku / GPT-4o-Mini
    └─ ML Ensemble
    ↓
Consensus Engine (vote/average confidence)
    ↓
Routing Decision
    ├─ model_name
    ├─ complexity_tier
    ├─ confidence_score
    ├─ layer_used
    └─ reasoning
    ↓
Model Selection & Agent Execution
```

---

## Phase-by-Phase Roadmap

### Phase 0: Foundation Setup (Days 1-2)
**Goal**: Set up project structure and infrastructure  
**Owner**: DevOps/Infrastructure  
**Deliverables**: Working development environment

#### Steps:

**0.1 - Create Project Structure**
```
model-classifier-framework/
├── classifier/
│   ├── __init__.py
│   ├── core.py                 # Main ClassificationPipeline
│   ├── types.py                # Data classes (Request, Decision, etc)
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── layer1_heuristics.py
│   │   ├── layer2_onprem.py
│   │   ├── layer3_api.py
│   │   └── layer4_ml_ensemble.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── extractor.py
│   │   └── cache.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py         # Model selector (which Haiku, which Gemma)
│   │   └── config.py           # Model configurations
│   ├── adapters/
│   │   ├── crewai_adapter.py
│   │   ├── adk_adapter.py
│   │   └── base_adapter.py
│   ├── logging/
│   │   ├── __init__.py
│   │   └── structured_logger.py
│   ├── cache/
│   │   ├── __init__.py
│   │   └── redis_cache.py      # Optional: for distributed caching
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── performance/
├── examples/
│   ├── crewai_example.py
│   └── adk_example.py
├── config/
│   ├── layer_config.yaml       # Layer thresholds, model mappings
│   ├── feature_config.yaml     # Feature extraction settings
│   └── model_config.yaml       # Model endpoints, keys
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── examples.md
├── requirements.txt
├── setup.py
└── README.md
```

**0.2 - Dependencies Installation**
```bash
pip install \
  anthropic \
  openai \
  scikit-learn \
  numpy \
  pandas \
  pydantic \
  pyyaml \
  redis \
  ollama \
  crewai \
  google-generativeai
```

**0.3 - Environment Setup**
```bash
# .env template
OPENAI_API_KEY=xxx
ANTHROPIC_API_KEY=xxx
GEMMA_MODEL_PATH=/models/gemma-2-7b
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
ENABLE_LAYER_2=true
ENABLE_LAYER_3=true
ENABLE_LAYER_4=false  # Start with false
```

---

### Phase 1: Layer 1 - Python Heuristics (Days 3-5)
**Goal**: Fast, rule-based task classification  
**Owner**: ML Engineer  
**Deliverables**: HeuristicClassifier working, 90%+ accuracy on known tasks  
**Target Performance**: < 10ms per request  

#### 1.1 - Define Task Complexity Tiers

```python
# classifier/types.py

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional

class TaskComplexity(Enum):
    """Task complexity tiers"""
    SIMPLE = "simple"          # Basic, factual, <500 tokens
    STANDARD = "standard"      # Moderate reasoning, 500-5k tokens
    COMPLEX = "complex"        # Deep reasoning, multi-step, 5k-15k tokens
    RESEARCH = "research"      # Comprehensive analysis, >15k tokens

@dataclass
class ClassificationSignals:
    """Feature signals extracted from request"""
    token_count: int
    keyword_scores: Dict[str, float]
    data_volume_mb: float
    has_tools: bool
    has_planning: bool
    has_reasoning: bool
    nesting_depth: int
    language: str
    domain_tags: List[str]

@dataclass
class LayerResult:
    """Result from a single classification layer"""
    complexity: TaskComplexity
    confidence: float          # 0.0-1.0
    reason: str
    layer_name: str
    latency_ms: float
    signals_used: ClassificationSignals

@dataclass
class ClassificationDecision:
    """Final routing decision"""
    model_name: str
    complexity: TaskComplexity
    tier: str                  # "light" / "standard" / "heavy" / "research"
    reasoning: str
    confidence: float
    layer_used: str            # "layer1" / "layer2" / "layer3" / "layer4"
    all_results: List[LayerResult]
    estimated_cost: float      # Estimated API call cost
    signals: ClassificationSignals
```

#### 1.2 - Feature Extraction

```python
# classifier/features/extractor.py

import re
from typing import Dict, List
from classifier.types import ClassificationSignals

class FeatureExtractor:
    """Extract signals from requests"""
    
    # Keyword definitions
    REASONING_KEYWORDS = [
        "analyze", "explain", "reason", "why", "how", "compare",
        "contrast", "evaluate", "assess", "interpret", "debate"
    ]
    
    PLANNING_KEYWORDS = [
        "step", "plan", "process", "workflow", "pipeline", "sequence",
        "stages", "phases", "organize", "structure"
    ]
    
    RESEARCH_KEYWORDS = [
        "research", "investigate", "comprehensive", "deep dive",
        "all aspects", "thorough", "exhaustive", "survey"
    ]
    
    CODING_KEYWORDS = [
        "code", "function", "implement", "debug", "algorithm",
        "optimize", "refactor", "design pattern"
    ]
    
    SIMPLE_KEYWORDS = [
        "what is", "define", "list", "explain briefly",
        "summarize", "simple"
    ]
    
    def extract(self, request: str, context: Dict = None) -> ClassificationSignals:
        """Extract all features from request"""
        
        token_count = self._count_tokens(request)
        keyword_scores = self._score_all_keywords(request)
        nesting_depth = self._calculate_nesting_depth(request)
        language = self._detect_language(request)
        domain_tags = self._extract_domain_tags(request)
        
        # Data volume from context
        data_volume_mb = 0.0
        if context and "data_size" in context:
            data_volume_mb = context["data_size"] / (1024 * 1024)
        
        # Tool requirements
        has_tools = context and context.get("requires_tools", False)
        
        # Planning and reasoning signals
        has_planning = any(kw in request.lower() for kw in self.PLANNING_KEYWORDS)
        has_reasoning = keyword_scores.get("reasoning", 0) > 0.5
        
        return ClassificationSignals(
            token_count=token_count,
            keyword_scores=keyword_scores,
            data_volume_mb=data_volume_mb,
            has_tools=has_tools,
            has_planning=has_planning,
            has_reasoning=has_reasoning,
            nesting_depth=nesting_depth,
            language=language,
            domain_tags=domain_tags
        )
    
    def _count_tokens(self, text: str) -> int:
        """Estimate tokens: 1 token ≈ 4 chars (rough estimate)"""
        return max(1, len(text) // 4)
    
    def _score_all_keywords(self, text: str) -> Dict[str, float]:
        """Score text against all keyword categories"""
        text_lower = text.lower()
        
        return {
            "reasoning": self._score_keywords(text_lower, self.REASONING_KEYWORDS),
            "planning": self._score_keywords(text_lower, self.PLANNING_KEYWORDS),
            "research": self._score_keywords(text_lower, self.RESEARCH_KEYWORDS),
            "coding": self._score_keywords(text_lower, self.CODING_KEYWORDS),
            "simple": self._score_keywords(text_lower, self.SIMPLE_KEYWORDS),
        }
    
    def _score_keywords(self, text: str, keywords: List[str]) -> float:
        """Score 0-1 based on keyword presence"""
        if not keywords:
            return 0.0
        
        matches = sum(1 for kw in keywords if kw in text)
        return min(matches / len(keywords), 1.0)
    
    def _calculate_nesting_depth(self, text: str) -> int:
        """Estimate nesting/complexity from brackets, indentation"""
        max_brackets = 0
        current = 0
        for char in text:
            if char in "([{":
                current += 1
                max_brackets = max(max_brackets, current)
            elif char in ")]}":
                current -= 1
        return max_brackets
    
    def _detect_language(self, text: str) -> str:
        """Detect language (simple heuristic)"""
        # For MVP: just detect if it contains code snippets
        if "def " in text or "function " in text or "import " in text:
            return "code"
        return "natural_language"
    
    def _extract_domain_tags(self, text: str) -> List[str]:
        """Extract domain hints"""
        tags = []
        domains = {
            "code": ["python", "javascript", "sql", "java", "go"],
            "data": ["dataframe", "csv", "database", "query"],
            "math": ["equation", "calculate", "formula", "algorithm"],
            "nlp": ["text", "language", "sentiment", "classification"],
        }
        
        for domain, keywords in domains.items():
            if any(kw in text.lower() for kw in keywords):
                tags.append(domain)
        
        return tags
```

#### 1.3 - Heuristic Classifier

```python
# classifier/layers/layer1_heuristics.py

import time
from classifier.types import TaskComplexity, LayerResult, ClassificationSignals

class HeuristicClassifier:
    """Fast rule-based classifier using feature signals"""
    
    def __init__(self):
        self.name = "layer1_heuristics"
    
    def classify(self, signals: ClassificationSignals) -> LayerResult:
        """Classify based on feature signals"""
        start_time = time.time()
        
        # Decision tree logic
        complexity, confidence, reason = self._decide(signals)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LayerResult(
            complexity=complexity,
            confidence=confidence,
            reason=reason,
            layer_name=self.name,
            latency_ms=latency_ms,
            signals_used=signals
        )
    
    def _decide(self, signals: ClassificationSignals) -> tuple:
        """Decision tree: signals → complexity"""
        
        # Rule 1: Token count is primary signal
        if signals.token_count > 15000:
            base_tier = TaskComplexity.RESEARCH
        elif signals.token_count > 5000:
            base_tier = TaskComplexity.COMPLEX
        elif signals.token_count > 500:
            base_tier = TaskComplexity.STANDARD
        else:
            base_tier = TaskComplexity.SIMPLE
        
        # Rule 2: Adjust by keyword signals
        keyword_max = max(signals.keyword_scores.values())
        
        if signals.keyword_scores["research"] > 0.6:
            adjusted_tier = TaskComplexity.RESEARCH
            reason = "Research keywords detected"
        elif signals.keyword_scores["reasoning"] > 0.7:
            adjusted_tier = max(base_tier, TaskComplexity.COMPLEX)
            reason = "Deep reasoning required"
        elif signals.keyword_scores["planning"] > 0.6:
            adjusted_tier = max(base_tier, TaskComplexity.STANDARD)
            reason = "Multi-step planning detected"
        elif signals.keyword_scores["simple"] > 0.5:
            adjusted_tier = TaskComplexity.SIMPLE
            reason = "Simple, factual question"
        else:
            adjusted_tier = base_tier
            reason = f"Token count based: {signals.token_count} tokens"
        
        # Rule 3: Boost by data volume
        if signals.data_volume_mb > 10:
            adjusted_tier = max(adjusted_tier, TaskComplexity.RESEARCH)
            reason += f" + large data ({signals.data_volume_mb}MB)"
        
        # Rule 4: Tools require moderate capability
        if signals.has_tools:
            adjusted_tier = max(adjusted_tier, TaskComplexity.STANDARD)
            reason += " + requires tools"
        
        # Calculate confidence based on signal agreement
        confidence = self._calculate_confidence(signals, adjusted_tier)
        
        return adjusted_tier, confidence, reason
    
    def _calculate_confidence(self, signals: ClassificationSignals, tier: TaskComplexity) -> float:
        """Confidence 0-1: higher if multiple signals agree"""
        
        agreement_score = 0
        weight = 0
        
        # Token count confidence
        if tier == TaskComplexity.SIMPLE and signals.token_count < 500:
            agreement_score += 0.3
        elif tier == TaskComplexity.STANDARD and 500 <= signals.token_count < 5000:
            agreement_score += 0.3
        elif tier == TaskComplexity.COMPLEX and 5000 <= signals.token_count < 15000:
            agreement_score += 0.3
        elif tier == TaskComplexity.RESEARCH and signals.token_count > 15000:
            agreement_score += 0.3
        weight += 0.3
        
        # Keyword agreement
        if tier == TaskComplexity.RESEARCH and signals.keyword_scores["research"] > 0.5:
            agreement_score += 0.2
        elif tier == TaskComplexity.COMPLEX and signals.keyword_scores["reasoning"] > 0.5:
            agreement_score += 0.2
        elif tier == TaskComplexity.SIMPLE and signals.keyword_scores["simple"] > 0.5:
            agreement_score += 0.2
        weight += 0.2
        
        # Domain clarity
        if signals.domain_tags:
            agreement_score += 0.1
        weight += 0.1
        
        # Language clarity (code vs natural)
        if signals.language != "unknown":
            agreement_score += 0.1
        weight += 0.1
        
        # Data volume agreement
        if signals.data_volume_mb > 0 and tier == TaskComplexity.RESEARCH:
            agreement_score += 0.2
        elif signals.data_volume_mb == 0 and tier != TaskComplexity.RESEARCH:
            agreement_score += 0.1
        weight += 0.2
        
        # Normalize
        confidence = agreement_score / weight if weight > 0 else 0.5
        
        # Cap at reasonable bounds
        return min(max(confidence, 0.0), 1.0)
```

#### 1.4 - Testing Layer 1

```python
# tests/unit/test_layer1_heuristics.py

import pytest
from classifier.features.extractor import FeatureExtractor
from classifier.layers.layer1_heuristics import HeuristicClassifier
from classifier.types import TaskComplexity

class TestHeuristicClassifier:
    
    @pytest.fixture
    def setup(self):
        self.extractor = FeatureExtractor()
        self.classifier = HeuristicClassifier()
    
    def test_simple_question(self, setup):
        """Simple factual question → SIMPLE tier"""
        request = "What is the capital of France?"
        signals = self.extractor.extract(request)
        result = self.classifier.classify(signals)
        
        assert result.complexity == TaskComplexity.SIMPLE
        assert result.confidence > 0.8
        assert result.latency_ms < 10
    
    def test_complex_reasoning(self, setup):
        """Complex multi-step reasoning → COMPLEX tier"""
        request = """
        Analyze the economic impact of AI on software engineering.
        Compare and contrast with historical parallels from:
        1. Industrial revolution
        2. Computer revolution
        3. Internet era
        
        Consider: jobs, salaries, skill requirements, market growth.
        """
        signals = self.extractor.extract(request)
        result = self.classifier.classify(signals)
        
        assert result.complexity in [TaskComplexity.COMPLEX, TaskComplexity.RESEARCH]
        assert result.confidence > 0.7
    
    def test_research_task(self, setup):
        """Research task with keywords → RESEARCH tier"""
        request = """
        Conduct a comprehensive research on the following:
        - Survey all recent AI models (2024-2026)
        - Investigate their performance metrics
        - Analyze cost-benefit trade-offs
        - Provide exhaustive comparison table
        - Deep dive into architectural differences
        """
        signals = self.extractor.extract(request)
        result = self.classifier.classify(signals)
        
        assert result.complexity == TaskComplexity.RESEARCH
        assert result.confidence > 0.8
    
    def test_large_data_volume(self, setup):
        """Large data → escalate complexity"""
        request = "Analyze the data"
        context = {"data_size": 50 * 1024 * 1024}  # 50MB
        
        signals = self.extractor.extract(request, context)
        result = self.classifier.classify(signals)
        
        assert result.complexity in [TaskComplexity.COMPLEX, TaskComplexity.RESEARCH]
        assert "large data" in result.reason.lower()
    
    def test_performance_requirement(self, setup):
        """Layer 1 must be < 10ms"""
        request = "What is 2+2?" * 100
        signals = self.extractor.extract(request)
        result = self.classifier.classify(signals)
        
        assert result.latency_ms < 10, f"Layer 1 took {result.latency_ms}ms, should be < 10ms"
```

---

### Phase 2: Layer 2 - On-Premise Model (Gemma-2/4) (Days 6-10)
**Goal**: Fast local inference for complex cases  
**Owner**: ML Engineer + DevOps  
**Deliverables**: Gemma-2/4 running locally, classification working, 100-200ms latency  
**Target Performance**: 50-200ms per request  

#### 2.1 - Setup Gemma-2 Local Model

```bash
# Install Ollama: https://ollama.ai
# Pull Gemma-2 model
ollama pull gemma2:7b

# Verify it's running
curl http://localhost:11434/api/generate -d '{"model":"gemma2","prompt":"test"}'
```

#### 2.2 - Gemma-2 Classifier

```python
# classifier/layers/layer2_onprem.py

import requests
import json
import time
from typing import Optional
from classifier.types import TaskComplexity, LayerResult, ClassificationSignals

class GemmaClassifier:
    """On-premise Gemma-2/4 classification"""
    
    def __init__(self, model_name: str = "gemma2:7b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.name = "layer2_onprem_gemma"
        self.prompt_template = self._build_prompt_template()
    
    def _build_prompt_template(self) -> str:
        """System prompt for Gemma classification"""
        return """You are an expert task complexity classifier. Analyze the following task and classify it.

Respond ONLY with JSON (no markdown, no explanation):
{
  "complexity": "simple|standard|complex|research",
  "confidence": 0.0-1.0,
  "reason": "brief explanation"
}

Classification guide:
- SIMPLE: Basic factual question, definition, simple lookup. <500 tokens context. Examples: "What is X?", "Define Y"
- STANDARD: Moderate reasoning, moderate length. 500-5000 tokens. Examples: "Compare X and Y", "Explain how Z works"
- COMPLEX: Deep reasoning, multi-step logic, problem-solving. 5000-15000 tokens. Examples: "Design a system for X", "Analyze the impact of Y"
- RESEARCH: Comprehensive analysis, long-form, multiple aspects. >15000 tokens. Examples: "Research and summarize all aspects of X", "Investigate Y across multiple domains"

Task to classify:
{task}"""
    
    def classify(self, signals: ClassificationSignals, request: str) -> LayerResult:
        """Classify using local Gemma model"""
        start_time = time.time()
        
        try:
            # Build prompt
            prompt = self.prompt_template.format(task=request[:3000])  # Limit input
            
            # Call Gemma via Ollama
            response = self._call_ollama(prompt)
            
            # Parse response
            complexity, confidence, reason = self._parse_response(response)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LayerResult(
                complexity=complexity,
                confidence=confidence,
                reason=reason,
                layer_name=self.name,
                latency_ms=latency_ms,
                signals_used=signals
            )
        
        except Exception as e:
            # Fallback on error
            latency_ms = (time.time() - start_time) * 1000
            return LayerResult(
                complexity=TaskComplexity.STANDARD,
                confidence=0.0,
                reason=f"Gemma error (fallback): {str(e)}",
                layer_name=self.name,
                latency_ms=latency_ms,
                signals_used=signals
            )
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,  # Low temperature for consistent classification
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to Ollama at {self.base_url}. Is it running?")
    
    def _parse_response(self, response: str) -> tuple:
        """Parse JSON response from Gemma"""
        try:
            # Extract JSON from response
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            data = json.loads(json_str)
            
            complexity = TaskComplexity[data["complexity"].upper()]
            confidence = float(data.get("confidence", 0.7))
            reason = data.get("reason", "Gemma classification")
            
            return complexity, min(confidence, 1.0), reason
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Failed to parse Gemma response: {response[:200]}") from e
```

#### 2.3 - Caching Layer (Optional but Recommended)

```python
# classifier/cache/redis_cache.py

import redis
import json
import hashlib
from typing import Optional
from classifier.types import LayerResult, TaskComplexity

class ClassificationCache:
    """Cache classification results to avoid redundant inference"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl_seconds: int = 86400):
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl_seconds
    
    def _hash_request(self, request: str) -> str:
        """Create cache key from request hash"""
        return f"clf:{hashlib.md5(request.encode()).hexdigest()}"
    
    def get(self, request: str) -> Optional[LayerResult]:
        """Get cached classification result"""
        key = self._hash_request(request)
        data = self.redis.get(key)
        if data:
            return self._deserialize(data)
        return None
    
    def set(self, request: str, result: LayerResult):
        """Cache a classification result"""
        key = self._hash_request(request)
        self.redis.setex(key, self.ttl, self._serialize(result))
    
    def _serialize(self, result: LayerResult) -> str:
        return json.dumps({
            "complexity": result.complexity.value,
            "confidence": result.confidence,
            "reason": result.reason,
            "layer_name": result.layer_name
        })
    
    def _deserialize(self, data: bytes) -> LayerResult:
        parsed = json.loads(data)
        # Reconstruct as basic LayerResult (without full signals)
        return LayerResult(
            complexity=TaskComplexity[parsed["complexity"].upper()],
            confidence=parsed["confidence"],
            reason=parsed["reason"],
            layer_name=parsed["layer_name"],
            latency_ms=0,  # Cached, no latency
            signals_used=None
        )
```

#### 2.4 - Testing Layer 2

```python
# tests/integration/test_layer2_gemma.py

import pytest
from classifier.layers.layer2_onprem import GemmaClassifier
from classifier.types import TaskComplexity

@pytest.mark.integration  # Requires Gemma running locally
class TestGemmaClassifier:
    
    @pytest.fixture
    def classifier(self):
        return GemmaClassifier(model_name="gemma2:7b")
    
    def test_gemma_simple_classification(self, classifier):
        """Test Gemma can classify simple task"""
        request = "What is Python?"
        signals = None  # Gemma doesn't need pre-extracted signals
        
        result = classifier.classify(signals, request)
        
        assert result.complexity in [TaskComplexity.SIMPLE, TaskComplexity.STANDARD]
        assert result.latency_ms < 500  # Should be reasonably fast
        assert result.confidence > 0.5
    
    def test_gemma_complex_classification(self, classifier):
        """Test Gemma on complex reasoning"""
        request = """
        Design a distributed system for real-time data processing with:
        1. Fault tolerance
        2. Scalability to 1M events/sec
        3. Sub-100ms latency
        4. Cost optimization
        """
        result = classifier.classify(None, request)
        
        assert result.complexity in [TaskComplexity.COMPLEX, TaskComplexity.RESEARCH]
```

---

### Phase 3: Layer 3 - API Classifier (Claude Haiku / GPT-4o-Mini) (Days 11-14)
**Goal**: Accurate classification via lightweight API call  
**Owner**: ML Engineer + Backend  
**Deliverables**: Claude Haiku classifier working, cost < $0.0002 per request  
**Target Performance**: 300-800ms per request  

#### 3.1 - API Classifier Implementation

```python
# classifier/layers/layer3_api.py

import anthropic
import openai
import json
import time
from typing import Optional, Literal
from classifier.types import TaskComplexity, LayerResult, ClassificationSignals

class APIClassifier:
    """Lightweight API-based classifier using Claude Haiku or GPT-4o-Mini"""
    
    def __init__(
        self,
        provider: Literal["anthropic", "openai"] = "anthropic",
        model: Optional[str] = None
    ):
        self.provider = provider
        
        if provider == "anthropic":
            self.client = anthropic.Anthropic()
            self.model = model or "claude-3-5-haiku-20241022"
        else:  # openai
            self.client = openai.OpenAI()
            self.model = model or "gpt-4o-mini"
        
        self.name = f"layer3_api_{provider}"
    
    def classify(self, signals: ClassificationSignals, request: str) -> LayerResult:
        """Classify using lightweight API"""
        start_time = time.time()
        
        try:
            # Call API
            if self.provider == "anthropic":
                response = self._call_claude(request)
            else:
                response = self._call_openai(request)
            
            # Parse response
            complexity, confidence, reason = self._parse_response(response)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LayerResult(
                complexity=complexity,
                confidence=confidence,
                reason=reason,
                layer_name=self.name,
                latency_ms=latency_ms,
                signals_used=signals
            )
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return LayerResult(
                complexity=TaskComplexity.STANDARD,
                confidence=0.0,
                reason=f"API error (fallback): {str(e)}",
                layer_name=self.name,
                latency_ms=latency_ms,
                signals_used=signals
            )
    
    def _call_claude(self, request: str) -> str:
        """Call Claude Haiku for classification"""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": self._build_prompt(request)
                }
            ]
        )
        return message.content[0].text
    
    def _call_openai(self, request: str) -> str:
        """Call GPT-4o-Mini for classification"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": self._build_prompt(request)
                }
            ],
            max_tokens=200,
            temperature=0.3
        )
        return response.choices[0].message.content
    
    def _build_prompt(self, request: str) -> str:
        """Build classification prompt"""
        return f"""Classify this task complexity. Respond ONLY with JSON (no markdown):
{{
  "complexity": "simple|standard|complex|research",
  "confidence": 0.0-1.0,
  "reason": "brief explanation"
}}

SIMPLE: Basic question, definition, lookup
STANDARD: Moderate reasoning, 500-5k tokens
COMPLEX: Deep reasoning, multi-step, 5k-15k tokens
RESEARCH: Comprehensive analysis, >15k tokens

Task:
{request[:2000]}"""
    
    def _parse_response(self, response: str) -> tuple:
        """Parse API response"""
        try:
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            data = json.loads(json_str)
            
            complexity = TaskComplexity[data["complexity"].upper()]
            confidence = float(data.get("confidence", 0.8))
            reason = data.get("reason", "API classification")
            
            return complexity, min(confidence, 1.0), reason
        
        except Exception as e:
            raise ValueError(f"Failed to parse API response: {response[:200]}") from e
```

#### 3.2 - Cost Tracking

```python
# classifier/logging/cost_tracker.py

from dataclasses import dataclass
from typing import Dict
import json

@dataclass
class CostMetrics:
    """Track cost of classification"""
    layer1_requests: int = 0
    layer2_requests: int = 0
    layer3_requests: int = 0  # These cost money
    layer4_requests: int = 0
    
    # Estimated costs
    layer3_cost_per_request: float = 0.0002  # Claude Haiku input/output
    
    def total_cost_estimate(self) -> float:
        """Estimate total classification cost"""
        return self.layer3_requests * self.layer3_cost_per_request
    
    def to_dict(self) -> Dict:
        return {
            "layer1_requests": self.layer1_requests,
            "layer2_requests": self.layer2_requests,
            "layer3_requests": self.layer3_requests,
            "layer4_requests": self.layer4_requests,
            "estimated_cost": self.total_cost_estimate()
        }
```

---

### Phase 4: Layer 4 - ML Ensemble (Days 15-18)
**Goal**: Async ML model that improves routing over time  
**Owner**: Data Science  
**Deliverables**: sklearn ensemble trained, online learning working  
**Target**: Async, triggers after Layer 3  

#### 4.1 - ML Ensemble Classifier

```python
# classifier/layers/layer4_ml_ensemble.py

import pickle
import numpy as np
from typing import Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import time

from classifier.types import TaskComplexity, LayerResult, ClassificationSignals

class MLEnsembleClassifier:
    """Scikit-learn ensemble for learning from real data"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.name = "layer4_ml_ensemble"
        self.complexity_to_label = {
            TaskComplexity.SIMPLE: 0,
            TaskComplexity.STANDARD: 1,
            TaskComplexity.COMPLEX: 2,
            TaskComplexity.RESEARCH: 3,
        }
        self.label_to_complexity = {v: k for k, v in self.complexity_to_label.items()}
        
        if model_path:
            self.model = pickle.load(open(model_path, "rb"))
        else:
            self.model = self._build_model()
    
    def _build_model(self) -> Pipeline:
        """Build ensemble pipeline"""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ensemble", RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42
            ))
        ])
    
    def classify(self, signals: ClassificationSignals) -> LayerResult:
        """Classify using trained ML model"""
        start_time = time.time()
        
        try:
            # Extract features
            features = self._extract_features(signals)
            
            # Predict
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            
            complexity = self.label_to_complexity[prediction]
            confidence = float(np.max(probabilities))
            reason = f"ML ensemble prediction (confidence: {confidence:.2f})"
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LayerResult(
                complexity=complexity,
                confidence=confidence,
                reason=reason,
                layer_name=self.name,
                latency_ms=latency_ms,
                signals_used=signals
            )
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return LayerResult(
                complexity=TaskComplexity.STANDARD,
                confidence=0.0,
                reason=f"ML error: {str(e)}",
                layer_name=self.name,
                latency_ms=latency_ms,
                signals_used=signals
            )
    
    def _extract_features(self, signals: ClassificationSignals) -> np.ndarray:
        """Convert signals to ML feature vector"""
        return np.array([
            signals.token_count,
            signals.keyword_scores.get("reasoning", 0),
            signals.keyword_scores.get("planning", 0),
            signals.keyword_scores.get("research", 0),
            signals.keyword_scores.get("coding", 0),
            signals.keyword_scores.get("simple", 0),
            signals.data_volume_mb,
            int(signals.has_tools),
            int(signals.has_planning),
            int(signals.has_reasoning),
            signals.nesting_depth,
            len(signals.domain_tags),
        ])
    
    def train(self, X_train, y_train):
        """Train on real data"""
        self.model.fit(X_train, y_train)
    
    def save(self, path: str):
        """Persist model"""
        pickle.dump(self.model, open(path, "wb"))
```

#### 4.2 - Online Learning

```python
# classifier/layers/layer4_ml_ensemble.py (continued)

class OnlineLearningManager:
    """Continuously improve ML model from real predictions"""
    
    def __init__(self, classifier: MLEnsembleClassifier):
        self.classifier = classifier
        self.training_buffer = []
        self.buffer_size = 100  # Retrain after 100 correct predictions
    
    def record_prediction(self, signals: ClassificationSignals, actual_complexity: TaskComplexity):
        """Record actual complexity for online learning"""
        features = self.classifier._extract_features(signals)
        label = self.classifier.complexity_to_label[actual_complexity]
        
        self.training_buffer.append((features, label))
        
        # Retrain when buffer is full
        if len(self.training_buffer) >= self.buffer_size:
            self.retrain()
    
    def retrain(self):
        """Retrain model on buffered data"""
        if not self.training_buffer:
            return
        
        X = np.array([x[0] for x in self.training_buffer])
        y = np.array([x[1] for x in self.training_buffer])
        
        # Incremental learning
        self.classifier.train(X, y)
        self.training_buffer = []  # Clear buffer
```

---

### Phase 5: Core Pipeline & Integration (Days 19-21)
**Goal**: Tie all layers together into a working pipeline  
**Owner**: Backend Engineer  
**Deliverables**: ClassificationPipeline class working, decision logging  

#### 5.1 - Main Pipeline Class

```python
# classifier/core.py

import time
from typing import Optional, Dict, List
from classifier.features.extractor import FeatureExtractor
from classifier.layers.layer1_heuristics import HeuristicClassifier
from classifier.layers.layer2_onprem import GemmaClassifier
from classifier.layers.layer3_api import APIClassifier
from classifier.layers.layer4_ml_ensemble import MLEnsembleClassifier, OnlineLearningManager
from classifier.types import ClassificationDecision, TaskComplexity, LayerResult
from classifier.logging.structured_logger import StructuredLogger
from classifier.models.registry import ModelRegistry

class ClassificationPipeline:
    """Main orchestrator for multi-layer classification"""
    
    def __init__(
        self,
        enable_layer2: bool = True,
        enable_layer3: bool = True,
        enable_layer4: bool = False,
        model_provider: str = "anthropic",
        logger: Optional[StructuredLogger] = None
    ):
        self.feature_extractor = FeatureExtractor()
        self.layer1 = HeuristicClassifier()
        self.layer2 = GemmaClassifier() if enable_layer2 else None
        self.layer3 = APIClassifier(provider=model_provider) if enable_layer3 else None
        self.layer4 = MLEnsembleClassifier() if enable_layer4 else None
        self.logger = logger or StructuredLogger()
        self.model_registry = ModelRegistry()
        
        # Online learning (optional)
        if self.layer4:
            self.online_learning = OnlineLearningManager(self.layer4)
        
        # Configuration thresholds
        self.layer1_confidence_threshold = 0.95
        self.layer2_confidence_threshold = 0.90
        self.layer3_confidence_threshold = 0.85
    
    def classify(
        self,
        request: str,
        context: Optional[Dict] = None,
        model_provider: str = "anthropic"
    ) -> ClassificationDecision:
        """Main classification entry point"""
        
        start_time = time.time()
        all_results: List[LayerResult] = []
        
        # Step 1: Extract features
        signals = self.feature_extractor.extract(request, context)
        
        # Step 2: Layer 1 - Heuristics
        layer1_result = self.layer1.classify(signals)
        all_results.append(layer1_result)
        
        if layer1_result.confidence > self.layer1_confidence_threshold:
            return self._make_decision(
                layer1_result, all_results, request, model_provider, time.time() - start_time
            )
        
        # Step 3: Layer 2 - Gemma (if enabled)
        if self.layer2:
            layer2_result = self.layer2.classify(signals, request)
            all_results.append(layer2_result)
            
            if layer2_result.confidence > self.layer2_confidence_threshold:
                return self._make_decision(
                    layer2_result, all_results, request, model_provider, time.time() - start_time
                )
        
        # Step 4: Layer 3 - API Classifier (if enabled)
        if self.layer3:
            layer3_result = self.layer3.classify(signals, request)
            all_results.append(layer3_result)
            
            if layer3_result.confidence > self.layer3_confidence_threshold:
                return self._make_decision(
                    layer3_result, all_results, request, model_provider, time.time() - start_time
                )
        
        # Step 5: Layer 4 - ML Ensemble (optional, can be async)
        if self.layer4:
            layer4_result = self.layer4.classify(signals)
            all_results.append(layer4_result)
            final_result = layer4_result
        else:
            final_result = all_results[-1]  # Use last result
        
        return self._make_decision(
            final_result, all_results, request, model_provider, time.time() - start_time
        )
    
    def _make_decision(
        self,
        chosen_result: LayerResult,
        all_results: List[LayerResult],
        request: str,
        model_provider: str,
        total_latency: float
    ) -> ClassificationDecision:
        """Convert classification result to routing decision"""
        
        # Get model name from registry
        model_name = self.model_registry.get_model(
            chosen_result.complexity, model_provider
        )
        
        # Estimate cost
        estimated_cost = self._estimate_cost(chosen_result)
        
        decision = ClassificationDecision(
            model_name=model_name,
            complexity=chosen_result.complexity,
            tier=self._complexity_to_tier(chosen_result.complexity),
            reasoning=chosen_result.reason,
            confidence=chosen_result.confidence,
            layer_used=chosen_result.layer_name,
            all_results=all_results,
            estimated_cost=estimated_cost,
            signals=chosen_result.signals_used
        )
        
        # Log decision
        self.logger.log_classification(
            request=request[:500],
            decision=decision,
            latency_ms=total_latency * 1000,
            all_layers=[r.layer_name for r in all_results]
        )
        
        return decision
    
    def _complexity_to_tier(self, complexity: TaskComplexity) -> str:
        """Map complexity to cost tier"""
        tier_map = {
            TaskComplexity.SIMPLE: "light",
            TaskComplexity.STANDARD: "standard",
            TaskComplexity.COMPLEX: "heavy",
            TaskComplexity.RESEARCH: "research"
        }
        return tier_map[complexity]
    
    def _estimate_cost(self, result: LayerResult) -> float:
        """Estimate API cost for this classification"""
        if "layer3" in result.layer_name:
            return 0.0002  # Claude Haiku approximate cost
        return 0.0  # Free layers have no cost
```

#### 5.2 - Model Registry

```python
# classifier/models/registry.py

from classifier.types import TaskComplexity

class ModelRegistry:
    """Central registry of available models"""
    
    def __init__(self):
        self.models = {
            "anthropic": {
                TaskComplexity.SIMPLE: "claude-3-5-haiku-20241022",
                TaskComplexity.STANDARD: "claude-3-5-sonnet-20241022",
                TaskComplexity.COMPLEX: "claude-3-opus-20250219",
                TaskComplexity.RESEARCH: "claude-3-opus-20250219",
            },
            "openai": {
                TaskComplexity.SIMPLE: "gpt-4o-mini",
                TaskComplexity.STANDARD: "gpt-4o",
                TaskComplexity.COMPLEX: "gpt-4-turbo",
                TaskComplexity.RESEARCH: "gpt-4-turbo",
            },
            "google": {
                TaskComplexity.SIMPLE: "gemini-1.5-flash",
                TaskComplexity.STANDARD: "gemini-2.0-flash",
                TaskComplexity.COMPLEX: "gemini-2.0-pro",
                TaskComplexity.RESEARCH: "gemini-2.0-pro",
            }
        }
    
    def get_model(self, complexity: TaskComplexity, provider: str = "anthropic") -> str:
        """Get recommended model for complexity level"""
        return self.models[provider][complexity]
    
    def add_model(self, provider: str, complexity: TaskComplexity, model_name: str):
        """Add/override a model"""
        if provider not in self.models:
            self.models[provider] = {}
        self.models[provider][complexity] = model_name
```

#### 5.3 - Structured Logging

```python
# classifier/logging/structured_logger.py

import json
import logging
from datetime import datetime
from typing import List, Dict

class StructuredLogger:
    """Structured logging for classification decisions"""
    
    def __init__(self, log_file: str = "classification.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("classifier")
        
        # File handler
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_classification(
        self,
        request: str,
        decision,
        latency_ms: float,
        all_layers: List[str]
    ):
        """Log classification decision in structured format"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_preview": request[:100],
            "complexity": decision.complexity.value,
            "model": decision.model_name,
            "tier": decision.tier,
            "confidence": decision.confidence,
            "layer_used": decision.layer_used,
            "all_layers_tried": all_layers,
            "latency_ms": round(latency_ms, 2),
            "estimated_cost": round(decision.estimated_cost, 6),
            "reasoning": decision.reasoning[:200]
        }
        
        self.logger.info(json.dumps(log_entry))
```

---

### Phase 6: Adapter Integration (Days 22-24)
**Goal**: Integrate with CrewAI and Google ADK  
**Owner**: Integration Engineer  

#### 6.1 - CrewAI Adapter

```python
# classifier/adapters/crewai_adapter.py

from crewai import Agent
from classifier.core import ClassificationPipeline
from classifier.types import ClassificationDecision

class CrewAIAdapter:
    """Integrate classification pipeline with CrewAI"""
    
    def __init__(self, pipeline: ClassificationPipeline):
        self.pipeline = pipeline
    
    def create_agent_with_smart_model(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: list,
        request: str = None
    ) -> Agent:
        """
        Create a CrewAI agent with intelligently selected model.
        
        Usage:
            adapter = CrewAIAdapter(pipeline)
            agent = adapter.create_agent_with_smart_model(
                role="researcher",
                goal="...",
                backstory="...",
                tools=[...],
                request=user_request  # Used to classify and select model
            )
        """
        
        # Classify the request
        if request:
            decision = self.pipeline.classify(request, model_provider="openai")
        else:
            # Default to standard
            decision = ClassificationDecision(
                model_name="gpt-4o",
                complexity="standard",
                tier="standard",
                reasoning="Default model",
                confidence=0.5,
                layer_used="default",
                all_results=[],
                estimated_cost=0,
                signals=None
            )
        
        # Create agent with selected model
        agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools,
            model=decision.model_name,  # Dynamically set model
            verbose=True
        )
        
        return agent, decision
```

#### 6.2 - Google ADK Adapter

```python
# classifier/adapters/adk_adapter.py

from google.generativeai import adk
from classifier.core import ClassificationPipeline

class ADKAdapter:
    """Integrate classification pipeline with Google ADK"""
    
    def __init__(self, pipeline: ClassificationPipeline):
        self.pipeline = pipeline
    
    def create_agent_with_smart_model(
        self,
        name: str,
        task_description: str,
        request: str = None
    ) -> tuple:
        """
        Create a Google ADK agent with intelligently selected model.
        
        Returns: (agent, decision)
        """
        
        # Classify
        if request:
            decision = self.pipeline.classify(request, model_provider="google")
        else:
            decision = self._default_decision()
        
        # Create agent with selected model
        agent = adk.LlmAgent(
            name=name,
            model=decision.model_name,
            task_description=task_description
        )
        
        return agent, decision
    
    def _default_decision(self):
        from classifier.types import TaskComplexity, ClassificationDecision
        return ClassificationDecision(
            model_name="gemini-2.0-flash",
            complexity=TaskComplexity.STANDARD,
            tier="standard",
            reasoning="Default model",
            confidence=0.5,
            layer_used="default",
            all_results=[],
            estimated_cost=0,
            signals=None
        )
```

---

### Phase 7: Testing & Validation (Days 25-28)
**Goal**: Comprehensive testing of all layers and integration  
**Owner**: QA Engineer  

#### 7.1 - Test Matrix

```
Test Category           | Coverage      | Latency SLA
─────────────────────────────────────────────────────
Unit Tests (Layers)     | 100%          | <50ms each
Integration (Layers)    | 95%+          | <1s per layer
End-to-End (Pipeline)   | 90%+          | <2s total
Performance             | TBD           | See SLAs below
Cost                    | TBD           | <$0.001 per request
```

#### 7.2 - Performance SLA

```python
# tests/performance/test_sla.py

import pytest
from classifier.core import ClassificationPipeline

class TestPerformanceSLA:
    """Validate performance against SLAs"""
    
    @pytest.fixture
    def pipeline(self):
        return ClassificationPipeline(
            enable_layer2=True,
            enable_layer3=True,
            enable_layer4=False
        )
    
    def test_layer1_under_10ms(self, pipeline):
        """Layer 1 must complete in < 10ms"""
        import time
        request = "What is 2+2?"
        
        start = time.time()
        decision = pipeline.classify(request)
        latency = (time.time() - start) * 1000
        
        assert decision.layer_used == "layer1_heuristics"
        assert latency < 10, f"Layer 1 took {latency}ms, SLA: <10ms"
    
    def test_layer2_under_200ms(self, pipeline):
        """Layer 2 must complete in < 200ms"""
        # Complex request forces Layer 2
        import time
        request = "Analyze and compare X with Y" * 20
        
        start = time.time()
        decision = pipeline.classify(request)
        latency = (time.time() - start) * 1000
        
        assert "layer2" in decision.layer_used or "layer1" in decision.layer_used
        assert latency < 200, f"Took {latency}ms, SLA: <200ms"
    
    def test_full_pipeline_under_2s(self, pipeline):
        """Full pipeline under 2 seconds"""
        import time
        request = "Complex reasoning task" * 50
        
        start = time.time()
        for i in range(5):  # 5 requests
            pipeline.classify(request)
        total_latency = (time.time() - start) * 1000 / 5
        
        assert total_latency < 2000, f"Average {total_latency}ms, SLA: <2s"
```

---

### Phase 8: Deployment & Monitoring (Days 29-30)
**Goal**: Production-ready deployment and monitoring  
**Owner**: DevOps  

#### 8.1 - Docker Compose for Full Stack

```yaml
# docker-compose.yml

version: '3.8'

services:
  # Ollama (Gemma-2)
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/health"]
      interval: 10s
      timeout: 5s
      retries: 5
  
  # Redis (Caching)
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
  
  # Classifier API
  classifier:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OLLAMA_URL=http://ollama:11434
      - REDIS_URL=redis://redis:6379
    depends_on:
      ollama:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs

volumes:
  ollama_data:
```

#### 8.2 - FastAPI Server

```python
# classifier/api/server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from classifier.core import ClassificationPipeline
from classifier.types import ClassificationDecision

app = FastAPI(title="Task Classifier API", version="1.0")
pipeline = ClassificationPipeline()

class ClassifyRequest(BaseModel):
    request: str
    context: Optional[dict] = None
    provider: str = "anthropic"

class ClassifyResponse(BaseModel):
    model: str
    complexity: str
    tier: str
    confidence: float
    layer_used: str
    estimated_cost: float
    reasoning: str

@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    """Classify a task and return routing decision"""
    try:
        decision = pipeline.classify(
            req.request,
            context=req.context,
            model_provider=req.provider
        )
        
        return ClassifyResponse(
            model=decision.model_name,
            complexity=decision.complexity.value,
            tier=decision.tier,
            confidence=decision.confidence,
            layer_used=decision.layer_used,
            estimated_cost=decision.estimated_cost,
            reasoning=decision.reasoning
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "layers_enabled": {
            "layer1": True,
            "layer2": pipeline.layer2 is not None,
            "layer3": pipeline.layer3 is not None,
            "layer4": pipeline.layer4 is not None,
        }
    }
```

#### 8.3 - Monitoring Dashboard (Prometheus metrics)

```python
# classifier/monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge

# Counters
classification_total = Counter(
    'classification_total',
    'Total classifications',
    ['complexity', 'layer', 'provider']
)

classifications_by_model = Counter(
    'classifications_by_model',
    'Classifications per model',
    ['model']
)

# Histograms
classification_latency = Histogram(
    'classification_latency_ms',
    'Classification latency in milliseconds',
    ['layer']
)

api_cost = Counter(
    'classifier_api_cost_dollars',
    'Total API cost in dollars'
)

# Gauges
accuracy_score = Gauge(
    'classification_accuracy',
    'Accuracy of classifications (0-1)',
    ['layer']
)
```

---

## Integration Points

### How Each Layer Feeds Into The Next

```
LAYER 1 (Heuristics)
  → confidence > 0.95? → RETURN
  → ELSE → LAYER 2

LAYER 2 (Gemma Local)
  → Cached? → RETURN cached
  → Latency > 200ms? → Skip to Layer 3
  → confidence > 0.90? → RETURN + CACHE
  → ELSE → LAYER 3

LAYER 3 (Claude Haiku / GPT-4o-Mini)
  → confidence > 0.85? → RETURN
  → ELSE → LAYER 4 (async, background)

LAYER 4 (ML Ensemble - Async)
  → Trigger after Layer 3 returns
  → Learn from actual classification accuracy
  → Update model weights for next prediction
```

### Decision Points & Fallbacks

```
If Layer 2 (Gemma) is unavailable:
  → Skip directly to Layer 3
  
If Layer 3 (API) fails:
  → Use Layer 2 result or Layer 1 result
  → Default to STANDARD tier
  
If ALL layers fail:
  → Return STANDARD/middle-ground model
  → Log error for investigation
```

---

## Testing Strategy

### Test Pyramid

```
         /\
        /  \          E2E & Integration (10% - 2-3 tests)
       /────\         "Full pipeline with real creweAI/ADK"
      /      \
     /        \       Integration (30% - 10-15 tests)
    /──────────\      "Layers together, real models/APIs"
   /            \
  /              \    Unit (60% - 30-40 tests)
 /────────────────\   "Individual components, mocked"
```

### Test Categories

1. **Unit Tests** (60%)
   - Layer 1 heuristics (keyword, token, nesting)
   - Feature extraction
   - Model registry
   - Type validations

2. **Integration Tests** (30%)
   - Layer 1 + Layer 2 (heuristics + Gemma)
   - Layer 2 + Layer 3 (Gemma + API)
   - Cache integration
   - Logging integration

3. **End-to-End Tests** (10%)
   - Full pipeline A-Z
   - CrewAI adapter
   - ADK adapter
   - Real model API calls (expensive, run nightly)

4. **Performance Tests** (Continuous)
   - Latency per layer
   - Accuracy metrics
   - Cost tracking

5. **Mutation Tests** (Optional)
   - Verify heuristic rules are correct
   - Verify keyword lists are effective

---

## Deployment & Operations

### Deployment Phases

```
Dev Environment
  ↓ (Week 1-2)
Staging Environment (limited traffic)
  ↓ (Week 3)
Production (10% traffic → 50% → 100%)
  ↓ (Week 4)
Monitor & Optimize
```

### Cost Optimization

```
API Calls Cost:
  Layer 3 (Claude Haiku): $0.80 per million input tokens
  
Example traffic:
  1,000 requests/day
  400 reach Layer 3 (others stop at Layer 1/2)
  → 400 * $0.0002 = $0.08/day = $2.40/month
  
ROI if saves 40% on model costs:
  Avg request costs: $0.001 (gpt-4) → $0.0002 (gpt-4o-mini when appropriate)
  1,000 req/day * $0.001 = $30/day without optimization
  With 40% savings: $18/day saved
  Classifier cost: $0.08/day
  Net savings: $17.92/day = $537/month
```

---

## Configuration Management

### Layer Configuration (layer_config.yaml)

```yaml
layers:
  layer1:
    enabled: true
    confidence_threshold: 0.95
    timeout_ms: 10
  
  layer2:
    enabled: true
    model: "gemma2:7b"
    confidence_threshold: 0.90
    timeout_ms: 200
    cache_ttl_seconds: 86400
  
  layer3:
    enabled: true
    provider: "anthropic"  # or "openai"
    model: "claude-3-5-haiku-20241022"
    confidence_threshold: 0.85
    timeout_ms: 800
  
  layer4:
    enabled: false  # Start with false
    model: "random_forest"
    timeout_ms: 500

model_mapping:
  anthropic:
    simple: "claude-3-5-haiku-20241022"
    standard: "claude-3-5-sonnet-20241022"
    complex: "claude-3-opus-20250219"
    research: "claude-3-opus-20250219"
  
  openai:
    simple: "gpt-4o-mini"
    standard: "gpt-4o"
    complex: "gpt-4-turbo"
    research: "gpt-4-turbo"
```

---

## Key Metrics & Monitoring

```
Metrics to Track:

1. Classification Accuracy
   - % of correct classifications (validate against actual agent performance)
   - Per-layer accuracy
   - By task complexity

2. Latency
   - P50, P95, P99 latencies per layer
   - Full pipeline latency distribution

3. Cost
   - Daily API cost
   - Cost per classification
   - Cost by provider

4. Layer Distribution
   - % stopping at Layer 1 (should be 60-70%)
   - % reaching Layer 2 (should be 20-30%)
   - % reaching Layer 3 (should be 5-10%)
   - % reaching Layer 4 (should be rare)

5. Model Usage
   - Which models chosen per complexity
   - Model cost distribution

6. System Health
   - Layer availability (especially Layer 2/3)
   - Fallback triggers
   - Errors per layer
```

---

## Decision Points & Trade-offs

### Why This Multi-Layer Approach?

```
Single Layer Trade-offs:
  Layer 1 Only (Heuristics)
    ✓ Fastest
    ✗ Least accurate (80% accuracy typical)
  
  Layer 2 Only (Gemma)
    ✓ Good balance
    ✗ Requires local compute
    ✗ Gemma might not be available everywhere
  
  Layer 3 Only (API)
    ✓ Most accurate
    ✗ Slowest
    ✗ Every call costs money
  
  Multi-Layer
    ✓ Fast (Layer 1 catches 70% of requests in <10ms)
    ✓ Accurate (Layer 3 handles edge cases)
    ✓ Cheap (Layer 3 only on 5-10% of requests)
    ✓ Reliable (fallbacks if a layer is unavailable)
```

### When to Enable Each Layer

```
Development:
  Layer 1: ✓ Yes (always)
  Layer 2: ✗ No (optional, slow to install)
  Layer 3: ✓ Yes (free quota)
  Layer 4: ✗ No (no training data yet)

Staging:
  Layer 1: ✓ Yes
  Layer 2: ✓ Yes (validate)
  Layer 3: ✓ Yes
  Layer 4: ✗ No

Production:
  Layer 1: ✓ Yes
  Layer 2: ✓ Yes (performance)
  Layer 3: ✓ Yes (accuracy)
  Layer 4: ✓ Yes (v2, with training data)
```

---

## Future Enhancements (Phase 2+)

1. **Fine-tuning** (Month 2)
   - Fine-tune Gemma-2 on your specific tasks
   - Fine-tune Claude Haiku for your domain

2. **Multi-Model Ensemble** (Month 3)
   - Ensemble Layer 3 across Claude + GPT-4o-Mini for consensus
   - Voting mechanism for high-confidence decisions

3. **Cost-Aware Routing** (Month 3)
   - Budget-constrained mode: minimize API spend
   - Quality-constrained mode: maximize accuracy

4. **Adaptive Learning** (Month 4)
   - Track which model performs best on actual tasks
   - Dynamically adjust model selection

5. **Distributed Caching** (Month 4)
   - Redis/Memcached for distributed cache
   - Share classifications across multiple servers

6. **Custom Models** (Month 5)
   - Train proprietary classifier on your data
   - Replace generic Layer 4 with domain-specific model

---

## Success Criteria

✅ **Implementation Complete When:**

1. All 4 layers integrated and tested
2. < 10ms median latency for 70% of requests
3. < 2s P95 latency for 100% of requests
4. 90%+ classification accuracy (validated)
5. < $0.0003 avg cost per classification
6. Full integration with CrewAI + Google ADK
7. Comprehensive monitoring & alerting
8. Documentation complete
9. Team trained on framework

---

## References & Resources

- **Heuristics**: Feature extraction techniques
- **Gemma-2**: https://ai.google.dev/gemma/
- **Claude API**: https://anthropic.com/api
- **OpenAI API**: https://openai.com/api
- **CrewAI**: https://github.com/crewaiinc/crewai
- **Google ADK**: https://github.com/google/adk-python
- **scikit-learn**: https://scikit-learn.org/
- **Ollama**: https://ollama.ai

---

## Appendix: Quick Start

### Install & Run (Development)

```bash
# 1. Clone repo
git clone <your-repo>
cd model-classifier-framework

# 2. Install deps
pip install -r requirements.txt

# 3. Set env vars
export ANTHROPIC_API_KEY=sk-...
export OPENAI_API_KEY=sk-...

# 4. Run tests
pytest tests/unit -v

# 5. Try CLI
python -c "
from classifier.core import ClassificationPipeline
pipeline = ClassificationPipeline()
decision = pipeline.classify('What is machine learning?')
print(decision)
"
```

### Integration with CrewAI

```python
from classifier.core import ClassificationPipeline
from classifier.adapters.crewai_adapter import CrewAIAdapter

pipeline = ClassificationPipeline()
adapter = CrewAIAdapter(pipeline)

# Create agent with auto-selected model
agent, decision = adapter.create_agent_with_smart_model(
    role="researcher",
    goal="Research AI topics",
    backstory="Expert researcher",
    tools=[search_tool, read_tool],
    request="Research the latest AI models 2024-2026"
)

print(f"Selected model: {decision.model_name}")
print(f"Complexity: {decision.complexity.value}")
print(f"Reasoning: {decision.reasoning}")

# Use agent normally
task = Task(
    description="...",
    agent=agent,
    expected_output="..."
)
```

---

**Document Version**: 1.0  
**Last Updated**: 2026-04-24  
**Status**: READY FOR IMPLEMENTATION  
**Next Step**: Begin Phase 0 - Foundation Setup
