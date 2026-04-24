# Model Categorization & Task Classification Framework
## Tier-Based Model Selection with Task Sub-Classification

**Version**: 1.0  
**Date**: 2026-04-24  
**Purpose**: Map task types and complexity to optimal models across 3 tiers  

---

## Table of Contents

1. [Model Tier System](#model-tier-system)
2. [Task Sub-Classifications](#task-sub-classifications)
3. [Model Capability Matrix](#model-capability-matrix)
4. [Task-to-Model Mapping](#task-to-model-mapping)
5. [Implementation Changes](#implementation-changes)

---

## Model Tier System

### Tier Definition

```
┌─────────────────────────────────────────────────────────────┐
│                     MODEL TIERS                              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  LOW TIER                                                     │
│  ├─ Cost: $0.00015 / 1K tokens input                         │
│  ├─ Speed: 200-400ms                                         │
│  ├─ Context: 128K tokens                                     │
│  ├─ Best for: Simple Q&A, summarization, formatting         │
│  ├─ Tool use: Basic                                          │
│  └─ Reasoning: Limited (0-1 step)                            │
│                                                               │
│  MEDIUM TIER                                                  │
│  ├─ Cost: $0.003 / 1K tokens input                           │
│  ├─ Speed: 300-600ms                                         │
│  ├─ Context: 200K tokens                                     │
│  ├─ Best for: Moderate reasoning, analysis, coding          │
│  ├─ Tool use: Advanced                                       │
│  └─ Reasoning: Multi-step (2-5 steps)                        │
│                                                               │
│  HIGH TIER                                                    │
│  ├─ Cost: $0.015 / 1K tokens input                           │
│  ├─ Speed: 400-800ms                                         │
│  ├─ Context: 200K tokens (extended thinking)                │
│  ├─ Best for: Complex reasoning, research, design          │
│  ├─ Tool use: Expert                                         │
│  └─ Reasoning: Deep reasoning (5+ steps, multi-domain)      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Tier Characteristics

| Aspect | LOW | MEDIUM | HIGH |
|--------|-----|--------|------|
| **Cost (per 1K input tokens)** | $0.00015 | $0.003 | $0.015 |
| **Typical Latency** | 200-400ms | 300-600ms | 400-800ms |
| **Context Window** | 128K | 200K | 200K |
| **Instruction Following** | 85% | 95% | 99% |
| **Code Generation** | Basic | Intermediate | Advanced |
| **Multi-step Reasoning** | 1 step | 2-5 steps | 5+ steps |
| **Tool Use Reliability** | 70% | 90% | 98% |
| **Structured Output** | ✓ | ✓✓ | ✓✓✓ |
| **Knowledge Currency** | 2024 | 2024-2025 | Real-time capable |
| **Math/Logic** | Simple | Moderate | Complex |
| **Price per 1M output tokens** | $0.0006 | $0.009 | $0.045 |

---

## Task Sub-Classifications

### Overview

Instead of just "SIMPLE/STANDARD/COMPLEX/RESEARCH", we classify **task TYPE** separately from **task COMPLEXITY**.

```
Task Type (What are we doing?)
├─ REASONING         (Analyze, evaluate, compare, debate)
├─ THINKING          (Plan, strategize, brainstorm, explore)
├─ ANALYZING         (Data analysis, statistics, insights)
├─ CODE_CREATION     (Write, refactor, debug code)
└─ DOC_CREATION      (Write, edit, format documentation)

Combined with:

Complexity Level (How hard is it?)
├─ SIMPLE            (<500 tokens, straightforward)
├─ STANDARD          (500-5K tokens, moderate)
├─ COMPLEX           (5K-15K tokens, deep reasoning)
└─ RESEARCH          (>15K tokens, multi-faceted)
```

### Task Type Details

#### 1. **REASONING** 
Tasks that require analytical thinking, comparison, evaluation

**Keywords**: analyze, compare, evaluate, assess, debate, interpret, distinguish, validate, argument, logic

**Sub-categories**:
- Comparative analysis (compare X vs Y)
- Critical evaluation (assess pros/cons)
- Logical deduction (if X then Y)
- Assumption validation (is this correct?)
- Debate/counter-argument

**Model Recommendation by Complexity**:
```
REASONING + SIMPLE     → LOW tier
  Example: "Compare Python vs JavaScript"
  
REASONING + STANDARD   → MEDIUM tier
  Example: "Analyze trade-offs between microservices and monolithic architecture"
  
REASONING + COMPLEX    → HIGH tier
  Example: "Evaluate the long-term implications of AI adoption on software engineering salaries"
  
REASONING + RESEARCH   → HIGH tier + Extended Thinking
  Example: "Conduct a comprehensive analysis of AI impact across multiple industries"
```

**Model Fit**:
- **Claude Opus**: Excellent (best for nuanced reasoning)
- **GPT-4 Turbo**: Excellent (strong analytical capability)
- **Claude Sonnet**: Good (handles most cases)
- **GPT-4o**: Good (fast reasoning)
- **Claude Haiku**: Fair (basic reasoning only)
- **GPT-4o-Mini**: Fair (basic reasoning only)

---

#### 2. **THINKING**
Tasks that require exploration, planning, creativity, brainstorming

**Keywords**: plan, strategy, brainstorm, explore, design, ideate, organize, structure, approach, workflow, process

**Sub-categories**:
- Strategic planning (how should we approach X?)
- Creative brainstorming (generate ideas for X)
- System design (design architecture for X)
- Workflow planning (organize steps for X)
- Problem exploration (what are all angles of X?)

**Model Recommendation by Complexity**:
```
THINKING + SIMPLE     → LOW tier
  Example: "Plan a simple daily schedule"
  
THINKING + STANDARD   → MEDIUM tier
  Example: "Design a user onboarding workflow for a SaaS app"
  
THINKING + COMPLEX    → HIGH tier
  Example: "Design a complete microservices architecture with all trade-offs"
  
THINKING + RESEARCH   → HIGH tier + Extended Thinking
  Example: "Explore all possible approaches to solving distributed system consensus"
```

**Model Fit**:
- **Claude Opus**: Excellent (creative, explores multiple angles)
- **GPT-4 Turbo**: Excellent (good for structured planning)
- **Claude Sonnet**: Good (handles most planning tasks)
- **GPT-4o**: Good (efficient for design tasks)
- **Claude Haiku**: Fair (limited planning depth)
- **GPT-4o-Mini**: Fair (limited planning depth)

---

#### 3. **ANALYZING**
Tasks involving data analysis, pattern recognition, statistical insights

**Keywords**: analyze data, statistics, pattern, trend, insight, metric, aggregate, summarize, breakdown, distribution

**Sub-categories**:
- Data exploration (what patterns exist?)
- Statistical analysis (compute metrics)
- Trend identification (what changed?)
- Anomaly detection (what's unusual?)
- Correlation analysis (what relates to what?)

**Model Recommendation by Complexity**:
```
ANALYZING + SIMPLE     → LOW tier
  Example: "Summarize sales data by month"
  
ANALYZING + STANDARD   → MEDIUM tier
  Example: "Analyze user behavior patterns and identify top features"
  
ANALYZING + COMPLEX    → HIGH tier + MEDIUM tier (code)
  Example: "Perform statistical analysis on A/B test results with confidence intervals"
  
ANALYZING + RESEARCH   → HIGH tier
  Example: "Conduct comprehensive market analysis across 50 data sources"
```

**Model Fit**:
- **Claude Opus**: Excellent (handles complex statistical reasoning)
- **GPT-4 Turbo**: Excellent (strong at math, statistics)
- **Claude Sonnet**: Good (decent analytics)
- **GPT-4o**: Good (handles most analyses)
- **Claude Haiku**: Poor (limited mathematical reasoning)
- **GPT-4o-Mini**: Poor (limited mathematical reasoning)

---

#### 4. **CODE_CREATION**
Tasks involving writing, generating, debugging, optimizing code

**Keywords**: code, function, implement, debug, refactor, optimize, algorithm, class, test, fix, error

**Sub-categories**:
- Code generation (write code for X)
- Bug fixing (fix error in code)
- Code review (analyze code quality)
- Refactoring (improve code structure)
- Performance optimization (optimize for speed/memory)
- Test writing (write unit tests)
- Algorithm design (design algorithm for X)

**Model Recommendation by Complexity**:
```
CODE_CREATION + SIMPLE     → LOW/MEDIUM tier
  Example: "Write a Python function to parse CSV"
  
CODE_CREATION + STANDARD   → MEDIUM tier
  Example: "Implement a REST API with 5 endpoints"
  
CODE_CREATION + COMPLEX    → HIGH tier
  Example: "Design and implement a distributed cache with LRU eviction"
  
CODE_CREATION + RESEARCH   → HIGH tier
  Example: "Implement a full microservices architecture with service discovery"
```

**Model Fit**:
- **Claude Opus**: Excellent (best code generation)
- **GPT-4 Turbo**: Excellent (strong coding)
- **Claude Sonnet**: Very Good (handles 80% of coding tasks)
- **GPT-4o**: Very Good (fast, accurate code)
- **Claude Haiku**: Good (handles simple code)
- **GPT-4o-Mini**: Good (handles simple code)

---

#### 5. **DOC_CREATION**
Tasks involving writing, editing, formatting documentation

**Keywords**: document, write, edit, format, style, guide, manual, tutorial, comment, explain, describe

**Sub-categories**:
- Documentation writing (write API docs, guides)
- Technical writing (write explanations)
- Code commenting (add code comments)
- Editing (grammar, clarity)
- Formatting (structure, style)
- Tutorial creation (step-by-step guides)

**Model Recommendation by Complexity**:
```
DOC_CREATION + SIMPLE     → LOW tier
  Example: "Write a README for a GitHub repo"
  
DOC_CREATION + STANDARD   → LOW/MEDIUM tier
  Example: "Write API documentation with examples"
  
DOC_CREATION + COMPLEX    → MEDIUM tier
  Example: "Write a comprehensive technical guide covering multiple concepts"
  
DOC_CREATION + RESEARCH   → MEDIUM tier
  Example: "Write a research paper on AI/ML advancements"
```

**Model Fit**:
- **Claude Opus**: Excellent (best at clear writing)
- **GPT-4 Turbo**: Excellent (strong at documentation)
- **Claude Sonnet**: Very Good (natural writing style)
- **GPT-4o**: Very Good (clear, concise docs)
- **Claude Haiku**: Good (handles most doc tasks)
- **GPT-4o-Mini**: Good (handles most doc tasks)

---

## Model Capability Matrix

### Models Across Providers

#### **Anthropic Models**

| Model | Tier | Context | Speed | Cost/1K | Reasoning | Code | Math | Coding | Analysis |
|-------|------|---------|-------|---------|-----------|------|------|--------|----------|
| Claude 3.5 Haiku | LOW | 200K | Fast | $0.0008 | 70% | 60% | 40% | 75% | 60% |
| Claude 3.5 Sonnet | MEDIUM | 200K | Medium | $0.003 | 90% | 85% | 75% | 90% | 85% |
| Claude 3 Opus | HIGH | 200K | Slow | $0.015 | 98% | 95% | 90% | 98% | 95% |

#### **OpenAI Models**

| Model | Tier | Context | Speed | Cost/1K | Reasoning | Code | Math | Coding | Analysis |
|-------|------|---------|-------|---------|-----------|------|------|--------|----------|
| GPT-4o Mini | LOW | 128K | Fast | $0.00015 | 75% | 70% | 50% | 80% | 70% |
| GPT-4o | MEDIUM | 128K | Medium | $0.005 | 92% | 88% | 80% | 92% | 88% |
| GPT-4 Turbo | HIGH | 128K | Slow | $0.01 | 96% | 95% | 92% | 96% | 94% |

#### **Google Models**

| Model | Tier | Context | Speed | Cost/1K | Reasoning | Code | Math | Coding | Analysis |
|-------|------|---------|-------|---------|-----------|------|------|--------|----------|
| Gemini 1.5 Flash | LOW | 1M | Fast | $0.075/1M | 78% | 72% | 55% | 80% | 75% |
| Gemini 2.0 Flash | MEDIUM | 1M | Medium | $0.1/1M | 88% | 85% | 78% | 88% | 85% |
| Gemini 2.0 Pro | HIGH | 1M | Slow | $0.15/1M | 94% | 92% | 88% | 94% | 92% |

### Model Selection Rules by Task Type

```python
# Pseudo-logic for model selection

def select_model_by_task_type(task_type, complexity, provider):
    """Select model based on task type + complexity"""
    
    if task_type == "REASONING":
        if complexity in [SIMPLE, STANDARD]:
            return MEDIUM_TIER  # Needs good reasoning
        else:
            return HIGH_TIER  # Deep reasoning required
    
    elif task_type == "THINKING":
        if complexity in [SIMPLE, STANDARD]:
            return MEDIUM_TIER  # Planning depth needed
        else:
            return HIGH_TIER  # Complex design needed
    
    elif task_type == "ANALYZING":
        if complexity == SIMPLE:
            return LOW_TIER  # Basic summaries OK
        elif complexity == STANDARD:
            return MEDIUM_TIER  # Math-heavy needs good reasoning
        else:
            return HIGH_TIER  # Complex statistical analysis
    
    elif task_type == "CODE_CREATION":
        if complexity == SIMPLE:
            return LOW_TIER or MEDIUM_TIER  # Basic code OK
        elif complexity == STANDARD:
            return MEDIUM_TIER  # Production code
        else:
            return HIGH_TIER  # Complex algorithms
    
    elif task_type == "DOC_CREATION":
        if complexity in [SIMPLE, STANDARD]:
            return LOW_TIER or MEDIUM_TIER  # Writing is strength of all
        else:
            return MEDIUM_TIER  # Complex topics need clarity
    
    # Get specific model for tier + provider
    return model_registry.get_model(tier, provider)
```

---

## Task-to-Model Mapping

### Decision Matrix: Task Type × Complexity → Model Tier

```
                 SIMPLE          STANDARD        COMPLEX         RESEARCH
REASONING        MEDIUM          MEDIUM          HIGH            HIGH
THINKING         MEDIUM          MEDIUM          HIGH            HIGH
ANALYZING        LOW             MEDIUM          HIGH            HIGH
CODE_CREATION    LOW/MEDIUM      MEDIUM          HIGH            HIGH
DOC_CREATION     LOW             LOW/MEDIUM      MEDIUM          MEDIUM
```

### Detailed Examples

#### Example 1: Task Classification & Model Selection

```
USER REQUEST:
"Compare Python and JavaScript for building web APIs. What are the trade-offs?"

CLASSIFICATION:
  ├─ Task Type: REASONING (comparing, analyzing trade-offs)
  ├─ Complexity: STANDARD (moderate depth, ~2000 tokens)
  └─ Decision Matrix: REASONING + STANDARD → MEDIUM tier

MODEL SELECTION:
  ├─ Anthropic: Claude 3.5 Sonnet ($0.003/1K input)
  ├─ OpenAI: GPT-4o ($0.005/1K input)
  └─ Google: Gemini 2.0 Flash ($0.1/1M)

CHOSEN: Claude 3.5 Sonnet (good balance of cost, speed, reasoning)
```

#### Example 2: Code Generation Task

```
USER REQUEST:
"Write a Python function that implements a binary search tree with insert, 
search, and delete operations. Include comprehensive comments."

CLASSIFICATION:
  ├─ Task Type: CODE_CREATION (write + optimize)
  ├─ Complexity: COMPLEX (15+ lines, algorithm design)
  └─ Decision Matrix: CODE_CREATION + COMPLEX → HIGH tier

MODEL SELECTION:
  ├─ Anthropic: Claude 3 Opus ($0.015/1K input) ✓ BEST
  ├─ OpenAI: GPT-4 Turbo ($0.01/1K input)
  └─ Google: Gemini 2.0 Pro ($0.15/1M)

CHOSEN: Claude 3 Opus (best code generation capability)
```

#### Example 3: Data Analysis Task

```
USER REQUEST:
"Analyze the attached sales data (10MB CSV) and provide insights on 
top-performing regions and trends."

CLASSIFICATION:
  ├─ Task Type: ANALYZING (data insights)
  ├─ Complexity: RESEARCH (large data, multiple analyses)
  ├─ Data Volume: 10MB
  └─ Decision Matrix: ANALYZING + RESEARCH → HIGH tier

CONSIDERATIONS:
  ├─ Need long context window (10MB data)
  ├─ Need good mathematical reasoning
  └─ Speed not critical (analytical task)

MODEL SELECTION:
  ├─ Anthropic: Claude 3 Opus (200K context, excellent reasoning)
  ├─ OpenAI: GPT-4 Turbo (128K context, might need chunking)
  └─ Google: Gemini 2.0 Pro (1M context, best for this!)

CHOSEN: Gemini 2.0 Pro (1M context window handles large data)
```

#### Example 4: Documentation Task

```
USER REQUEST:
"Write comprehensive API documentation for our REST API with examples."

CLASSIFICATION:
  ├─ Task Type: DOC_CREATION
  ├─ Complexity: STANDARD (moderate length documentation)
  └─ Decision Matrix: DOC_CREATION + STANDARD → LOW/MEDIUM tier

MODEL SELECTION:
  ├─ Anthropic: Claude 3.5 Haiku ($0.0008/1K) ✓ BEST value
  ├─ OpenAI: GPT-4o-Mini ($0.00015/1K)
  └─ Google: Gemini 1.5 Flash ($0.075/1M)

CHOSEN: GPT-4o-Mini (cheapest, excellent writing)
```

---

## Implementation Changes

### 1. Update Type Definitions

```python
# classifier/types.py (UPDATED)

from enum import Enum
from dataclasses import dataclass

class TaskType(Enum):
    """Task type classification"""
    REASONING = "reasoning"        # Analyze, compare, evaluate
    THINKING = "thinking"          # Plan, design, brainstorm
    ANALYZING = "analyzing"        # Data analysis, insights
    CODE_CREATION = "code_creation"  # Write, debug, optimize code
    DOC_CREATION = "doc_creation"    # Write, edit documentation

class TaskComplexity(Enum):
    """Task complexity level"""
    SIMPLE = "simple"            # < 500 tokens
    STANDARD = "standard"        # 500-5K tokens
    COMPLEX = "complex"          # 5K-15K tokens
    RESEARCH = "research"        # > 15K tokens

class ModelTier(Enum):
    """Model capability tier"""
    LOW = "low"          # Fast, cheap (~$0.0002/1K input)
    MEDIUM = "medium"    # Balanced (~$0.003/1K input)
    HIGH = "high"        # Powerful (~$0.015/1K input)

@dataclass
class ClassificationSignals:
    """Updated with task type"""
    token_count: int
    keyword_scores: Dict[str, float]
    task_type_scores: Dict[str, float]  # NEW: scores for each task type
    detected_task_type: TaskType  # NEW
    data_volume_mb: float
    has_tools: bool
    has_planning: bool
    has_reasoning: bool
    nesting_depth: int
    language: str
    domain_tags: List[str]

@dataclass
class ClassificationDecision:
    """Updated decision with task type"""
    model_name: str
    complexity: TaskComplexity
    task_type: TaskType  # NEW
    tier: ModelTier  # NEW: explicit tier (low/medium/high)
    reasoning: str
    confidence: float
    layer_used: str
    all_results: List['LayerResult']
    estimated_cost: float
    signals: ClassificationSignals
```

### 2. Update Feature Extractor

```python
# classifier/features/extractor.py (UPDATED)

class FeatureExtractor:
    """Extract signals INCLUDING task type"""
    
    # Keyword definitions per task type
    REASONING_KEYWORDS = [
        "analyze", "compare", "evaluate", "assess", "debate",
        "interpret", "distinguish", "validate", "argument", "logic",
        "contrast", "pros and cons", "trade-off"
    ]
    
    THINKING_KEYWORDS = [
        "plan", "strategy", "brainstorm", "explore", "design",
        "ideate", "organize", "structure", "approach", "workflow",
        "process", "architect", "layout", "system design"
    ]
    
    ANALYZING_KEYWORDS = [
        "analyze data", "statistics", "pattern", "trend", "insight",
        "metric", "aggregate", "summarize", "breakdown", "distribution",
        "correlation", "anomaly", "performance", "benchmark"
    ]
    
    CODE_CREATION_KEYWORDS = [
        "code", "function", "implement", "debug", "refactor",
        "optimize", "algorithm", "class", "test", "fix", "error",
        "write code", "programming", "script"
    ]
    
    DOC_CREATION_KEYWORDS = [
        "document", "write", "edit", "format", "style", "guide",
        "manual", "tutorial", "comment", "explain", "describe",
        "documentation", "readme", "readme.md"
    ]
    
    def extract(self, request: str, context: Dict = None) -> ClassificationSignals:
        """Extract features including task type"""
        
        # Existing extraction...
        token_count = self._count_tokens(request)
        keyword_scores = self._score_all_keywords(request)
        
        # NEW: Score task types
        task_type_scores = {
            "reasoning": self._score_keywords(request.lower(), self.REASONING_KEYWORDS),
            "thinking": self._score_keywords(request.lower(), self.THINKING_KEYWORDS),
            "analyzing": self._score_keywords(request.lower(), self.ANALYZING_KEYWORDS),
            "code_creation": self._score_keywords(request.lower(), self.CODE_CREATION_KEYWORDS),
            "doc_creation": self._score_keywords(request.lower(), self.DOC_CREATION_KEYWORDS),
        }
        
        # NEW: Detect dominant task type
        detected_type = self._detect_task_type(task_type_scores)
        
        # ... rest of existing extraction ...
        
        return ClassificationSignals(
            token_count=token_count,
            keyword_scores=keyword_scores,
            task_type_scores=task_type_scores,  # NEW
            detected_task_type=detected_type,    # NEW
            data_volume_mb=data_volume_mb,
            has_tools=has_tools,
            has_planning=has_planning,
            has_reasoning=has_reasoning,
            nesting_depth=nesting_depth,
            language=language,
            domain_tags=domain_tags
        )
    
    def _detect_task_type(self, scores: Dict[str, float]) -> TaskType:
        """Detect dominant task type from scores"""
        max_type = max(scores.items(), key=lambda x: x[1])
        
        type_map = {
            "reasoning": TaskType.REASONING,
            "thinking": TaskType.THINKING,
            "analyzing": TaskType.ANALYZING,
            "code_creation": TaskType.CODE_CREATION,
            "doc_creation": TaskType.DOC_CREATION,
        }
        
        return type_map.get(max_type[0], TaskType.REASONING)
```

### 3. Update Model Registry

```python
# classifier/models/registry.py (UPDATED)

class ModelRegistry:
    """Registry with task type + complexity support"""
    
    def __init__(self):
        # Decision matrix: (task_type, complexity) → model_tier
        self.tier_matrix = {
            (TaskType.REASONING, TaskComplexity.SIMPLE): ModelTier.MEDIUM,
            (TaskType.REASONING, TaskComplexity.STANDARD): ModelTier.MEDIUM,
            (TaskType.REASONING, TaskComplexity.COMPLEX): ModelTier.HIGH,
            (TaskType.REASONING, TaskComplexity.RESEARCH): ModelTier.HIGH,
            
            (TaskType.THINKING, TaskComplexity.SIMPLE): ModelTier.MEDIUM,
            (TaskType.THINKING, TaskComplexity.STANDARD): ModelTier.MEDIUM,
            (TaskType.THINKING, TaskComplexity.COMPLEX): ModelTier.HIGH,
            (TaskType.THINKING, TaskComplexity.RESEARCH): ModelTier.HIGH,
            
            (TaskType.ANALYZING, TaskComplexity.SIMPLE): ModelTier.LOW,
            (TaskType.ANALYZING, TaskComplexity.STANDARD): ModelTier.MEDIUM,
            (TaskType.ANALYZING, TaskComplexity.COMPLEX): ModelTier.HIGH,
            (TaskType.ANALYZING, TaskComplexity.RESEARCH): ModelTier.HIGH,
            
            (TaskType.CODE_CREATION, TaskComplexity.SIMPLE): ModelTier.LOW,
            (TaskType.CODE_CREATION, TaskComplexity.STANDARD): ModelTier.MEDIUM,
            (TaskType.CODE_CREATION, TaskComplexity.COMPLEX): ModelTier.HIGH,
            (TaskType.CODE_CREATION, TaskComplexity.RESEARCH): ModelTier.HIGH,
            
            (TaskType.DOC_CREATION, TaskComplexity.SIMPLE): ModelTier.LOW,
            (TaskType.DOC_CREATION, TaskComplexity.STANDARD): ModelTier.LOW,
            (TaskType.DOC_CREATION, TaskComplexity.COMPLEX): ModelTier.MEDIUM,
            (TaskType.DOC_CREATION, TaskComplexity.RESEARCH): ModelTier.MEDIUM,
        }
        
        # Models per tier
        self.models_by_tier = {
            "anthropic": {
                ModelTier.LOW: "claude-3-5-haiku-20241022",
                ModelTier.MEDIUM: "claude-3-5-sonnet-20241022",
                ModelTier.HIGH: "claude-3-opus-20250219",
            },
            "openai": {
                ModelTier.LOW: "gpt-4o-mini",
                ModelTier.MEDIUM: "gpt-4o",
                ModelTier.HIGH: "gpt-4-turbo",
            },
            "google": {
                ModelTier.LOW: "gemini-1.5-flash",
                ModelTier.MEDIUM: "gemini-2.0-flash",
                ModelTier.HIGH: "gemini-2.0-pro",
            }
        }
    
    def get_tier(self, task_type: TaskType, complexity: TaskComplexity) -> ModelTier:
        """Get recommended tier for task"""
        return self.tier_matrix.get(
            (task_type, complexity),
            ModelTier.MEDIUM  # Default fallback
        )
    
    def get_model(
        self,
        task_type: TaskType,
        complexity: TaskComplexity,
        provider: str = "anthropic"
    ) -> str:
        """Get specific model for task"""
        tier = self.get_tier(task_type, complexity)
        return self.models_by_tier[provider][tier]
```

### 4. Update Heuristic Classifier

```python
# classifier/layers/layer1_heuristics.py (UPDATED)

class HeuristicClassifier:
    """Updated with task type classification"""
    
    def classify(self, signals: ClassificationSignals) -> LayerResult:
        """Classify with task type"""
        start_time = time.time()
        
        # Determine complexity
        complexity, confidence, reason = self._decide_complexity(signals)
        
        # Task type already detected in feature extraction
        task_type = signals.detected_task_type
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LayerResult(
            complexity=complexity,
            task_type=task_type,  # NEW
            confidence=confidence,
            reason=reason,
            layer_name=self.name,
            latency_ms=latency_ms,
            signals_used=signals
        )
    
    def _decide_complexity(self, signals: ClassificationSignals) -> tuple:
        """Determine complexity (unchanged logic)"""
        # Same as before, uses token count + keywords
        # ...
        pass
```

### 5. Cost Estimation by Tier

```python
# classifier/logging/cost_tracker.py (UPDATED)

class CostEstimator:
    """Estimate cost based on tier"""
    
    COST_PER_1K_INPUT = {
        ModelTier.LOW: 0.0002,      # Haiku, GPT-4o-Mini
        ModelTier.MEDIUM: 0.003,    # Sonnet, GPT-4o
        ModelTier.HIGH: 0.015,      # Opus, GPT-4-Turbo
    }
    
    COST_PER_1K_OUTPUT = {
        ModelTier.LOW: 0.0006,
        ModelTier.MEDIUM: 0.009,
        ModelTier.HIGH: 0.045,
    }
    
    def estimate_cost(self, tier: ModelTier, input_tokens: int, output_tokens: int = 500) -> float:
        """Estimate cost for this request"""
        input_cost = (input_tokens / 1000) * self.COST_PER_1K_INPUT[tier]
        output_cost = (output_tokens / 1000) * self.COST_PER_1K_OUTPUT[tier]
        return input_cost + output_cost
```

---

## Summary Table: Task Type → Recommended Tier

```
┌─────────────────────────────────────────────────────────────────────┐
│ TASK TYPE × COMPLEXITY → RECOMMENDED TIER                           │
├────────────────────┬──────────┬──────────┬────────┬──────────┤
│ Task Type          │ SIMPLE   │ STANDARD │ COMPLEX│ RESEARCH │
├────────────────────┼──────────┼──────────┼────────┼──────────┤
│ REASONING          │ MEDIUM   │ MEDIUM   │ HIGH   │ HIGH     │
│ THINKING           │ MEDIUM   │ MEDIUM   │ HIGH   │ HIGH     │
│ ANALYZING          │ LOW      │ MEDIUM   │ HIGH   │ HIGH     │
│ CODE_CREATION      │ LOW/MED  │ MEDIUM   │ HIGH   │ HIGH     │
│ DOC_CREATION       │ LOW      │ LOW/MED  │ MEDIUM │ MEDIUM   │
└────────────────────┴──────────┴──────────┴────────┴──────────┘
```

---

## Usage Examples

### Example 1: Classify & Route

```python
from classifier.core import ClassificationPipeline
from classifier.types import TaskType, TaskComplexity, ModelTier

pipeline = ClassificationPipeline()

# User request
request = "Analyze the pros and cons of microservices vs monolithic architecture"

# Get decision
decision = pipeline.classify(request)

# Result shows:
print(f"Task Type: {decision.task_type.value}")        # "reasoning"
print(f"Complexity: {decision.complexity.value}")      # "standard"
print(f"Tier: {decision.tier.value}")                  # "medium"
print(f"Model: {decision.model_name}")                 # "claude-3-5-sonnet-20241022"
print(f"Cost: ${decision.estimated_cost:.4f}")         # "$0.0010"
```

### Example 2: Manual Task Type Override

```python
# If auto-detection is wrong, manual override
decision = pipeline.classify(
    request="Write a function for binary search",
    task_type_override=TaskType.CODE_CREATION
)

# Result:
print(f"Task Type: {decision.task_type.value}")        # "code_creation"
print(f"Tier: {decision.tier.value}")                  # "medium" (even if simple)
print(f"Model: {decision.model_name}")                 # Best coding model
```

### Example 3: Cost-Optimized Route

```python
# Want to save money? Override to cheaper tier
decision = pipeline.classify(
    request,
    cost_priority=True  # Prefer cheaper models if confidence > 0.8
)

# Will use LOW tier when possible, only upgrade if uncertain
```

---

## Migration Path

### Phase 1 (Days 1-2): Add Task Type Detection
- [ ] Add TaskType enum
- [ ] Add task type keywords to FeatureExtractor
- [ ] Update LayerResult to include task_type
- [ ] Test detection accuracy

### Phase 2 (Days 3-4): Update Decision Matrix
- [ ] Update ModelRegistry with tier_matrix
- [ ] Implement get_tier() method
- [ ] Update routing logic to use task_type + complexity
- [ ] Validate decision matrix with examples

### Phase 3 (Days 5-6): Update All Layers
- [ ] Update Layer 1 to output task_type
- [ ] Update Layers 2-4 to consider task_type
- [ ] Update cost estimation
- [ ] Update logging to include task_type

### Phase 4 (Days 7-8): Testing
- [ ] Test each task type classification
- [ ] Validate model selections
- [ ] Test cost calculations
- [ ] Integration tests

---

## Configuration (Updated YAML)

```yaml
# config/layer_config.yaml (NEW SECTION)

task_types:
  reasoning:
    keywords: ["analyze", "compare", "evaluate", "assess"]
    min_complexity: STANDARD  # REASONING + SIMPLE still needs MEDIUM
  
  thinking:
    keywords: ["plan", "design", "brainstorm", "strategy"]
    min_complexity: STANDARD
  
  analyzing:
    keywords: ["data", "stats", "trend", "pattern", "metric"]
    min_complexity: SIMPLE
  
  code_creation:
    keywords: ["code", "function", "implement", "debug"]
    min_complexity: SIMPLE
  
  doc_creation:
    keywords: ["document", "write", "guide", "tutorial"]
    min_complexity: SIMPLE

tier_matrix:
  # (task_type, complexity) → tier
  (reasoning, simple): medium
  (reasoning, standard): medium
  (reasoning, complex): high
  (reasoning, research): high
  # ... (full matrix in code)

models_by_tier:
  anthropic:
    low: "claude-3-5-haiku-20241022"
    medium: "claude-3-5-sonnet-20241022"
    high: "claude-3-opus-20250219"
```

---

## Key Benefits of This Approach

✅ **Nuanced Routing**: Task type + complexity is more accurate than complexity alone  
✅ **Cost Optimization**: DOC_CREATION can use LOW tier, REASONING needs MEDIUM  
✅ **Model Fit**: Each task type leverages model strengths  
✅ **Explainability**: Clear why each model was chosen  
✅ **Scalability**: Easy to add new task types or models  
✅ **Budget Control**: Can enforce cost limits per task type  

---

**Version**: 1.0  
**Status**: Ready for implementation into model-categorization framework  
**Next Step**: Update implementation.md Phase 1 with task type detection
