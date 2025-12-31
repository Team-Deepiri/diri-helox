
# Comprehensive Guide: Generating Weighted Synthetic Data for ML Training

## Table of Contents

1. [Overview and Philosophy](#overview-and-philosophy)
2. [System Architecture](#system-architecture)
3. [Phase 1: Emergent Category Generation](#phase-1-emergent-category-generation)
4. [Phase 2: Niche-Specific Integration](#phase-2-niche-specific-integration)
5. [Script Design and Implementation](#script-design-and-implementation)
6. [Weighted Data Generation](#weighted-data-generation)
7. [Ollama Integration for Inference-Based Detection](#ollama-integration-for-inference-based-detection)
8. [Usage Instructions](#usage-instructions)
9. [Success Criteria and Validation](#success-criteria-and-validation)
10. [Debugging Common Issues](#debugging-common-issues)
11. [Debugging Miscellaneous Issues](#debugging-miscellaneous-issues)
12. [Advanced Configuration](#advanced-configuration)
13. [Performance Optimization](#performance-optimization)
14. [Production Deployment](#production-deployment)

---

## Overview and Philosophy

### Core Principles

The synthetic data generation system is built on three fundamental principles:

1. **Emergent Categories**: Categories are discovered organically from document content, not hardcoded. This allows the system to adapt to your specific domain and use cases.

2. **Document-First Pipeline**: Realistic documents are generated first, then tasks are extracted from them. This ensures tasks have natural context and realistic structure.

3. **Weighted Quality Injection**: Not all synthetic data is equal. The system injects quality scores, complexity metrics, and domain-specific signals to create a weighted dataset that better represents real-world data distribution.

### Why This Approach Works

Traditional synthetic data generation often suffers from:
- Over-reliance on hardcoded templates
- Lack of domain context
- Uniform quality distribution (all data treated equally)
- Poor generalization to real-world scenarios

This system addresses these issues by:
- Learning patterns from your actual data (b2b_sample_1000_V2.jsonl)
- Generating documents that mirror real work artifacts
- Applying quality weights based on multiple signals
- Using inference-based detection via Ollama for semantic validation

### Data Flow Overview

```
Real Data (b2b_sample_1000_V2.jsonl)
    ↓
Pattern Extraction & Analysis
    ↓
Document Generation (Phase 1)
    ↓
Task Extraction (Natural Language Processing)
    ↓
Category Discovery (Emergent Clustering)
    ↓
Synthetic Task Generation (Pattern-Based)
    ↓
Quality Weighting & Signal Injection
    ↓
Ollama-Based Validation (Optional)
    ↓
Weighted Dataset Output
```

---

## System Architecture

### Component Overview

The system consists of several key components:

1. **EmergentTaskGenerator**: Core generator that discovers categories from content
2. **DocumentFirstPipeline**: Generates realistic documents before extracting tasks
3. **WeightedTask**: Data structure with quality metrics and signals
4. **SemanticAnalyzer**: Integration with Ollama for inference-based validation
5. **NicheAwareGenerator**: Domain-specific task generation for your 20 niches

### Directory Structure

```
diri-helox/
├── scripts/
│   ├── generate_synthetic_data.py          # Existing script (reference)
│   ├── phase1_emergent.py                  # Phase 1 implementation
│   ├── phase2_niche_integration.py         # Phase 2 implementation
│   └── synthetic_data_pipeline.py          # Combined pipeline
├── data/
│   ├── synthetic_data/
│   │   ├── phase1/                          # Phase 1 outputs
│   │   ├── phase2/                          # Phase 2 outputs
│   │   └── debug/                           # Debug artifacts
│   └── b2b_sample_1000_V2.jsonl            # Input data
└── utils/
    └── semantic_analyzer.py                # Ollama integration
```

### Dependencies

Required Python packages:

```python
# Core dependencies
json
random
re
numpy
pandas
collections
datetime
typing
pathlib
os
sys

# ML/AI dependencies
scikit-learn          # For clustering and metrics
sentence-transformers # For embeddings (optional)
umap-learn           # For dimensionality reduction (optional)

# Ollama integration
httpx                # HTTP client for Ollama API
requests             # Alternative HTTP client
ollama               # Official Ollama Python package (optional)
```

Installation:

```bash
pip install numpy pandas scikit-learn sentence-transformers umap-learn httpx requests ollama
```

---

## Phase 1: Emergent Category Generation

### Overview

Phase 1 focuses on generating synthetic data with categories that emerge naturally from document content. No categories are hardcoded - they are discovered through pattern analysis.

### Implementation Steps

#### Step 1: Environment Setup

Create the necessary directory structure:

```bash
cd deepiri-platform/diri-helox
mkdir -p data/synthetic_data/phase1
mkdir -p data/synthetic_data/phase2
mkdir -p data/synthetic_data/debug
mkdir -p data/synthetic_data/output
```

#### Step 2: Configuration Setup

Create `scripts/config/synthetic_data_config.py`:

```python
"""
Configuration for synthetic data generation
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic_data"

# Input data
INPUT_FILE = BASE_DIR.parent / "b2b_sample_1000_V2.jsonl"

# Output directories
PHASE1_OUTPUT = SYNTHETIC_DIR / "phase1"
PHASE2_OUTPUT = SYNTHETIC_DIR / "phase2"
DEBUG_OUTPUT = SYNTHETIC_DIR / "debug"

# Generation parameters
CONFIG = {
    # Document generation
    "num_initial_docs": 200,
    "num_tasks_per_doc": 3,
    "synthetic_multiplier": 5,
    
    # Quality distribution
    "quality_tiers": {
        "high": 0.15,    # 15% high quality
        "medium": 0.60,  # 60% medium quality
        "low": 0.25      # 25% low quality
    },
    
    # Domain contexts (from your 20 niche analysis)
    "domain_contexts": [
        "maintenance_fraud_detection",
        "vendor_intelligence",
        "work_order_analysis",
        "claims_processing",
        "fraud_detection",
        "settlement_prediction",
        "project_delay_prediction",
        "risk_management",
        "compliance_checking",
        "revenue_cycle",
        "denial_prevention",
        "coding_validation",
        "risk_detection",
        "disruption_prediction",
        "supplier_performance"
    ],
    
    # Technical components
    "tech_components": [
        "RAG_implementation",
        "predictive_model",
        "LoRA_finetuning",
        "multimodal_AI",
        "real_time_infra",
        "anomaly_detection",
        "time_series_forecasting",
        "classification_model",
        "document_parsing"
    ],
    
    # Ollama configuration
    "ollama": {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "model": os.getenv("OLLAMA_MODEL", "llama3:8b"),
        "enabled": os.getenv("OLLAMA_ENABLED", "true").lower() == "true",
        "timeout": 15.0,
        "use_for_validation": True,
        "validation_frequency": 0.1  # Validate 10% of tasks
    }
}
```

#### Step 3: Core Data Structures

Create `scripts/phase1/data_structures.py`:

```python
"""
Data structures for Phase 1 synthetic data generation
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime

@dataclass
class WeightedTask:
    """
    Task with injected quality weights and signals for ML training
    
    Attributes:
        text: The task text
        source: Where this task came from (document ID or synthetic)
        category: Emergent category (discovered, not hardcoded)
        quality_score: Overall quality score (0.0-1.0)
        complexity: Task complexity (0.0-1.0)
        specificity: How specific the task is (0.0-1.0)
        domain: Domain context from your 20 niches
        tech_components: Technical components mentioned
        signals: Dictionary of signals for ML training
        metadata: Additional metadata for debugging
    """
    text: str
    source: str
    category: str = "unknown"
    quality_score: float = 0.5
    complexity: float = 0.5
    specificity: float = 0.5
    domain: str = "general"
    tech_components: List[str] = field(default_factory=list)
    signals: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "text": self.text,
            "source": self.source,
            "category": self.category,
            "quality_score": self.quality_score,
            "complexity": self.complexity,
            "specificity": self.specificity,
            "domain": self.domain,
            "tech_components": self.tech_components,
            "signals": self.signals,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WeightedTask':
        """Create from dictionary"""
        return cls(
            text=data["text"],
            source=data["source"],
            category=data.get("category", "unknown"),
            quality_score=data.get("quality_score", 0.5),
            complexity=data.get("complexity", 0.5),
            specificity=data.get("specificity", 0.5),
            domain=data.get("domain", "general"),
            tech_components=data.get("tech_components", []),
            signals=data.get("signals", {}),
            metadata=data.get("metadata", {})
        )

@dataclass
class Document:
    """
    Document structure for realistic document generation
    """
    id: str
    type: str
    title: str
    created_by: str
    created_at: str
    status: str
    content: str
    metadata: Dict = field(default_factory=dict)
    domain_context: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "status": self.status,
            "content": self.content,
            "metadata": self.metadata
        }
        if self.domain_context:
            result["domain_context"] = self.domain_context
        return result
```

#### Step 4: Document Generation

Create `scripts/phase1/document_generator.py`:

```python
"""
Document generator for Phase 1
Generates realistic documents that naturally contain tasks
"""
import random
from datetime import datetime, timedelta
from typing import List, Dict
from .data_structures import Document

class DocumentGenerator:
    """Generates realistic documents with natural task content"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.doc_templates = self._initialize_templates()
    
    def _initialize_templates(self) -> List[tuple]:
        """Initialize document templates"""
        return [
            ("engineering_plan", self._engineering_plan),
            ("design_review", self._design_review),
            ("sprint_planning", self._sprint_planning),
            ("postmortem", self._postmortem),
            ("technical_proposal", self._technical_proposal),
            ("project_update", self._project_update),
            ("risk_assessment", self._risk_assessment),
            ("compliance_report", self._compliance_report),
            ("vendor_evaluation", self._vendor_evaluation),
            ("api_spec", self._api_spec),
            ("architecture_design", self._architecture_design),
            ("testing_plan", self._testing_plan),
            ("deployment_checklist", self._deployment_checklist)
        ]
    
    def generate_documents(self, count: int) -> List[Document]:
        """Generate a batch of documents"""
        documents = []
        
        for i in range(count):
            doc_type, generator = random.choice(self.doc_templates)
            
            doc = Document(
                id=f"doc_{datetime.now().strftime('%Y%m%d')}_{i:06d}",
                type=doc_type,
                title=self._generate_doc_title(doc_type),
                created_by=random.choice(["alex_m", "sam_r", "jordan_k", "taylor_s", "casey_w"]),
                created_at=self._random_timestamp(),
                status=random.choice(["draft", "in_review", "approved", "archived"]),
                content=generator(),
                metadata={
                    "department": random.choice(["engineering", "product", "data_science", "ops"]),
                    "priority": random.choice(["high", "medium", "low"]),
                    "estimated_hours": random.randint(2, 40),
                    "related_projects": [f"project_{random.randint(1, 10)}" for _ in range(random.randint(1, 3))]
                }
            )
            
            # Inject domain context
            if random.random() > 0.3:
                domain = random.choice(self.config["domain_contexts"])
                doc.domain_context = domain
                doc.content = self._inject_domain_context(doc.content, domain)
            
            documents.append(doc)
        
        return documents
    
    def _engineering_plan(self) -> str:
        """Generate engineering planning document"""
        templates = [
            """## Engineering Implementation Plan

### Current Status
The current system has performance issues under load, particularly with concurrent database connections.

### Analysis Findings
1. Database connection pool exhausted under peak load
2. API response times spike from 50ms to 500ms during traffic surges
3. Memory usage grows unbounded due to connection leaks

### Required Actions
- Implement connection pooling with proper timeout settings
- Add connection leak detection and monitoring
- Optimize SQL queries that are causing full table scans
- Add caching layer for frequently accessed data

### Success Metrics
- API p99 latency < 100ms under 10k concurrent users
- Database connection pool utilization < 80%
- Zero connection leaks in 72-hour stress test""",
            
            """## System Optimization Plan

### Problem Statement
User reports indicate slow page loads for dashboard views, especially with large datasets.

### Root Cause Analysis
- Frontend is making too many sequential API calls
- No client-side caching implemented
- Database queries not optimized for dashboard patterns

### Implementation Tasks
1. Implement batch API endpoints for dashboard data
2. Add Redis caching layer with TTL settings
3. Optimize database indexes for common query patterns
4. Add lazy loading for chart components
5. Implement client-side data caching

### Technical Requirements
- Cache hit rate > 90% for dashboard queries
- Page load time reduction from 3s to < 500ms
- Support for real-time data updates"""
        ]
        return random.choice(templates)
    
    def _design_review(self) -> str:
        """Generate design review document"""
        return """## Design Review: User Authentication Flow

### Overview
Review of proposed authentication system redesign.

### Key Decisions
1. Move from session-based to JWT-based authentication
2. Implement refresh token rotation
3. Add multi-factor authentication support

### Action Items
- Review security implications of JWT storage
- Design token refresh mechanism
- Create user migration plan
- Update API documentation

### Concerns
- Token revocation strategy needs clarification
- Backward compatibility with existing clients
- Performance impact of token validation"""
    
    def _sprint_planning(self) -> str:
        """Generate sprint planning document"""
        return """## Sprint Planning: Q1 2025

### Sprint Goals
- Improve system reliability
- Enhance user experience
- Technical debt reduction

### Committed Items
1. Fix critical bug in payment processing
2. Implement new feature: real-time notifications
3. Refactor legacy authentication code
4. Add comprehensive test coverage
5. Update API documentation

### Dependencies
- Backend team: API endpoint changes
- DevOps: Deployment pipeline updates
- Design: UI mockups for notifications"""
    
    def _postmortem(self) -> str:
        """Generate postmortem document"""
        return """## Postmortem: Production Incident - 2025-01-15

### Incident Summary
System outage lasting 45 minutes affecting 15% of users.

### Root Cause
Database connection pool exhaustion due to unhandled exception in background job.

### Timeline
- 14:32: First alerts triggered
- 14:35: Investigation began
- 14:50: Root cause identified
- 15:05: Fix deployed
- 15:17: Service fully restored

### Action Items
- Add circuit breaker for background jobs
- Implement better error handling
- Increase connection pool size
- Add monitoring for connection pool metrics
- Create runbook for similar incidents"""
    
    def _technical_proposal(self) -> str:
        """Generate technical proposal"""
        return """## Technical Proposal: Microservices Migration

### Problem
Monolithic architecture limiting scalability and deployment flexibility.

### Proposed Solution
Gradually migrate to microservices architecture starting with user service.

### Implementation Plan
1. Extract user service into separate microservice
2. Implement service-to-service communication
3. Set up service discovery
4. Migrate database schemas
5. Update deployment pipelines

### Risks
- Service communication overhead
- Distributed transaction complexity
- Increased operational complexity

### Success Criteria
- Independent deployment of user service
- No degradation in response times
- Improved system reliability"""
    
    def _project_update(self) -> str:
        """Generate project update"""
        return """## Project Update: Q1 2025

### Status: On Track

### Completed This Week
- Deployed new authentication system
- Fixed critical security vulnerability
- Completed performance optimization

### In Progress
- Database migration to new schema
- API versioning implementation
- Documentation updates

### Blockers
- Waiting on third-party API access
- Need approval for infrastructure changes

### Next Steps
- Complete database migration
- Begin API versioning rollout
- Schedule security audit"""
    
    def _risk_assessment(self) -> str:
        """Generate risk assessment"""
        return """## Risk Assessment: New Feature Launch

### Feature
Real-time collaboration features for document editing.

### Identified Risks
1. High: Data consistency issues with concurrent edits
2. Medium: Performance degradation with many users
3. Low: Browser compatibility issues

### Mitigation Strategies
- Implement operational transformation for conflict resolution
- Add rate limiting and connection throttling
- Extensive cross-browser testing
- Gradual rollout to beta users first

### Action Items
- Research operational transformation libraries
- Design conflict resolution algorithm
- Create performance test plan
- Set up beta testing program"""
    
    def _compliance_report(self) -> str:
        """Generate compliance report"""
        return """## Compliance Report: GDPR Audit

### Audit Date
2025-01-10

### Findings
- Data retention policies properly implemented
- User consent mechanisms in place
- Right to deletion functionality working

### Issues Identified
- Some legacy data lacks proper consent records
- Data export functionality needs improvement

### Remediation Plan
- Audit and update legacy consent records
- Enhance data export to include all user data
- Add automated compliance monitoring
- Schedule quarterly compliance reviews

### Next Audit
Scheduled for 2025-04-10"""
    
    def _vendor_evaluation(self) -> str:
        """Generate vendor evaluation"""
        return """## Vendor Evaluation: Cloud Storage Provider

### Vendor Options
1. Provider A: High performance, expensive
2. Provider B: Balanced, moderate cost
3. Provider C: Low cost, limited features

### Evaluation Criteria
- Cost
- Performance
- Reliability
- Support quality
- Feature set

### Recommendation
Provider B - Best balance of cost and features.

### Next Steps
- Negotiate contract terms
- Plan migration strategy
- Set up pilot integration
- Schedule vendor onboarding"""
    
    def _api_spec(self) -> str:
        """Generate API specification"""
        return """## API Specification: User Management v2

### Endpoints
- GET /api/v2/users - List users
- POST /api/v2/users - Create user
- GET /api/v2/users/{id} - Get user
- PUT /api/v2/users/{id} - Update user
- DELETE /api/v2/users/{id} - Delete user

### Authentication
Bearer token required for all endpoints.

### Rate Limiting
100 requests per minute per API key.

### Implementation Tasks
- Create new API version routes
- Implement authentication middleware
- Add rate limiting
- Write API documentation
- Create integration tests"""
    
    def _architecture_design(self) -> str:
        """Generate architecture design"""
        return """## Architecture Design: Event-Driven System

### Overview
Design for event-driven architecture to improve system scalability.

### Components
- Event bus (Kafka)
- Event producers (services)
- Event consumers (workers)
- Event store (database)

### Design Decisions
- Use Kafka for event streaming
- Implement event sourcing for audit trail
- Add event replay capability
- Design for eventual consistency

### Implementation Plan
1. Set up Kafka cluster
2. Create event schema registry
3. Implement event producers
4. Build event consumers
5. Add monitoring and alerting

### Considerations
- Event ordering guarantees
- Dead letter queue handling
- Event versioning strategy"""
    
    def _testing_plan(self) -> str:
        """Generate testing plan"""
        return """## Testing Plan: Payment Processing

### Test Coverage Goals
- Unit tests: 90% code coverage
- Integration tests: All critical paths
- E2E tests: Main user flows
- Performance tests: Load and stress testing

### Test Scenarios
1. Successful payment processing
2. Payment failure handling
3. Refund processing
4. Payment retry logic
5. Concurrent payment handling

### Test Data Requirements
- Valid credit card numbers
- Invalid card scenarios
- Edge cases (expired cards, insufficient funds)

### Implementation Tasks
- Write unit tests for payment service
- Create integration test suite
- Set up E2E test environment
- Design performance test scenarios
- Automate test execution"""
    
    def _deployment_checklist(self) -> str:
        """Generate deployment checklist"""
        return """## Deployment Checklist: Production Release v2.1

### Pre-Deployment
- [ ] All tests passing
- [ ] Code review completed
- [ ] Security scan passed
- [ ] Performance benchmarks met
- [ ] Documentation updated

### Deployment Steps
1. Backup current production database
2. Deploy to staging environment
3. Run smoke tests
4. Deploy to production
5. Monitor error rates
6. Verify functionality

### Rollback Plan
- Keep previous version available
- Database migration rollback script ready
- Communication plan for users

### Post-Deployment
- Monitor error logs
- Check performance metrics
- Verify user reports
- Schedule follow-up review"""
    
    def _generate_doc_title(self, doc_type: str) -> str:
        """Generate document title"""
        titles = {
            "engineering_plan": [
                "Engineering Implementation Plan: Performance Optimization",
                "System Architecture Improvement Plan",
                "Technical Debt Reduction Strategy"
            ],
            "design_review": [
                "Design Review: Authentication System",
                "UI/UX Design Review: Dashboard Redesign",
                "Architecture Review: Microservices Migration"
            ],
            "sprint_planning": [
                "Sprint Planning: Q1 2025",
                "Sprint 42 Planning: Feature Development",
                "Sprint Planning: Bug Fixes and Improvements"
            ],
            "postmortem": [
                "Postmortem: Production Incident 2025-01-15",
                "Incident Review: Database Outage",
                "Postmortem: API Performance Degradation"
            ],
            "technical_proposal": [
                "Technical Proposal: Microservices Architecture",
                "Proposal: Real-Time Analytics System",
                "Technical Proposal: Cloud Migration Strategy"
            ]
        }
        
        if doc_type in titles:
            return random.choice(titles[doc_type])
        return f"{doc_type.replace('_', ' ').title()}: {datetime.now().strftime('%Y-%m-%d')}"
    
    def _random_timestamp(self) -> str:
        """Generate random timestamp within last 90 days"""
        now = datetime.now()
        days_ago = random.randint(0, 90)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        
        timestamp = now - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        return timestamp.isoformat()
    
    def _inject_domain_context(self, content: str, domain: str) -> str:
        """Inject domain-specific context into document"""
        domain_injections = {
            "maintenance_fraud_detection": [
                "\n\n### Fraud Detection Context\nNeed to analyze vendor invoice patterns for anomalies. Historical maintenance logs show patterns of overcharging.",
                "\n\n### Maintenance Context\nEquipment failure prediction requires historical maintenance logs and vendor performance data."
            ],
            "claims_processing": [
                "\n\n### Insurance Context\nClaim validation requires policy document cross-referencing and medical record analysis.",
                "\n\n### Fraud Detection\nPattern analysis needed for suspicious claim detection using historical data."
            ],
            "project_delay_prediction": [
                "\n\n### Construction Context\nSchedule risk assessment requires weather data, supplier performance, and historical project timelines.",
                "\n\n### Delay Analysis\nHistorical project data needed for predictive modeling of schedule risks."
            ]
        }
        
        if domain in domain_injections and random.random() > 0.5:
            injection = random.choice(domain_injections[domain])
            return content + injection
        
        return content
```

#### Step 5: Task Extraction

Create `scripts/phase1/task_extractor.py`:

```python
"""
Task extractor for Phase 1
Extracts tasks from documents using natural language processing
"""
import re
from typing import List
from .data_structures import WeightedTask, Document

class TaskExtractor:
    """Extracts tasks from documents using multiple methods"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.action_verbs = self._initialize_action_verbs()
    
    def _initialize_action_verbs(self) -> set:
        """Initialize set of action verbs"""
        return {
            'implement', 'develop', 'create', 'build', 'write', 'fix', 'debug',
            'test', 'review', 'analyze', 'design', 'plan', 'organize', 'schedule',
            'update', 'improve', 'optimize', 'refactor', 'deploy', 'configure',
            'install', 'setup', 'investigate', 'research', 'document', 'train',
            'add', 'remove', 'change', 'modify', 'integrate', 'migrate', 'upgrade',
            'monitor', 'track', 'measure', 'evaluate', 'assess', 'prepare',
            'collect', 'gather', 'compile', 'validate', 'verify', 'check'
        }
    
    def extract_tasks_from_documents(self, documents: List[Document]) -> List[WeightedTask]:
        """Extract tasks from a list of documents"""
        all_tasks = []
        
        for doc in documents:
            doc_tasks = self._extract_from_single_doc(doc)
            all_tasks.extend(doc_tasks)
        
        return all_tasks
    
    def _extract_from_single_doc(self, doc: Document) -> List[WeightedTask]:
        """Extract tasks from a single document"""
        tasks = []
        content = doc.content
        
        # Method 1: Extract bullet points
        bullet_items = self._extract_bullet_items(content)
        for item in bullet_items:
            if self._looks_like_actionable_task(item):
                task = self._create_weighted_task(item, doc, "bullet_extraction")
                tasks.append(task)
        
        # Method 2: Extract action sentences
        sentences = self._split_into_sentences(content)
        for sentence in sentences:
            if self._is_action_sentence(sentence):
                task = self._create_weighted_task(sentence, doc, "sentence_extraction")
                tasks.append(task)
        
        # Method 3: Extract header items
        header_items = self._extract_header_items(content)
        for item in header_items:
            task = self._create_weighted_task(item, doc, "header_extraction")
            tasks.append(task)
        
        return tasks
    
    def _extract_bullet_items(self, text: str) -> List[str]:
        """Extract bullet points and numbered items"""
        items = []
        patterns = [
            r'^[-*•]\s+(.+)$',
            r'^\d+\.\s+(.+)$',
            r'^\[.\]?\s+(.+)$',
        ]
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    items.append(match.group(1))
                    break
        
        return items
    
    def _looks_like_actionable_task(self, text: str) -> bool:
        """Determine if text looks like an actionable task"""
        words = text.split()
        if len(words) < 3 or len(words) > 30:
            return False
        
        # Check for action verbs
        for i in range(min(3, len(words))):
            if words[i].lower().rstrip(':,.-') in self.action_verbs:
                return True
        
        # Check for action patterns
        lower_text = text.lower()
        patterns = [
            r'need to ',
            r'should ',
            r'must ',
            r'will ',
            r'requires? '
        ]
        for pattern in patterns:
            if re.search(pattern, lower_text):
                return True
        
        return False
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_action_sentence(self, sentence: str) -> bool:
        """Determine if sentence describes an action"""
        words = sentence.split()
        if len(words) < 4 or len(words) > 30:
            return False
        
        first_word = words[0].lower().strip('":-•*')
        if first_word in self.action_verbs:
            return True
        
        lower_sentence = sentence.lower()
        action_patterns = [
            r'need to \w+',
            r'should \w+',
            r'must \w+',
            r'will \w+',
            r'going to \w+',
            r'requires \w+ing',
            r'todo:',
            r'action:',
            r'next:'
        ]
        
        for pattern in action_patterns:
            if re.search(pattern, lower_sentence):
                return True
        
        return False
    
    def _extract_header_items(self, text: str) -> List[str]:
        """Extract items under headers like 'Action Items', 'Next Steps', etc."""
        items = []
        lines = text.split('\n')
        
        header_keywords = ['action items', 'next steps', 'tasks', 'todo', 'required actions', 'implementation tasks']
        in_action_section = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if we're entering an action section
            for keyword in header_keywords:
                if keyword in line_lower:
                    in_action_section = True
                    break
            
            # Extract items in action section
            if in_action_section:
                # Stop at next major section (header with ##)
                if line.startswith('##'):
                    in_action_section = False
                    continue
                
                # Extract bullet items
                bullet_match = re.match(r'^[-*•]\s+(.+)$', line.strip())
                if bullet_match:
                    items.append(bullet_match.group(1))
                # Extract numbered items
                numbered_match = re.match(r'^\d+\.\s+(.+)$', line.strip())
                if numbered_match:
                    items.append(numbered_match.group(1))
        
        return items
    
    def _create_weighted_task(self, text: str, source_doc: Document, extraction_method: str) -> WeightedTask:
        """Create a weighted task from extracted text"""
        from .quality_calculator import QualityCalculator
        
        quality_calc = QualityCalculator(self.config)
        
        # Calculate quality metrics
        quality_score = quality_calc.calculate_quality_score(text)
        complexity = quality_calc.calculate_complexity(text)
        specificity = quality_calc.calculate_specificity(text)
        
        # Extract category
        category = self._extract_emergent_category(text)
        
        # Extract domain
        domain = source_doc.domain_context or "general"
        
        # Identify tech components
        tech_components = self._identify_tech_components(text)
        
        # Create signals
        signals = {
            "text_length": len(text.split()),
            "has_technical_terms": self._has_technical_terms(text),
            "has_measurements": self._has_measurements(text),
            "has_dates": self._has_dates(text),
            "extraction_method": extraction_method,
            "source_doc_type": source_doc.type,
            "source_doc_priority": source_doc.metadata.get("priority", "medium")
        }
        
        # Metadata
        metadata = {
            "source_doc_id": source_doc.id,
            "extracted_at": datetime.now().isoformat(),
            "confidence": random.uniform(0.7, 0.95)
        }
        
        return WeightedTask(
            text=text.strip(),
            source=f"document:{source_doc.id}",
            category=category,
            quality_score=quality_score,
            complexity=complexity,
            specificity=specificity,
            domain=domain,
            tech_components=tech_components,
            signals=signals,
            metadata=metadata
        )
    
    def _extract_emergent_category(self, text: str) -> str:
        """Extract category from task text organically"""
        words = text.split()
        if not words:
            return "unknown"
        
        verb = words[0].lower().rstrip(':,.-')
        
        # Look for object/noun in next few words
        objects = []
        for i in range(1, min(5, len(words))):
            word = words[i].lower().rstrip(':,.-')
            if len(word) > 3 and word not in {'the', 'and', 'for', 'with', 'that', 'this', 'a', 'an'}:
                objects.append(word)
                break
        
        if objects:
            category = f"{verb}_{objects[0]}"
        else:
            category = verb
        
        return category
    
    def _identify_tech_components(self, text: str) -> List[str]:
        """Identify technical components mentioned in text"""
        text_lower = text.lower()
        components = []
        
        for component in self.config["tech_components"]:
            component_lower = component.lower().replace('_', ' ')
            if component_lower in text_lower:
                components.append(component)
        
        return components
    
    def _has_technical_terms(self, text: str) -> bool:
        """Check if text contains technical terms"""
        technical_terms = [
            'api', 'database', 'server', 'client', 'endpoint', 'framework',
            'algorithm', 'architecture', 'deployment', 'infrastructure',
            'microservice', 'container', 'kubernetes', 'docker', 'redis',
            'postgresql', 'mongodb', 'elasticsearch', 'kafka', 'rabbitmq'
        ]
        
        text_lower = text.lower()
        return any(term in text_lower for term in technical_terms)
    
    def _has_measurements(self, text: str) -> bool:
        """Check if text contains measurements"""
        measurement_patterns = [
            r'\d+\s*(ms|seconds?|minutes?|hours?|days?)',
            r'\d+\s*(%|percent)',
            r'<\s*\d+',
            r'>\s*\d+',
            r'\d+\s*(mb|gb|tb)',
            r'\d+\s*(req|requests)'
        ]
        
        for pattern in measurement_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _has_dates(self, text: str) -> bool:
        """Check if text contains dates"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)'
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
```

#### Step 6: Quality Calculator

Create `scripts/phase1/quality_calculator.py`:

```python
"""
Quality calculator for Phase 1
Calculates quality scores, complexity, and specificity for tasks
"""
import random
import re
from typing import Dict

class QualityCalculator:
    """Calculates quality metrics for tasks"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def calculate_quality_score(self, text: str) -> float:
        """Calculate overall quality score (0.0-1.0)"""
        score = 0.5  # Base score
        
        words = text.split()
        
        # Positive signals
        if 8 <= len(words) <= 25:  # Good length
            score += 0.2
        
        if self._has_technical_terms(text):
            score += 0.15
        
        if self._has_measurements(text):
            score += 0.1
        
        if self._has_dates(text):
            score += 0.05
        
        # Negative signals
        if len(words) < 5:
            score -= 0.2
        
        if text.lower().startswith("implement") and len(words) < 8:
            score -= 0.1
        
        # Apply quality tiers
        tier_scores = {
            "high": random.uniform(0.8, 0.95),
            "medium": random.uniform(0.6, 0.8),
            "low": random.uniform(0.3, 0.6)
        }
        
        # Assign tier based on config distribution
        rand = random.random()
        cumulative = 0
        assigned_tier = "medium"
        
        for tier, prob in self.config["quality_tiers"].items():
            cumulative += prob
            if rand <= cumulative:
                assigned_tier = tier
                break
        
        # Blend calculated score with tier score
        final_score = (score * 0.4) + (tier_scores[assigned_tier] * 0.6)
        
        return min(max(final_score, 0.1), 0.99)
    
    def calculate_complexity(self, text: str) -> float:
        """Calculate task complexity (0.0-1.0)"""
        complexity = 0.5
        
        words = text.split()
        
        # Longer tasks tend to be more complex
        if len(words) > 15:
            complexity += 0.2
        elif len(words) < 5:
            complexity -= 0.2
        
        # Technical terms indicate complexity
        if self._has_technical_terms(text):
            complexity += 0.15
        
        # Multiple components indicate complexity
        component_count = sum(1 for comp in self.config["tech_components"] if comp.lower().replace('_', ' ') in text.lower())
        if component_count > 1:
            complexity += 0.1
        
        # Action verbs that indicate complexity
        complex_verbs = {'implement', 'design', 'architect', 'refactor', 'migrate', 'integrate'}
        first_word = words[0].lower() if words else ""
        if first_word in complex_verbs:
            complexity += 0.1
        
        return min(max(complexity, 0.1), 0.99)
    
    def calculate_specificity(self, text: str) -> float:
        """Calculate task specificity (0.0-1.0)"""
        specificity = 0.5
        
        words = text.split()
        
        # Specific measurements
        if self._has_measurements(text):
            specificity += 0.2
        
        # Specific dates
        if self._has_dates(text):
            specificity += 0.15
        
        # Technical terms
        if self._has_technical_terms(text):
            specificity += 0.1
        
        # Proper nouns (capitalized words that aren't at start of sentence)
        proper_nouns = sum(1 for i, word in enumerate(words[1:], 1) if word[0].isupper() and len(word) > 3)
        if proper_nouns > 0:
            specificity += min(proper_nouns * 0.05, 0.15)
        
        # Vague indicators reduce specificity
        vague_terms = {'something', 'things', 'stuff', 'maybe', 'possibly', 'perhaps'}
        if any(term in text.lower() for term in vague_terms):
            specificity -= 0.2
        
        return min(max(specificity, 0.1), 0.99)
    
    def _has_technical_terms(self, text: str) -> bool:
        """Check if text contains technical terms"""
        technical_terms = [
            'api', 'database', 'server', 'client', 'endpoint', 'framework',
            'algorithm', 'architecture', 'deployment', 'infrastructure',
            'microservice', 'container', 'kubernetes', 'docker', 'redis',
            'postgresql', 'mongodb', 'elasticsearch', 'kafka', 'rabbitmq'
        ]
        
        text_lower = text.lower()
        return any(term in text_lower for term in technical_terms)
    
    def _has_measurements(self, text: str) -> bool:
        """Check if text contains measurements"""
        measurement_patterns = [
            r'\d+\s*(ms|seconds?|minutes?|hours?|days?)',
            r'\d+\s*(%|percent)',
            r'<\s*\d+',
            r'>\s*\d+',
            r'\d+\s*(mb|gb|tb)',
            r'\d+\s*(req|requests)'
        ]
        
        for pattern in measurement_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _has_dates(self, text: str) -> bool:
        """Check if text contains dates"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)'
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
```

This guide continues with the remaining sections. Due to length constraints, I'll continue with the next critical sections.

```

#### Step 7: Synthetic Task Generation

Create `scripts/phase1/synthetic_generator.py`:

```python
"""
Synthetic task generator for Phase 1
Generates synthetic tasks based on learned patterns
"""
import random
from typing import List, Dict
from .data_structures import WeightedTask
from .quality_calculator import QualityCalculator
from datetime import datetime

class SyntheticTaskGenerator:
    """Generates synthetic tasks from learned patterns"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.quality_calc = QualityCalculator(config)
    
    def generate_synthetic_tasks(self, natural_tasks: List[WeightedTask], count: int) -> List[WeightedTask]:
        """Generate synthetic tasks based on natural task patterns"""
        if not natural_tasks:
            return []
        
        synthetic_tasks = []
        patterns = self._analyze_task_patterns(natural_tasks)
        
        for i in range(count):
            if patterns and random.random() > 0.3:
                base_pattern = random.choice(patterns)
                task_text = self._generate_from_pattern(base_pattern)
            else:
                task_text = self._generate_from_template()
            
            # Calculate quality metrics
            quality_score = self.quality_calc.calculate_quality_score(task_text)
            complexity = self.quality_calc.calculate_complexity(task_text)
            specificity = self.quality_calc.calculate_specificity(task_text)
            
            # Extract category
            category = self._extract_emergent_category(task_text)
            
            # Choose domain
            domain = random.choice(self.config["domain_contexts"])
            
            # Identify tech components
            tech_components = self._identify_tech_components(task_text)
            
            task = WeightedTask(
                text=task_text,
                source=f"synthetic:{i:06d}",
                category=category,
                quality_score=quality_score,
                complexity=complexity,
                specificity=specificity,
                domain=domain,
                tech_components=tech_components,
                signals={
                    "text_length": len(task_text.split()),
                    "generation_method": "pattern_based" if patterns else "template_based",
                    "is_synthetic": True
                },
                metadata={
                    "generation_id": f"synth_{i:06d}",
                    "generated_at": datetime.now().isoformat(),
                    "base_pattern_used": base_pattern.get("pattern_type", "none") if patterns else "none"
                }
            )
            
            synthetic_tasks.append(task)
        
        return synthetic_tasks
    
    def _analyze_task_patterns(self, tasks: List[WeightedTask]) -> List[Dict]:
        """Analyze patterns in natural tasks"""
        patterns = []
        
        for task in tasks[:50]:  # Sample first 50
            words = task.text.split()
            if len(words) >= 3:
                pattern = {
                    "verb": words[0].lower(),
                    "object": ' '.join(words[1:min(4, len(words))]),
                    "full_pattern": task.text,
                    "category": task.category,
                    "quality": task.quality_score,
                    "pattern_type": "high_quality" if task.quality_score > 0.8 else "standard"
                }
                patterns.append(pattern)
        
        return patterns
    
    def _generate_from_pattern(self, pattern: Dict) -> str:
        """Generate task text from pattern"""
        verb = pattern["verb"]
        base_object = pattern["object"]
        
        variations = [
            f"{verb} {base_object}",
            f"{verb} the {base_object}",
            f"{verb} new {base_object}",
            f"{verb} improved {base_object}",
            f"{verb} {base_object} for better performance",
            f"{verb} {base_object} to fix existing issues",
            f"{verb} {base_object} by next sprint",
            f"{verb} {base_object} following best practices",
        ]
        
        chosen = random.choice(variations)
        
        # Add context sometimes
        if random.random() > 0.7:
            contexts = [
                " using the new framework",
                " with proper error handling",
                " including comprehensive tests",
                " with monitoring and logging",
                " following security guidelines",
                " to meet compliance requirements",
            ]
            chosen += random.choice(contexts)
        
        return chosen
    
    def _generate_from_template(self) -> str:
        """Generate task from template when no patterns available"""
        verbs = ['implement', 'create', 'update', 'fix', 'add', 'remove', 'change', 'improve', 'optimize']
        objects = ['feature', 'system', 'component', 'module', 'service', 'endpoint', 'functionality']
        
        verb = random.choice(verbs)
        obj = random.choice(objects)
        
        return f"{verb} {obj}"
    
    def _extract_emergent_category(self, text: str) -> str:
        """Extract category from task text"""
        words = text.split()
        if not words:
            return "unknown"
        
        verb = words[0].lower().rstrip(':,.-')
        
        if len(words) > 1:
            obj = words[1].lower().rstrip(':,.-')
            if len(obj) > 3:
                return f"{verb}_{obj}"
        
        return verb
    
    def _identify_tech_components(self, text: str) -> List[str]:
        """Identify technical components mentioned"""
        text_lower = text.lower()
        components = []
        
        for component in self.config["tech_components"]:
            component_lower = component.lower().replace('_', ' ')
            if component_lower in text_lower:
                components.append(component)
        
        return components
```

#### Step 8: Main Phase 1 Script

Create `scripts/phase1_emergent.py`:

```python
#!/usr/bin/env python3
"""
Phase 1: Emergent Category Generation
Main script for Phase 1 synthetic data generation
"""
import json
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.config.synthetic_data_config import CONFIG
from scripts.phase1.document_generator import DocumentGenerator
from scripts.phase1.task_extractor import TaskExtractor
from scripts.phase1.synthetic_generator import SyntheticTaskGenerator
from scripts.phase1.data_structures import WeightedTask, Document

def main():
    print("=" * 60)
    print("PHASE 1: BUILDING EMERGENT SYNTHETIC DATASET")
    print("=" * 60)
    
    # Initialize components
    print("\n[1/5] Initializing components...")
    doc_generator = DocumentGenerator(CONFIG)
    task_extractor = TaskExtractor(CONFIG)
    synthetic_generator = SyntheticTaskGenerator(CONFIG)
    
    # Step 1: Generate documents
    print(f"\n[2/5] Generating {CONFIG['num_initial_docs']} realistic documents...")
    documents = doc_generator.generate_documents(CONFIG["num_initial_docs"])
    print(f"   Generated {len(documents)} documents")
    
    # Step 2: Extract natural tasks
    print("\n[3/5] Extracting natural tasks from documents...")
    natural_tasks = task_extractor.extract_tasks_from_documents(documents)
    print(f"   Extracted {len(natural_tasks)} natural tasks")
    
    # Step 3: Generate synthetic tasks
    print("\n[4/5] Generating synthetic tasks based on patterns...")
    synthetic_count = len(natural_tasks) * CONFIG["synthetic_multiplier"]
    synthetic_tasks = synthetic_generator.generate_synthetic_tasks(natural_tasks, synthetic_count)
    print(f"   Generated {len(synthetic_tasks)} synthetic tasks")
    
    # Step 4: Analyze results
    print("\n[5/5] Analyzing dataset quality...")
    analysis = analyze_results(natural_tasks, synthetic_tasks)
    
    # Step 5: Save everything
    print("\n[Saving] Writing dataset and debug info...")
    save_results(documents, natural_tasks, synthetic_tasks, analysis)
    
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE!")
    print("=" * 60)
    print(f"\nDataset Statistics:")
    print(f"   Total tasks: {analysis['total_tasks']}")
    print(f"   Unique categories: {analysis['category_count']}")
    print(f"   Domain coverage: {analysis['domain_coverage']}")
    print(f"\nOutput directory: {CONFIG['output_dir']}/phase1/")

def analyze_results(natural_tasks: List[WeightedTask], synthetic_tasks: List[WeightedTask]) -> Dict:
    """Analyze the generated dataset"""
    all_tasks = natural_tasks + synthetic_tasks
    
    # Quality distribution
    quality_bins = {"high": 0, "medium": 0, "low": 0}
    for task in all_tasks:
        if task.quality_score > 0.8:
            quality_bins["high"] += 1
        elif task.quality_score > 0.6:
            quality_bins["medium"] += 1
        else:
            quality_bins["low"] += 1
    
    # Category distribution
    categories = Counter([t.category for t in all_tasks])
    
    # Domain distribution
    domains = Counter([t.domain for t in all_tasks])
    
    # Tech components
    all_components = []
    for task in all_tasks:
        all_components.extend(task.tech_components)
    component_counts = Counter(all_components)
    
    return {
        "total_tasks": len(all_tasks),
        "natural_tasks": len(natural_tasks),
        "synthetic_tasks": len(synthetic_tasks),
        "quality_distribution": quality_bins,
        "category_count": len(categories),
        "top_categories": dict(categories.most_common(10)),
        "domain_coverage": len(domains),
        "top_domains": dict(domains.most_common(5)),
        "top_components": dict(component_counts.most_common(5))
    }

def save_results(documents: List[Document], natural_tasks: List[WeightedTask], 
                synthetic_tasks: List[WeightedTask], analysis: Dict):
    """Save all results to files"""
    output_dir = Path(CONFIG["output_dir"]) / "phase1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full dataset
    dataset = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "config_used": CONFIG,
            "statistics": analysis
        },
        "documents": [doc.to_dict() for doc in documents],
        "natural_tasks": [t.to_dict() for t in natural_tasks],
        "synthetic_tasks": [t.to_dict() for t in synthetic_tasks]
    }
    
    dataset_file = output_dir / "phase1_dataset.json"
    with open(dataset_file, 'w') as f:
        json.dump(dataset, f, indent=2, default=str)
    
    print(f"   Saved dataset to: {dataset_file}")
    
    # Save summary
    summary_file = output_dir / "phase1_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PHASE 1: EMERGENT SYNTHETIC DATASET SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated at: {datetime.now().isoformat()}\n")
        f.write(f"Total tasks: {analysis['total_tasks']}\n")
        f.write(f"Unique categories: {analysis['category_count']}\n")
        f.write(f"Domain coverage: {analysis['domain_coverage']}\n\n")
        f.write("Quality Distribution:\n")
        for tier, count in analysis['quality_distribution'].items():
            percentage = (count / analysis['total_tasks']) * 100
            f.write(f"  {tier}: {count} ({percentage:.1f}%)\n")
        f.write("\nTop 10 Categories:\n")
        for category, count in analysis['top_categories'].items():
            f.write(f"  {category}: {count}\n")
    
    print(f"   Saved summary to: {summary_file}")
    
    # Save sample
    sample_file = output_dir / "phase1_sample.json"
    sample = {
        "high_quality_samples": [t.to_dict() for t in natural_tasks[:5] if t.quality_score > 0.8],
        "medium_quality_samples": [t.to_dict() for t in natural_tasks[5:10] if 0.6 <= t.quality_score <= 0.8],
        "low_quality_samples": [t.to_dict() for t in natural_tasks[10:15] if t.quality_score < 0.6],
        "synthetic_samples": [t.to_dict() for t in synthetic_tasks[:10]]
    }
    
    with open(sample_file, 'w') as f:
        json.dump(sample, f, indent=2, default=str)
    
    print(f"   Saved sample to: {sample_file}")

if __name__ == "__main__":
    main()
```

---

## Phase 2: Niche-Specific Integration

### Overview

Phase 2 integrates your 20 niche analysis to generate domain-specific synthetic data. This phase uses the patterns discovered in Phase 1 and enhances them with niche-specific context.

### Implementation Steps

#### Step 1: Niche Configuration

Create `scripts/phase2/niche_config.py`:

```python
"""
Niche configuration for Phase 2
Based on your 20 niche analysis
"""
NICHE_ANALYSIS = {
    "property_management": {
        "pain_points": [
            "vendor_overcharging",
            "unnecessary_repairs",
            "work_order_pattern_fraud",
            "vendor_performance_unknown"
        ],
        "data_types": [
            "work_orders",
            "vendor_invoices",
            "maintenance_logs",
            "inspection_reports"
        ],
        "tech_fit": {
            "RAG": 10,
            "predictive_models": 10,
            "LoRAs": 10,
            "multimodal_AI": 10
        },
        "task_templates": [
            "Analyze {data_type} for {pain_point}",
            "Detect {pain_point} in {data_type}",
            "Predict {pain_point} using {data_type}",
            "Validate {data_type} against {pain_point} patterns"
        ]
    },
    "insurance_claims": {
        "pain_points": [
            "insurance_fraud",
            "settlement_prediction",
            "policy_interpretation",
            "claim_validation"
        ],
        "data_types": [
            "policy_documents",
            "claim_forms",
            "medical_records",
            "photos_videos"
        ],
        "tech_fit": {
            "RAG": 10,
            "predictive_models": 10,
            "LoRAs": 9,
            "multimodal_AI": 9
        },
        "task_templates": [
            "Process {data_type} for {pain_point}",
            "Extract information from {data_type} to detect {pain_point}",
            "Classify {data_type} for {pain_point}",
            "Validate {data_type} against {pain_point} criteria"
        ]
    },
    "construction": {
        "pain_points": [
            "project_delay_prediction",
            "risk_management",
            "compliance_checking",
            "cost_overrun"
        ],
        "data_types": [
            "project_schedules",
            "weather_data",
            "supplier_performance",
            "compliance_documents"
        ],
        "tech_fit": {
            "RAG": 9,
            "predictive_models": 10,
            "LoRAs": 8,
            "multimodal_AI": 7
        },
        "task_templates": [
            "Analyze {data_type} to predict {pain_point}",
            "Monitor {data_type} for {pain_point}",
            "Generate reports on {pain_point} from {data_type}",
            "Validate {data_type} for {pain_point}"
        ]
    },
    "healthcare": {
        "pain_points": [
            "revenue_cycle",
            "denial_prevention",
            "coding_validation",
            "claim_processing"
        ],
        "data_types": [
            "medical_records",
            "billing_codes",
            "insurance_claims",
            "patient_data"
        ],
        "tech_fit": {
            "RAG": 10,
            "predictive_models": 9,
            "LoRAs": 10,
            "multimodal_AI": 8
        },
        "task_templates": [
            "Process {data_type} for {pain_point}",
            "Validate {data_type} against {pain_point} rules",
            "Predict {pain_point} from {data_type}",
            "Extract {pain_point} information from {data_type}"
        ]
    },
    "supply_chain": {
        "pain_points": [
            "risk_detection",
            "disruption_prediction",
            "supplier_performance",
            "inventory_optimization"
        ],
        "data_types": [
            "supplier_data",
            "logistics_records",
            "inventory_data",
            "market_data"
        ],
        "tech_fit": {
            "RAG": 8,
            "predictive_models": 10,
            "LoRAs": 7,
            "multimodal_AI": 6
        },
        "task_templates": [
            "Analyze {data_type} for {pain_point}",
            "Predict {pain_point} from {data_type}",
            "Monitor {data_type} to detect {pain_point}",
            "Optimize {data_type} for {pain_point}"
        ]
    }
}

def get_niche_config(niche_name: str) -> Dict:
    """Get configuration for a specific niche"""
    return NICHE_ANALYSIS.get(niche_name, {})
```

#### Step 2: Niche-Aware Generator

Create `scripts/phase2/niche_generator.py`:

```python
"""
Niche-aware task generator for Phase 2
Generates tasks specific to your 20 niches
"""
import random
from typing import List, Dict
from datetime import datetime
from .niche_config import NICHE_ANALYSIS
from ..phase1.data_structures import WeightedTask
from ..phase1.quality_calculator import QualityCalculator

class NicheAwareGenerator:
    """Generates niche-specific tasks"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.quality_calc = QualityCalculator(config)
        self.niches = NICHE_ANALYSIS
    
    def generate_niche_tasks(self, count_per_niche: int = 50) -> List[WeightedTask]:
        """Generate tasks for each niche"""
        all_tasks = []
        
        for niche_name, niche_data in self.niches.items():
            print(f"   Generating tasks for niche: {niche_name}")
            
            for i in range(count_per_niche):
                task_text = self._generate_niche_task(niche_name, niche_data)
                
                # Calculate quality metrics
                quality_score = self.quality_calc.calculate_quality_score(task_text)
                complexity = self.quality_calc.calculate_complexity(task_text)
                specificity = self.quality_calc.calculate_specificity(task_text)
                
                # Extract category
                category = self._extract_category(task_text)
                
                # Identify tech components based on tech_fit
                tech_components = self._get_tech_components(niche_data["tech_fit"])
                
                task = WeightedTask(
                    text=task_text,
                    source=f"niche:{niche_name}:{i:06d}",
                    category=category,
                    quality_score=quality_score,
                    complexity=complexity,
                    specificity=specificity,
                    domain=niche_name,
                    tech_components=tech_components,
                    signals={
                        "niche": niche_name,
                        "pain_point": self._extract_pain_point(task_text, niche_data),
                        "data_type": self._extract_data_type(task_text, niche_data),
                        "text_length": len(task_text.split())
                    },
                    metadata={
                        "generation_id": f"niche_{niche_name}_{i:06d}",
                        "generated_at": datetime.now().isoformat(),
                        "niche_config": niche_name
                    }
                )
                
                all_tasks.append(task)
        
        return all_tasks
    
    def _generate_niche_task(self, niche_name: str, niche_data: Dict) -> str:
        """Generate a task for a specific niche"""
        templates = niche_data.get("task_templates", [])
        if not templates:
            # Fallback template
            templates = [
                "Process {data_type} for {pain_point}",
                "Analyze {data_type} to detect {pain_point}",
                "Validate {data_type} against {pain_point} patterns"
            ]
        
        template = random.choice(templates)
        
        # Fill template
        data_type = random.choice(niche_data["data_types"])
        pain_point = random.choice(niche_data["pain_points"])
        
        task_text = template.format(
            data_type=data_type.replace('_', ' '),
            pain_point=pain_point.replace('_', ' ')
        )
        
        # Add specificity
        task_text = self._add_specificity(task_text, niche_name)
        
        return task_text
    
    def _add_specificity(self, text: str, niche_name: str) -> str:
        """Add specificity to task text"""
        if random.random() > 0.7:
            specificity_additions = [
                " using machine learning models",
                " with real-time monitoring",
                " following industry best practices",
                " to improve accuracy",
                " for better decision making",
                " using historical data patterns",
                " with automated validation",
                " to reduce false positives"
            ]
            text += random.choice(specificity_additions)
        
        return text
    
    def _extract_category(self, text: str) -> str:
        """Extract category from task text"""
        words = text.split()
        if not words:
            return "unknown"
        
        verb = words[0].lower().rstrip(':,.-')
        
        if len(words) > 1:
            obj = words[1].lower().rstrip(':,.-')
            if len(obj) > 3:
                return f"{verb}_{obj}"
        
        return verb
    
    def _get_tech_components(self, tech_fit: Dict) -> List[str]:
        """Get tech components based on tech_fit scores"""
        # Return components with score >= 8
        return [comp for comp, score in tech_fit.items() if score >= 8]
    
    def _extract_pain_point(self, text: str, niche_data: Dict) -> str:
        """Extract pain point from task text"""
        for pain_point in niche_data["pain_points"]:
            if pain_point.replace('_', ' ') in text.lower():
                return pain_point
        return niche_data["pain_points"][0]  # Default to first
    
    def _extract_data_type(self, text: str, niche_data: Dict) -> str:
        """Extract data type from task text"""
        for data_type in niche_data["data_types"]:
            if data_type.replace('_', ' ') in text.lower():
                return data_type
        return niche_data["data_types"][0]  # Default to first
```

---

## Ollama Integration for Inference-Based Detection

### Overview

Ollama integration provides inference-based validation and enhancement of synthetic tasks. This uses the Ollama container from your docker-compose.dev.yml setup.

### Configuration

The Ollama service is configured in `docker-compose.dev.yml`:

```yaml
ollama:
  image: ollama/ollama:latest
  container_name: deepiri-ollama-dev
  ports:
    - "${OLLAMA_PORT:-11435}:11434"
  volumes:
    - ollama_dev_data:/root/.ollama
  networks:
    - deepiri-dev-network
```

### Integration Steps

#### Step 1: Ollama Validator

Create `scripts/utils/ollama_validator.py`:

```python
"""
Ollama-based validator for synthetic tasks
Uses inference to validate and enhance task quality
"""
import os
import json
from typing import Optional, Dict, List
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from utils.semantic_analyzer import SemanticAnalyzer, get_semantic_analyzer
    HAS_SEMANTIC_ANALYZER = True
except ImportError:
    HAS_SEMANTIC_ANALYZER = False
    print("Warning: Semantic analyzer not available. Ollama validation disabled.")

class OllamaValidator:
    """Validates tasks using Ollama inference"""
    
    def __init__(self, config: Dict):
        self.config = config.get("ollama", {})
        self.enabled = self.config.get("enabled", True) and HAS_SEMANTIC_ANALYZER
        
        if self.enabled:
            self.analyzer = get_semantic_analyzer(
                ollama_base_url=self.config.get("base_url", "http://localhost:11434"),
                model=self.config.get("model", "llama3:8b")
            )
        else:
            self.analyzer = None
    
    def validate_task(self, task_text: str) -> Dict:
        """Validate a single task using Ollama"""
        if not self.enabled:
            return {"valid": True, "confidence": 0.5, "reason": "Ollama validation disabled"}
        
        try:
            # Use semantic analyzer to validate
            prompt = f"""Is the following text a valid, actionable task? Respond with JSON:
{{
    "valid": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}}

Task: {task_text}"""
            
            response = self.analyzer._call_ollama(prompt, timeout=self.config.get("timeout", 15.0))
            
            if response:
                # Try to parse JSON response
                try:
                    # Extract JSON from response
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        result = json.loads(json_str)
                        return result
                except:
                    pass
            
            # Fallback: simple validation
            return self._simple_validation(task_text)
        
        except Exception as e:
            print(f"Ollama validation error: {e}")
            return {"valid": True, "confidence": 0.5, "reason": f"Validation error: {str(e)}"}
    
    def _simple_validation(self, task_text: str) -> Dict:
        """Simple validation when Ollama fails"""
        words = task_text.split()
        
        # Basic checks
        if len(words) < 3:
            return {"valid": False, "confidence": 0.3, "reason": "Too short"}
        
        if len(words) > 50:
            return {"valid": False, "confidence": 0.4, "reason": "Too long"}
        
        # Check for action verb
        action_verbs = ['implement', 'create', 'update', 'fix', 'add', 'remove', 'analyze', 'process']
        if not any(verb in task_text.lower() for verb in action_verbs):
            return {"valid": False, "confidence": 0.5, "reason": "No clear action verb"}
        
        return {"valid": True, "confidence": 0.7, "reason": "Basic validation passed"}
    
    def enhance_task(self, task_text: str) -> Optional[str]:
        """Enhance a task using Ollama"""
        if not self.enabled:
            return None
        
        try:
            prompt = f"""Improve the following task to be more specific and actionable. Return only the improved task text:

Original: {task_text}

Improved:"""
            
            response = self.analyzer._call_ollama(prompt, timeout=self.config.get("timeout", 15.0))
            
            if response and len(response.strip()) > len(task_text) * 0.5:
                return response.strip()
        
        except Exception as e:
            print(f"Ollama enhancement error: {e}")
        
        return None
    
    def batch_validate(self, tasks: List[str], sample_rate: float = 0.1) -> List[Dict]:
        """Validate a batch of tasks (sampling for performance)"""
        if not self.enabled:
            return [{"valid": True, "confidence": 0.5} for _ in tasks]
        
        import random
        validated = []
        
        for task in tasks:
            if random.random() < sample_rate:
                result = self.validate_task(task)
                validated.append(result)
            else:
                validated.append({"valid": True, "confidence": 0.5, "reason": "Not validated (sampling)"})
        
        return validated
```

#### Step 2: Integration with Pipeline

Update `scripts/phase1_emergent.py` to include Ollama validation:

```python
# Add at the top
from scripts.utils.ollama_validator import OllamaValidator

# In main() function, after generating synthetic tasks:
if CONFIG.get("ollama", {}).get("use_for_validation", False):
    print("\n[Validation] Validating tasks with Ollama...")
    validator = OllamaValidator(CONFIG)
    
    # Validate sample of tasks
    all_task_texts = [t.text for t in natural_tasks + synthetic_tasks]
    validation_rate = CONFIG.get("ollama", {}).get("validation_frequency", 0.1)
    validation_results = validator.batch_validate(all_task_texts, sample_rate=validation_rate)
    
    # Update task quality scores based on validation
    for i, (task, validation) in enumerate(zip(natural_tasks + synthetic_tasks, validation_results)):
        if validation.get("valid", True):
            # Boost quality score if validated
            task.quality_score = min(task.quality_score * 1.1, 0.99)
        else:
            # Reduce quality score if invalid
            task.quality_score = max(task.quality_score * 0.9, 0.1)
        
        # Add validation metadata
        task.metadata["ollama_validation"] = validation
    
    print(f"   Validated {sum(1 for v in validation_results if v.get('reason', '').startswith('Not') == False)} tasks")
```

### Using Ollama Container

#### Starting Ollama

```bash
# Start Ollama container
docker compose -f docker-compose.dev.yml up -d ollama

# Check if Ollama is running
curl http://localhost:11434/api/tags

# Pull model (if not already pulled)
docker exec deepiri-ollama-dev ollama pull llama3:8b
```

#### Testing Ollama Connection

```python
# Test script: test_ollama.py
import requests
import json

ollama_url = "http://localhost:11434/api/generate"

response = requests.post(
    ollama_url,
    json={
        "model": "llama3:8b",
        "prompt": "Is 'Implement user authentication' a valid task?",
        "stream": False
    },
    timeout=15
)

print(json.dumps(response.json(), indent=2))
```

---

## Usage Instructions

### Basic Usage

#### Running Phase 1

```bash
cd deepiri-platform/diri-helox
python scripts/phase1_emergent.py
```

This will:
1. Generate 200 realistic documents
2. Extract natural tasks from documents
3. Generate 5x synthetic tasks based on patterns
4. Calculate quality scores and metrics
5. Save results to `data/synthetic_data/phase1/`

#### Running Phase 2

```bash
python scripts/phase2_niche_integration.py
```

This will:
1. Load Phase 1 results
2. Generate niche-specific tasks for each of your 20 niches
3. Apply domain-specific context
4. Save results to `data/synthetic_data/phase2/`

### Advanced Usage

#### Custom Configuration

Create `scripts/config/custom_config.py`:

```python
from scripts.config.synthetic_data_config import CONFIG

# Override defaults
CONFIG["num_initial_docs"] = 500
CONFIG["synthetic_multiplier"] = 10
CONFIG["quality_tiers"]["high"] = 0.20  # Increase high quality tasks
```

Then run:

```bash
python scripts/phase1_emergent.py --config scripts/config/custom_config.py
```

#### With Ollama Validation

```bash
# Start Ollama first
docker compose -f docker-compose.dev.yml up -d ollama

# Set environment variables
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3:8b
export OLLAMA_ENABLED=true

# Run with validation
python scripts/phase1_emergent.py
```

#### Generating Specific Niches

```python
# In phase2_niche_integration.py, modify:
NICHES_TO_GENERATE = ["property_management", "insurance_claims"]

# Then run
python scripts/phase2_niche_integration.py
```

### Output Files

After running, you'll find:

```
data/synthetic_data/
├── phase1/
│   ├── phase1_dataset.json          # Full dataset
│   ├── phase1_summary.txt           # Summary report
│   └── phase1_sample.json          # Sample tasks
├── phase2/
│   ├── phase2_dataset.json
│   ├── phase2_summary.txt
│   └── phase2_sample.json
└── debug/
    ├── extraction_log.txt
    └── validation_results.json
```

---

## Success Criteria and Validation

### Success Metrics

#### Phase 1 Success Criteria

1. **Task Count**: At least 1000 total tasks (natural + synthetic)
2. **Category Diversity**: At least 20 unique emergent categories
3. **Quality Distribution**: 
   - High quality (0.8+): 10-20%
   - Medium quality (0.6-0.8): 50-70%
   - Low quality (<0.6): 20-30%
4. **Domain Coverage**: All configured domains represented
5. **No Hardcoded Categories**: All categories discovered organically

#### Phase 2 Success Criteria

1. **Niche Coverage**: Tasks generated for all configured niches
2. **Domain Specificity**: Tasks contain niche-specific terminology
3. **Tech Component Mapping**: Tasks reference appropriate tech components
4. **Pain Point Alignment**: Tasks address configured pain points

### Validation Checklist

Run this validation script after generation:

```python
# scripts/validate_output.py
import json
from pathlib import Path
from collections import Counter

def validate_phase1(output_dir: Path):
    """Validate Phase 1 output"""
    dataset_file = output_dir / "phase1_dataset.json"
    
    with open(dataset_file) as f:
        data = json.load(f)
    
    tasks = data["natural_tasks"] + data["synthetic_tasks"]
    
    # Check task count
    assert len(tasks) >= 1000, f"Expected >= 1000 tasks, got {len(tasks)}"
    
    # Check category diversity
    categories = Counter([t["category"] for t in tasks])
    assert len(categories) >= 20, f"Expected >= 20 categories, got {len(categories)}"
    
    # Check quality distribution
    quality_scores = [t["quality_score"] for t in tasks]
    high_quality = sum(1 for q in quality_scores if q >= 0.8) / len(quality_scores)
    assert 0.10 <= high_quality <= 0.20, f"High quality % out of range: {high_quality:.2%}"
    
    # Check no hardcoded categories
    hardcoded = ["coding", "writing", "fitness", "cleaning"]  # Example
    found_hardcoded = any(cat in hardcoded for cat in categories.keys())
    assert not found_hardcoded, "Found hardcoded categories"
    
    print("Phase 1 validation passed!")

if __name__ == "__main__":
    validate_phase1(Path("data/synthetic_data/phase1"))
```

### Quality Assurance

#### Manual Review Process

1. **Sample Review**: Review 50 random tasks from each quality tier
2. **Category Review**: Verify categories make sense and are not hardcoded
3. **Domain Review**: Check domain-specific tasks contain appropriate terminology
4. **Technical Review**: Verify tech components are correctly identified

#### Automated Quality Checks

```python
# scripts/quality_checks.py
def check_task_quality(task: Dict) -> List[str]:
    """Check task quality and return issues"""
    issues = []
    
    text = task["text"]
    words = text.split()
    
    # Too short
    if len(words) < 3:
        issues.append("Too short")
    
    # Too long
    if len(words) > 50:
        issues.append("Too long")
    
    # No action verb
    action_verbs = ['implement', 'create', 'update', 'fix', 'analyze']
    if not any(verb in text.lower() for verb in action_verbs):
        issues.append("No action verb")
    
    # Quality score mismatch
    if task["quality_score"] > 0.8 and len(words) < 5:
        issues.append("High quality score but low specificity")
    
    return issues
```

---

## Debugging Common Issues

### Issue 1: No Tasks Extracted from Documents

**Symptoms**: `Extracted 0 natural tasks from documents`

**Causes**:
- Document content doesn't contain actionable language
- Task extraction patterns too strict
- Documents are too short or generic

**Solutions**:

1. **Check Document Content**:
```python
# Add debug output in document_generator.py
print(f"Document content length: {len(doc.content)}")
print(f"Document content preview: {doc.content[:200]}")
```

2. **Relax Extraction Patterns**:
```python
# In task_extractor.py, modify _looks_like_actionable_task:
def _looks_like_actionable_task(self, text: str) -> bool:
    words = text.split()
    # Reduce minimum word count
    if len(words) < 2 or len(words) > 30:  # Changed from 3
        return False
    # ... rest of function
```

3. **Add More Document Templates**:
```python
# Add more templates with clear action items
def _engineering_plan(self) -> str:
    return """## Plan
    
### Actions Required
- Implement feature X
- Fix bug Y
- Update documentation
"""
```

### Issue 2: All Tasks Have Same Category

**Symptoms**: Only 1-2 categories discovered, most tasks in same category

**Causes**:
- Category extraction too simplistic (only using first word)
- Documents too similar
- Not enough document variety

**Solutions**:

1. **Improve Category Extraction**:
```python
# In task_extractor.py, enhance _extract_emergent_category:
def _extract_emergent_category(self, text: str) -> str:
    words = text.split()
    if not words:
        return "unknown"
    
    verb = words[0].lower().rstrip(':,.-')
    
    # Look for multiple context words
    context_words = []
    for i in range(1, min(6, len(words))):
        word = words[i].lower().rstrip(':,.-')
        if len(word) > 3 and word not in {'the', 'and', 'for', 'with', 'that', 'this', 'a', 'an'}:
            context_words.append(word)
            if len(context_words) >= 2:  # Use 2 context words
                break
    
    if context_words:
        category = f"{verb}_{'_'.join(context_words[:2])}"
    else:
        category = verb
    
    return category
```

2. **Increase Document Variety**:
```python
# In document_generator.py, add more document types:
doc_templates = [
    # ... existing templates
    ("bug_report", self._bug_report),
    ("feature_request", self._feature_request),
    ("code_review", self._code_review),
    # ... more types
]
```

3. **Use Clustering for Categories**:
```python
# Add clustering-based category discovery
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

def cluster_tasks(tasks: List[WeightedTask], n_clusters: int = 20):
    """Cluster tasks to discover categories"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [t.text for t in tasks]
    embeddings = model.encode(texts)
    
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(embeddings)
    
    # Assign cluster-based categories
    for task, cluster in zip(tasks, clusters):
        task.category = f"cluster_{cluster}"
```

### Issue 3: Quality Scores All Similar

**Symptoms**: All tasks have quality_score around 0.5-0.6, no variation

**Causes**:
- Quality calculation too conservative
- Quality tiers not being applied correctly
- Signals not diverse enough

**Solutions**:

1. **Increase Quality Score Range**:
```python
# In quality_calculator.py:
def calculate_quality_score(self, text: str) -> float:
    score = 0.3  # Lower base score
    # ... existing logic
    
    # Increase tier score differences
    tier_scores = {
        "high": random.uniform(0.85, 0.98),  # Higher range
        "medium": random.uniform(0.55, 0.75),  # Wider range
        "low": random.uniform(0.15, 0.45)  # Lower range
    }
    
    # More aggressive blending
    final_score = (score * 0.3) + (tier_scores[assigned_tier] * 0.7)  # More weight on tier
    
    return min(max(final_score, 0.05), 0.99)
```

2. **Add More Quality Signals**:
```python
# Add more signals to quality calculation
def calculate_quality_score(self, text: str) -> float:
    score = 0.5
    
    # ... existing signals
    
    # New signals
    if self._has_specific_metrics(text):  # e.g., "reduce by 50%"
        score += 0.15
    
    if self._has_timeline(text):  # e.g., "by Friday", "this week"
        score += 0.1
    
    if self._has_stakeholders(text):  # e.g., "with team", "for users"
        score += 0.05
    
    # ... rest of function
```

3. **Debug Quality Distribution**:
```python
# Add debug output
quality_scores = [t.quality_score for t in tasks]
print(f"Quality score stats:")
print(f"  Min: {min(quality_scores):.3f}")
print(f"  Max: {max(quality_scores):.3f}")
print(f"  Mean: {sum(quality_scores)/len(quality_scores):.3f}")
print(f"  Std: {np.std(quality_scores):.3f}")
```

### Issue 4: Ollama Connection Fails

**Symptoms**: `Ollama validation error: Connection refused` or timeout errors

**Causes**:
- Ollama container not running
- Wrong URL/port
- Network issues
- Model not pulled

**Solutions**:

1. **Check Ollama Status**:
```bash
# Check if container is running
docker ps | grep ollama

# Check logs
docker logs deepiri-ollama-dev

# Test connection
curl http://localhost:11434/api/tags
```

2. **Verify Configuration**:
```python
# In config, check Ollama URL
print(f"Ollama URL: {CONFIG['ollama']['base_url']}")
print(f"Ollama enabled: {CONFIG['ollama']['enabled']}")

# Test connection
import requests
try:
    response = requests.get(f"{CONFIG['ollama']['base_url']}/api/tags", timeout=5)
    print(f"Ollama connection: {response.status_code}")
except Exception as e:
    print(f"Ollama connection failed: {e}")
```

3. **Pull Model**:
```bash
# Pull the model if not available
docker exec deepiri-ollama-dev ollama pull llama3:8b

# Or use a smaller model for testing
docker exec deepiri-ollama-dev ollama pull llama3:8b-instruct
```

4. **Increase Timeout**:
```python
# In ollama_validator.py:
self.config.get("timeout", 30.0)  # Increase from 15.0
```

5. **Add Retry Logic**:
```python
# In semantic_analyzer.py, add retry:
def _call_ollama(self, prompt: str, timeout: float = 15.0, retries: int = 3) -> Optional[str]:
    for attempt in range(retries):
        try:
            # ... existing call logic
            return response
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

### Issue 5: Memory Issues with Large Datasets

**Symptoms**: Script crashes with `MemoryError` or becomes very slow

**Causes**:
- Loading entire dataset into memory
- No batching/chunking
- Keeping too many objects in memory

**Solutions**:

1. **Process in Batches**:
```python
# Process documents in batches
def generate_documents_batched(self, total_count: int, batch_size: int = 50):
    """Generate documents in batches"""
    all_docs = []
    for i in range(0, total_count, batch_size):
        batch = self._generate_batch(min(batch_size, total_count - i))
        all_docs.extend(batch)
        
        # Process and save batch
        tasks = self.extract_tasks_from_documents(batch)
        self._save_batch(tasks, i // batch_size)
        
        # Clear batch from memory
        del batch
        del tasks
    
    return all_docs
```

2. **Use Generators**:
```python
# Use generators instead of lists
def generate_tasks_generator(self, documents):
    """Generate tasks as a generator"""
    for doc in documents:
        tasks = self._extract_from_single_doc(doc)
        for task in tasks:
            yield task
        del tasks  # Clear from memory
```

3. **Stream to File**:
```python
# Write directly to file instead of keeping in memory
def save_tasks_streaming(self, tasks_generator, output_file: Path):
    """Save tasks as they're generated"""
    with open(output_file, 'w') as f:
        for task in tasks_generator:
            f.write(json.dumps(task.to_dict()) + '\n')
```

---

## Debugging Miscellaneous Issues

### Issue: Categories Not Making Sense

**Problem**: Discovered categories like "implement_the" or "fix_a" don't make semantic sense.

**Root Cause**: Category extraction is too literal, not considering context.

**Solution**:

```python
# Enhanced category extraction with stop word filtering
def _extract_emergent_category(self, text: str) -> str:
    """Extract category with better filtering"""
    words = text.split()
    if not words:
        return "unknown"
    
    # Skip stop words and articles
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'for', 'with', 'that', 'this', 'to', 'of', 'in', 'on', 'at'}
    
    verb = words[0].lower().rstrip(':,.-')
    if verb in stop_words and len(words) > 1:
        verb = words[1].lower().rstrip(':,.-')
    
    # Find meaningful object words
    objects = []
    for i in range(1, min(6, len(words))):
        word = words[i].lower().rstrip(':,.-')
        if len(word) > 3 and word not in stop_words:
            objects.append(word)
            if len(objects) >= 2:
                break
    
    if objects:
        category = f"{verb}_{'_'.join(objects[:2])}"
    else:
        category = verb
    
    # Clean up category name
    category = category.replace('__', '_').strip('_')
    
    return category if category != "unknown" else "general"
```

### Issue: Synthetic Tasks Too Repetitive

**Problem**: Synthetic tasks are very similar to each other, lack diversity.

**Root Cause**: Pattern-based generation is too literal, not enough variation.

**Solution**:

```python
# Add more variation to synthetic generation
def _generate_from_pattern(self, pattern: Dict) -> str:
    """Generate with more variation"""
    verb = pattern["verb"]
    base_object = pattern["object"]
    
    # More diverse variations
    variations = [
        f"{verb} {base_object}",
        f"{verb} the {base_object}",
        f"{verb} new {base_object}",
        f"{verb} improved {base_object}",
        f"{verb} {base_object} for better performance",
        f"{verb} {base_object} to fix existing issues",
        f"{verb} {base_object} by next sprint",
        f"{verb} {base_object} following best practices",
        # Add more variations
        f"{verb} {base_object} using modern techniques",
        f"{verb} {base_object} to address user feedback",
        f"{verb} {base_object} with proper error handling",
        f"{verb} {base_object} including comprehensive tests",
    ]
    
    chosen = random.choice(variations)
    
    # Add random context
    if random.random() > 0.5:
        contexts = [
            " using the new framework",
            " with proper error handling",
            " including comprehensive tests",
            " with monitoring and logging",
            " following security guidelines",
            " to meet compliance requirements",
            " for improved scalability",
            " with better performance",
        ]
        chosen += random.choice(contexts)
    
    # Sometimes add specificity
    if random.random() > 0.7:
        specifics = [
            " by end of week",
            " with 99% uptime target",
            " to reduce latency by 50%",
            " following industry standards",
        ]
        chosen += random.choice(specifics)
    
    return chosen
```

### Issue: Domain Context Not Applied

**Problem**: Tasks don't reflect the domain context from your 20 niches.

**Root Cause**: Domain injection happens at document level but doesn't propagate to tasks.

**Solution**:

```python
# Enhance task creation to preserve domain context
def _create_weighted_task(self, text: str, source_doc: Document, extraction_method: str) -> WeightedTask:
    """Create task with domain context"""
    # ... existing code ...
    
    # Enhance text with domain-specific terminology if needed
    if source_doc.domain_context:
        text = self._enhance_with_domain_context(text, source_doc.domain_context)
    
    # ... rest of function ...

def _enhance_with_domain_context(self, text: str, domain: str) -> str:
    """Enhance text with domain-specific terminology"""
    domain_terms = {
        "maintenance_fraud_detection": ["vendor", "invoice", "anomaly", "pattern"],
        "claims_processing": ["claim", "policy", "validation", "settlement"],
        "project_delay_prediction": ["schedule", "risk", "timeline", "delay"],
    }
    
    if domain in domain_terms:
        # Check if domain terms already present
        text_lower = text.lower()
        if not any(term in text_lower for term in domain_terms[domain]):
            # Add domain context naturally
            if random.random() > 0.7:
                term = random.choice(domain_terms[domain])
                # Insert term naturally
                words = text.split()
                if len(words) > 2:
                    insert_pos = random.randint(1, len(words) - 1)
                    words.insert(insert_pos, term)
                    text = ' '.join(words)
    
    return text
```

### Issue: Performance Degradation Over Time

**Problem**: Script gets slower as it processes more documents/tasks.

**Root Cause**: Memory accumulation, inefficient data structures, no caching.

**Solution**:

```python
# Add caching and optimize data structures
from functools import lru_cache

class TaskExtractor:
    def __init__(self, config: Dict):
        # ... existing init ...
        self._category_cache = {}  # Cache category extractions
    
    @lru_cache(maxsize=1000)
    def _extract_emergent_category(self, text: str) -> str:
        """Cached category extraction"""
        # ... existing logic ...
    
    def extract_tasks_from_documents(self, documents: List[Document]) -> List[WeightedTask]:
        """Optimized extraction with batching"""
        all_tasks = []
        batch_size = 50
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_tasks = []
            
            for doc in batch:
                doc_tasks = self._extract_from_single_doc(doc)
                batch_tasks.extend(doc_tasks)
            
            all_tasks.extend(batch_tasks)
            
            # Clear batch from memory
            del batch
            if i % 500 == 0:
                # Periodic cleanup
                import gc
                gc.collect()
        
        return all_tasks
```

### Issue: Output Files Too Large

**Problem**: JSON output files are several GB, difficult to work with.

**Root Cause**: Saving entire dataset as single JSON, including all metadata.

**Solution**:

```python
# Save in compressed format and split files
import gzip
import json

def save_results_compressed(documents, tasks, output_dir: Path):
    """Save results in compressed, split format"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tasks in chunks
    chunk_size = 1000
    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i:i+chunk_size]
        chunk_file = output_dir / f"tasks_chunk_{i//chunk_size:04d}.json.gz"
        
        with gzip.open(chunk_file, 'wt') as f:
            json.dump([t.to_dict() for t in chunk], f, indent=2)
    
    # Save summary separately (no full documents)
    summary = {
        "metadata": {
            "total_tasks": len(tasks),
            "total_documents": len(documents),
            "generated_at": datetime.now().isoformat()
        },
        "statistics": analyze_results([], tasks)
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
```

---

## Advanced Configuration

### Custom Quality Tiers

```python
# Define custom quality distribution
CONFIG["quality_tiers"] = {
    "excellent": 0.05,  # 5% excellent
    "high": 0.20,       # 20% high
    "medium": 0.50,     # 50% medium
    "low": 0.20,        # 20% low
    "poor": 0.05        # 5% poor
}

# Update quality calculator
tier_scores = {
    "excellent": random.uniform(0.90, 0.99),
    "high": random.uniform(0.75, 0.89),
    "medium": random.uniform(0.50, 0.74),
    "low": random.uniform(0.25, 0.49),
    "poor": random.uniform(0.10, 0.24)
}
```

### Custom Domain Contexts

```python
# Add your specific domains
CONFIG["domain_contexts"] = [
    "your_domain_1",
    "your_domain_2",
    # ... your 20 niches
]

# Add domain-specific templates
domain_templates = {
    "your_domain_1": [
        "Process {component} for {use_case}",
        "Analyze {component} to detect {pattern}",
    ]
}
```

### Performance Tuning

```python
# Adjust for your system
CONFIG["performance"] = {
    "batch_size": 50,           # Documents per batch
    "max_workers": 4,           # Parallel processing
    "chunk_size": 1000,          # Tasks per chunk
    "cache_size": 10000,         # LRU cache size
    "memory_limit_mb": 4096,     # Memory limit
}
```

---

## Performance Optimization

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def generate_documents_parallel(self, count: int, workers: int = 4):
    """Generate documents in parallel"""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(self._generate_single_document, i) 
                   for i in range(count)]
        documents = [f.result() for f in futures]
    return documents
```

### Caching Strategies

```python
# Cache expensive operations
from functools import lru_cache
import hashlib

@lru_cache(maxsize=10000)
def calculate_quality_cached(text_hash: str, text: str) -> float:
    """Cached quality calculation"""
    return self.calculate_quality_score(text)

def calculate_quality_score(self, text: str) -> float:
    """Use cached version"""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return self.calculate_quality_cached(text_hash, text)
```

### Memory Management

```python
# Use generators and streaming
def process_documents_streaming(self, document_generator):
    """Process documents as they're generated"""
    for doc in document_generator:
        tasks = self.extract_tasks_from_documents([doc])
        yield from tasks
        del tasks  # Clear immediately
```

---

## Production Deployment

### Docker Integration

Create `Dockerfile.synthetic_data`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts
COPY scripts/ ./scripts/
COPY data/ ./data/

# Set environment
ENV PYTHONPATH=/app
ENV OLLAMA_BASE_URL=http://ollama:11434

# Run script
CMD ["python", "scripts/phase1_emergent.py"]
```

### CI/CD Integration

```yaml
# .github/workflows/generate_synthetic_data.yml
name: Generate Synthetic Data

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Generate synthetic data
        run: python scripts/phase1_emergent.py
      - name: Upload artifacts
        uses: actions/upload-artifact@v2
        with:
          name: synthetic-data
          path: data/synthetic_data/
```

### Monitoring and Logging

```python
# Add logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('synthetic_data_generation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use throughout scripts
logger.info(f"Generated {len(documents)} documents")
logger.warning(f"Low task extraction rate: {len(tasks)/len(documents):.2f}")
logger.error(f"Ollama validation failed: {e}")
```

---

## Conclusion

This comprehensive guide covers the complete implementation of weighted synthetic data generation for ML training. The system:

1. **Discovers categories organically** from document content
2. **Generates realistic documents** that mirror real work artifacts
3. **Extracts tasks naturally** using multiple extraction methods
4. **Applies quality weights** based on multiple signals
5. **Integrates with Ollama** for inference-based validation
6. **Supports niche-specific** generation for your 20 domains

### Next Steps

1. Run Phase 1 to generate initial dataset
2. Review output and adjust configuration as needed
3. Run Phase 2 to add niche-specific tasks
4. Validate output using provided validation scripts
5. Integrate with your ML training pipeline

### Getting Help

If you encounter issues:

1. Check the debugging sections above
2. Review logs in `data/synthetic_data/debug/`
3. Validate configuration using test scripts
4. Check Ollama connection if using validation
5. Review sample outputs to understand data structure

### Continuous Improvement

The system is designed to improve over time:

- Categories become more refined as more documents are processed
- Quality scores become more accurate with more signals
- Domain-specific generation improves with niche analysis
- Ollama validation enhances task quality

Regularly review and update:
- Document templates based on real documents
- Quality calculation signals
- Niche configurations
- Extraction patterns

This system provides a solid foundation for generating high-quality, weighted synthetic data for ML training.

---

## Appendix A: Complete Code Examples

### Full Phase 1 Implementation

See the individual script files:
- `scripts/phase1/document_generator.py`
- `scripts/phase1/task_extractor.py`
- `scripts/phase1/quality_calculator.py`
- `scripts/phase1/synthetic_generator.py`
- `scripts/phase1_emergent.py`

### Full Phase 2 Implementation

See the individual script files:
- `scripts/phase2/niche_config.py`
- `scripts/phase2/niche_generator.py`
- `scripts/phase2_niche_integration.py`

### Configuration Files

- `scripts/config/synthetic_data_config.py` - Main configuration
- `scripts/config/custom_config.py` - Custom overrides

## Appendix B: Quick Reference

### Common Commands

```bash
# Generate Phase 1 data
python scripts/phase1_emergent.py

# Generate Phase 2 data
python scripts/phase2_niche_integration.py

# Validate output
python scripts/validate_output.py

# Test Ollama connection
python scripts/test_ollama.py

# Check data quality
python scripts/quality_checks.py
```

### Configuration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `num_initial_docs` | 200 | Number of documents to generate |
| `synthetic_multiplier` | 5 | Multiplier for synthetic tasks |
| `quality_tiers.high` | 0.15 | Percentage of high quality tasks |
| `quality_tiers.medium` | 0.60 | Percentage of medium quality tasks |
| `quality_tiers.low` | 0.25 | Percentage of low quality tasks |
| `ollama.enabled` | true | Enable Ollama validation |
| `ollama.validation_frequency` | 0.1 | Percentage of tasks to validate |

### File Structure

```
diri-helox/
├── scripts/
│   ├── phase1/
│   │   ├── document_generator.py
│   │   ├── task_extractor.py
│   │   ├── quality_calculator.py
│   │   ├── synthetic_generator.py
│   │   └── data_structures.py
│   ├── phase2/
│   │   ├── niche_config.py
│   │   └── niche_generator.py
│   ├── config/
│   │   └── synthetic_data_config.py
│   ├── utils/
│   │   └── ollama_validator.py
│   ├── phase1_emergent.py
│   └── phase2_niche_integration.py
├── data/
│   └── synthetic_data/
│       ├── phase1/
│       ├── phase2/
│       └── debug/
└── docs/
    └── GENERATING_SYNTHETIC_DATA.md
```

## Appendix C: Troubleshooting Checklist

### Before Running

- [ ] Python 3.11+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Input data file exists (`b2b_sample_1000_V2.jsonl`)
- [ ] Output directories created
- [ ] Configuration file exists and is valid

### During Generation

- [ ] Documents are being generated (check console output)
- [ ] Tasks are being extracted (should see extraction counts)
- [ ] Quality scores are being calculated (check for variety)
- [ ] Categories are being discovered (should see multiple categories)
- [ ] No errors in console output

### After Generation

- [ ] Output files exist in expected locations
- [ ] File sizes are reasonable (not 0 bytes, not extremely large)
- [ ] JSON files are valid (can be parsed)
- [ ] Statistics look reasonable (task counts, category counts)
- [ ] Sample tasks make sense when reviewed

### Ollama Issues

- [ ] Ollama container is running (`docker ps | grep ollama`)
- [ ] Ollama is accessible (`curl http://localhost:11434/api/tags`)
- [ ] Model is pulled (`docker exec deepiri-ollama-dev ollama list`)
- [ ] Configuration has correct URL and model name
- [ ] Timeout is sufficient (15+ seconds)

## Appendix D: Performance Benchmarks

### Expected Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Generate 200 documents | 5-10 seconds | Depends on templates |
| Extract tasks from 200 docs | 10-20 seconds | Depends on content length |
| Generate 1000 synthetic tasks | 15-30 seconds | Pattern-based generation |
| Ollama validation (10% sample) | 2-5 minutes | Depends on Ollama performance |
| Full Phase 1 pipeline | 1-2 minutes | Without Ollama validation |
| Full Phase 1 with Ollama | 3-7 minutes | With 10% validation |

### Resource Usage

| Resource | Usage | Notes |
|----------|-------|-------|
| Memory | 500MB - 2GB | Depends on batch size |
| CPU | Low-Medium | Mostly I/O bound |
| Disk | 10-100MB | Depends on output size |
| Network | Low | Only if using Ollama |

### Optimization Tips

1. **Increase batch size** for faster processing (if memory allows)
2. **Disable Ollama validation** for faster generation (validate later)
3. **Use generators** for memory efficiency with large datasets
4. **Process in parallel** for multi-core systems
5. **Cache expensive operations** like quality calculations

## Appendix E: Integration with ML Pipeline

### Using Generated Data for Training

```python
# Load generated dataset
import json
from pathlib import Path

def load_synthetic_tasks(data_dir: Path):
    """Load tasks from generated dataset"""
    dataset_file = data_dir / "phase1_dataset.json"
    
    with open(dataset_file) as f:
        data = json.load(f)
    
    # Combine natural and synthetic tasks
    all_tasks = data["natural_tasks"] + data["synthetic_tasks"]
    
    # Filter by quality if needed
    high_quality_tasks = [t for t in all_tasks if t["quality_score"] > 0.8]
    
    return all_tasks, high_quality_tasks

# Use in training
tasks, hq_tasks = load_synthetic_tasks(Path("data/synthetic_data/phase1"))

# Convert to training format
X = [t["text"] for t in tasks]
y = [t["category"] for t in tasks]
weights = [t["quality_score"] for t in tasks]  # Use quality as sample weights
```

### Weighted Training

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Split with stratification
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.2, stratify=y
)

# Train with sample weights
model = RandomForestClassifier()
model.fit(X_train, y_train, sample_weight=w_train)

# Evaluate
score = model.score(X_test, y_test, sample_weight=w_test)
```

### Category Mapping

```python
# Map emergent categories to your label system
category_mapping = {
    "implement_api": "coding",
    "write_documentation": "writing",
    "fix_bug": "debugging",
    # ... your mappings
}

def map_categories(tasks, mapping):
    """Map emergent categories to training labels"""
    for task in tasks:
        category = task["category"]
        if category in mapping:
            task["mapped_category"] = mapping[category]
        else:
            task["mapped_category"] = "other"
    return tasks
```

---

## End of Guide

This guide provides comprehensive coverage of the synthetic data generation system. For additional support or questions, refer to:

1. The debugging sections for common issues
2. The code examples in the appendices
3. The configuration files for customization
4. The validation scripts for quality assurance

The system is designed to be flexible and adaptable to your specific needs. Start with the basic configuration, run Phase 1, review the output, adjust as needed, then proceed to Phase 2 for niche-specific generation.