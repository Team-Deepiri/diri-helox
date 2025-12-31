# Universal RAG System - Complete Guide

## Overview

The Universal RAG (Retrieval-Augmented Generation) system is a reusable component that works across **all industry niches** in the Deepiri platform.

### Supported Industries

- **Insurance** - Claims, policies, regulations
- **Manufacturing** - Equipment manuals, maintenance logs, safety procedures
- **Property Management** - Maintenance requests, tenant records, building codes
- **Healthcare** - Medical records, compliance documents, protocols
- **Construction** - Project specs, safety guidelines, inspection reports
- **Automotive** - Service records, repair manuals, warranty claims
- **And more...**

### Core Capabilities

1. **Document Indexing** - Regulations, policies, manuals, contracts
2. **Historical Data Retrieval** - Work orders, claims, maintenance logs
3. **Knowledge Base Queries** - FAQs, repair guides, compliance advice
4. **Natural Language Search** - "Is this repair necessary?", "Is this covered?"

---

## Architecture

### Components

```
deepiri-modelkit/rag/              # Shared RAG abstractions
├── base.py                        # Base classes (UniversalRAGEngine, Document)
├── processors.py                  # Document processors
└── retrievers.py                  # Retrieval strategies

diri-cyrex/app/integrations/       # Production implementation
└── universal_rag_engine.py        # Milvus-based RAG engine

diri-cyrex/app/routes/             # API endpoints
└── universal_rag_api.py           # REST API
```

### Data Flow

```
1. Raw Documents → Document Processors → Structured Documents
2. Structured Documents → Embedding Model → Vector Embeddings
3. Vector Embeddings → Milvus Vector DB → Indexed
4. User Query → Retrieval → Relevant Documents
5. Relevant Documents + Query → LLM → Generated Response
```

---

## Quick Start

### 1. Index Documents

**Example: Index a manufacturing equipment manual**

```python
from deepiri_modelkit.rag import Document, DocumentType, IndustryNiche
from diri_cyrex.app.integrations.universal_rag_engine import create_universal_rag_engine

# Create RAG engine for manufacturing
engine = create_universal_rag_engine(industry=IndustryNiche.MANUFACTURING)

# Index a manual
document = Document(
    id="manual_compressor_xyz_2024",
    content="Compressor Model XYZ Maintenance Guide...",
    doc_type=DocumentType.MANUAL,
    industry=IndustryNiche.MANUFACTURING,
    title="Compressor Model XYZ Manual",
    source="manufacturer_website",
    metadata={
        "equipment_model": "XYZ-500",
        "manufacturer": "AcmeCorp",
        "version": "2024.1"
    }
)

engine.index_document(document)
```

### 2. Search Documents

```python
from deepiri_modelkit.rag import RAGQuery

# Search for compressor maintenance info
query = RAGQuery(
    query="How do I replace the belt on compressor model XYZ?",
    industry=IndustryNiche.MANUFACTURING,
    doc_types=[DocumentType.MANUAL, DocumentType.PROCEDURE],
    top_k=5
)

results = engine.retrieve(query)

for result in results:
    print(f"Document: {result.document.title}")
    print(f"Score: {result.score}")
    print(f"Content: {result.document.content[:200]}...")
```

### 3. Generate Answer with RAG

```python
# Generate answer using retrieved context
generation = engine.generate_with_context(
    query="How do I replace the belt on compressor model XYZ?",
    retrieved_docs=results
)

print(generation["prompt"])  # Full prompt with context
print(generation["context"])  # Retrieved context
```

---

## API Usage

### Index a Document

**POST** `/api/v1/universal-rag/index`

```bash
curl -X POST http://localhost:8000/api/v1/universal-rag/index \
  -H "Content-Type: application/json" \
  -d '{
    "id": "reg_osha_2024_001",
    "content": "OSHA requires all manufacturing facilities to...",
    "doc_type": "regulation",
    "industry": "manufacturing",
    "title": "OSHA Safety Standards 2024",
    "source": "osha.gov",
    "metadata": {
      "regulation_number": "29 CFR 1910",
      "effective_date": "2024-01-01"
    }
  }'
```

### Search Documents

**POST** `/api/v1/universal-rag/search`

```bash
curl -X POST http://localhost:8000/api/v1/universal-rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are fire safety requirements for manufacturing?",
    "industry": "manufacturing",
    "doc_types": ["regulation", "safety_guideline"],
    "top_k": 5
  }'
```

### Generate with RAG

**POST** `/api/v1/universal-rag/generate`

```bash
curl -X POST http://localhost:8000/api/v1/universal-rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I file a property damage claim?",
    "industry": "insurance",
    "doc_types": ["policy", "faq"],
    "top_k": 3
  }'
```

---

## Industry-Specific Use Cases

### Insurance

**Documents:**
- Policies (auto, home, commercial)
- Claim records
- Regulations (state insurance laws)
- FAQs

**Queries:**
- "Is water damage covered under policy ABC123?"
- "What documents do I need to file a claim?"
- "What are the coverage limits for hurricane damage?"

**Example:**

```python
# Create insurance RAG engine
engine = create_universal_rag_engine(
    industry=IndustryNiche.INSURANCE,
    collection_name="deepiri_insurance_rag"
)

# Index insurance policy
policy = Document(
    id="policy_homeowners_2024",
    content="Homeowners Insurance Policy... Water damage coverage...",
    doc_type=DocumentType.POLICY,
    industry=IndustryNiche.INSURANCE,
    title="Standard Homeowners Policy",
    metadata={
        "policy_type": "homeowners",
        "coverage_limits": {"water_damage": 50000}
    }
)
engine.index_document(policy)

# Query
results = engine.search(
    query="Is water damage covered?",
    industry=IndustryNiche.INSURANCE,
    doc_types=[DocumentType.POLICY]
)
```

### Manufacturing

**Documents:**
- Equipment manuals
- Maintenance logs
- Safety procedures
- Work orders

**Queries:**
- "How do I calibrate machine XYZ?"
- "What maintenance is due this month?"
- "What are the safety requirements for welding?"

**Example:**

```python
# Create manufacturing RAG engine
engine = create_universal_rag_engine(
    industry=IndustryNiche.MANUFACTURING
)

# Index maintenance log
log = Document(
    id="maint_log_machine_123_2024_03",
    content="Machine 123 maintenance completed. Replaced bearings...",
    doc_type=DocumentType.MAINTENANCE_LOG,
    industry=IndustryNiche.MANUFACTURING,
    title="Machine 123 March 2024 Maintenance",
    metadata={
        "machine_id": "123",
        "maintenance_type": "preventive",
        "technician": "John Doe"
    }
)
engine.index_document(log)

# Query historical maintenance
results = engine.search(
    query="When was the last bearing replacement on machine 123?",
    industry=IndustryNiche.MANUFACTURING,
    doc_types=[DocumentType.MAINTENANCE_LOG, DocumentType.WORK_ORDER]
)
```

### Property Management

**Documents:**
- Building codes
- Maintenance requests
- Inspection reports
- Tenant records

**Queries:**
- "What's the process for handling plumbing emergencies?"
- "What are the fire safety requirements for this building?"
- "Show me recent HVAC maintenance history"

### Healthcare

**Documents:**
- Medical protocols
- Compliance regulations (HIPAA, etc.)
- Equipment manuals
- Safety guidelines

**Queries:**
- "What's the protocol for administering medication X?"
- "What are the HIPAA requirements for patient records?"

---

## Document Processing

### Automatic Chunking

Documents are automatically chunked for optimal retrieval:

```python
from deepiri_modelkit.rag.processors import get_processor

# Get processor for document type
processor = get_processor(
    doc_type=DocumentType.REGULATION,
    chunk_size=1000,
    chunk_overlap=200
)

# Process document (auto-chunks and extracts metadata)
chunks = processor.process(
    raw_content=long_regulation_text,
    metadata={"source": "osha.gov", "regulation_number": "29 CFR 1910"}
)

# Each chunk is a separate document
for chunk in chunks:
    engine.index_document(chunk)
```

### Supported Document Types

| Type | Description | Example Industries |
|------|-------------|-------------------|
| `REGULATION` | Laws, regulations, compliance | All industries |
| `POLICY` | Insurance policies, company policies | Insurance, HR |
| `MANUAL` | Equipment manuals, operation guides | Manufacturing, Healthcare |
| `CONTRACT` | Legal contracts, agreements | Legal, Real Estate |
| `WORK_ORDER` | Service requests, repair orders | Manufacturing, Property Mgmt |
| `CLAIM_RECORD` | Insurance claims, warranty claims | Insurance, Automotive |
| `MAINTENANCE_LOG` | Equipment maintenance history | Manufacturing, Facilities |
| `FAQ` | Frequently asked questions | All industries |
| `KNOWLEDGE_BASE` | General knowledge articles | All industries |
| `PROCEDURE` | Standard operating procedures | Manufacturing, Healthcare |
| `SAFETY_GUIDELINE` | Safety protocols | Manufacturing, Construction |

---

## Advanced Features

### Metadata Filtering

Filter by custom metadata:

```python
query = RAGQuery(
    query="Recent compressor failures",
    industry=IndustryNiche.MANUFACTURING,
    doc_types=[DocumentType.MAINTENANCE_LOG],
    metadata_filters={
        "equipment_model": "XYZ-500",
        "status": "failure"
    },
    top_k=10
)
```

### Date Range Filtering

Filter by date:

```python
from datetime import datetime, timedelta

# Get documents from last 30 days
start_date = datetime.now() - timedelta(days=30)
end_date = datetime.now()

query = RAGQuery(
    query="Recent maintenance issues",
    date_range=(start_date, end_date),
    top_k=5
)
```

### Hybrid Retrieval

Combines semantic search + keyword search:

```python
from deepiri_modelkit.rag.retrievers import HybridRetriever

# Configure hybrid retrieval
config = RAGConfig(
    industry=IndustryNiche.MANUFACTURING,
    use_reranking=True,  # Enable cross-encoder reranking
    top_k=10
)

engine = UniversalRAGEngine(config)
```

---

## Configuration

### RAGConfig Options

```python
from deepiri_modelkit.rag import RAGConfig, IndustryNiche

config = RAGConfig(
    # Industry
    industry=IndustryNiche.MANUFACTURING,
    
    # Vector Database
    vector_db_type="milvus",
    collection_name="deepiri_manufacturing_rag",
    vector_db_host="milvus",
    vector_db_port=19530,
    
    # Embeddings
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_dimension=384,
    
    # Retrieval
    top_k=5,
    similarity_threshold=0.7,
    use_reranking=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    
    # Chunking
    chunk_size=1000,
    chunk_overlap=200,
    
    # Features
    enable_metadata_filtering=True,
    date_range_filtering=True,
)
```

---

## Best Practices

### 1. Use Appropriate Document Types

Choose the right document type for better filtering:

```python
# Good: Specific type
Document(doc_type=DocumentType.MAINTENANCE_LOG, ...)

# Less optimal: Generic type
Document(doc_type=DocumentType.OTHER, ...)
```

### 2. Add Rich Metadata

More metadata = better filtering:

```python
Document(
    content="...",
    metadata={
        "equipment_model": "XYZ-500",
        "manufacturer": "AcmeCorp",
        "department": "Production",
        "priority": "high",
        "technician": "John Doe"
    }
)
```

### 3. Use Industry-Specific Collections

Separate collections per industry for better performance:

```python
# Good: Separate collections
insurance_engine = create_universal_rag_engine(industry=IndustryNiche.INSURANCE)
manufacturing_engine = create_universal_rag_engine(industry=IndustryNiche.MANUFACTURING)

# Less optimal: Mix everything
generic_engine = create_universal_rag_engine(industry=IndustryNiche.GENERIC)
```

### 4. Batch Index for Performance

Use batch indexing for large datasets:

```python
# Good: Batch
engine.index_documents(documents)  # 1000 documents at once

# Less optimal: One-by-one
for doc in documents:
    engine.index_document(doc)  # 1000 separate calls
```

---

## Troubleshooting

### No Results Returned

**Cause:** Similarity threshold too high

**Solution:** Lower threshold or check embeddings

```python
config = RAGConfig(
    similarity_threshold=0.5,  # Lower threshold
    use_reranking=True  # Helps improve relevance
)
```

### Slow Retrieval

**Cause:** Too many documents, inefficient indexing

**Solution:** Use metadata filtering, separate collections

```python
# Filter before searching
query = RAGQuery(
    query="...",
    metadata_filters={"department": "production"}  # Narrow search space
)
```

### Poor Quality Results

**Cause:** Query-document mismatch

**Solution:** Use hybrid retrieval, reranking

```python
config = RAGConfig(
    use_reranking=True,  # Cross-encoder reranking
)
```

---

## Next Steps

1. **Deploy to Production** - Scale with Kubernetes
2. **Add Custom Processors** - Industry-specific document processing
3. **Integrate with LLM** - Connect to GPT-4, Claude, or local models
4. **Monitor Performance** - Track retrieval accuracy, latency

---

## Support

For questions or issues:
- Check logs: `diri-cyrex/app/logs/`
- API documentation: `http://localhost:8000/docs`
- System health: `GET /api/v1/universal-rag/health`

