## Universal RAG - Practical Examples

Complete, runnable examples for common RAG use cases across industries.

---

## Example 1: Insurance Claims Processing

**Scenario:** Index insurance policies and claim FAQs, then query coverage questions

```python
from deepiri_modelkit.rag import (
    Document, DocumentType, IndustryNiche, RAGQuery
)
from diri_cyrex.app.integrations.universal_rag_engine import create_universal_rag_engine

# Create insurance RAG engine
engine = create_universal_rag_engine(
    industry=IndustryNiche.INSURANCE,
    collection_name="deepiri_insurance_rag"
)

# Index homeowners policy
homeowners_policy = Document(
    id="policy_homeowners_standard_2024",
    content="""
    Homeowners Insurance Policy - Coverage Details
    
    Water Damage: Covered up to $50,000 for sudden and accidental water damage
    including burst pipes, appliance malfunctions, and roof leaks.
    
    Exclusions: Flood damage is NOT covered under standard homeowners insurance.
    Requires separate flood insurance policy.
    
    Deductible: $1,000 per claim
    
    Filing Process: Contact claims department within 24 hours of damage.
    Document damage with photos. Obtain repair estimates.
    """,
    doc_type=DocumentType.POLICY,
    industry=IndustryNiche.INSURANCE,
    title="Standard Homeowners Policy - Water Damage Coverage",
    source="policy_documents/homeowners_2024.pdf",
    metadata={
        "policy_type": "homeowners",
        "coverage_category": "water_damage",
        "coverage_limit": 50000,
        "deductible": 1000
    }
)

engine.index_document(homeowners_policy)

# Index FAQ
faq = Document(
    id="faq_water_damage_claims",
    content="""
    Q: How do I file a water damage claim?
    A: Contact our claims department at 1-800-CLAIMS within 24 hours. 
    Have your policy number ready. Document the damage with photos and videos.
    Obtain at least two repair estimates from licensed contractors.
    
    Q: What's not covered under water damage?
    A: Flood damage, gradual leaks, damage from lack of maintenance.
    
    Q: How long does claim processing take?
    A: Most claims are processed within 7-10 business days after receiving
    all required documentation.
    """,
    doc_type=DocumentType.FAQ,
    industry=IndustryNiche.INSURANCE,
    title="Water Damage Claims FAQ",
    source="internal_faq",
    metadata={
        "category": "claims",
        "subcategory": "water_damage"
    }
)

engine.index_document(faq)

# Query 1: Coverage question
query1 = RAGQuery(
    query="Is water damage from a burst pipe covered?",
    industry=IndustryNiche.INSURANCE,
    doc_types=[DocumentType.POLICY],
    top_k=3
)

results1 = engine.retrieve(query1)
print("=== Query: Is water damage from a burst pipe covered? ===")
for result in results1:
    print(f"\nDocument: {result.document.title}")
    print(f"Score: {result.score:.3f}")
    print(f"Answer: {result.document.content[:300]}...")

# Query 2: Process question
query2 = RAGQuery(
    query="How do I file a water damage claim?",
    industry=IndustryNiche.INSURANCE,
    doc_types=[DocumentType.FAQ, DocumentType.POLICY],
    top_k=3
)

results2 = engine.retrieve(query2)
print("\n\n=== Query: How do I file a water damage claim? ===")
for result in results2:
    print(f"\nDocument: {result.document.title}")
    print(f"Answer: {result.document.content[:300]}...")

# Generate comprehensive answer
generation = engine.generate_with_context(
    query="Is flood damage covered under homeowners insurance?",
    retrieved_docs=engine.retrieve(RAGQuery(
        query="flood damage coverage homeowners insurance",
        industry=IndustryNiche.INSURANCE
    ))
)

print("\n\n=== Generated Answer ===")
print(generation["prompt"])
```

---

## Example 2: Manufacturing Equipment Maintenance

**Scenario:** Index equipment manuals and maintenance logs for troubleshooting

```python
from datetime import datetime

# Create manufacturing RAG engine
engine = create_universal_rag_engine(
    industry=IndustryNiche.MANUFACTURING
)

# Index equipment manual
compressor_manual = Document(
    id="manual_compressor_xyz_500_2024",
    content="""
    Compressor Model XYZ-500 Maintenance Manual
    
    Belt Replacement Procedure:
    1. Power off compressor and disconnect from power source
    2. Remove safety guard (4 bolts)
    3. Loosen tensioner pulley
    4. Remove old belt
    5. Install new belt (Part #: BLT-XYZ-500)
    6. Adjust tension to 50-60 lbs
    7. Replace safety guard
    8. Test operation
    
    Belt Replacement Frequency: Every 2000 operating hours or 12 months
    
    Troubleshooting:
    - Squealing noise: Belt too loose, adjust tension
    - Excessive vibration: Belt misaligned or worn, replace immediately
    """,
    doc_type=DocumentType.MANUAL,
    industry=IndustryNiche.MANUFACTURING,
    title="Compressor XYZ-500 Manual - Belt Replacement",
    source="manufacturer/acmecorp",
    version="2024.1",
    metadata={
        "equipment_model": "XYZ-500",
        "manufacturer": "AcmeCorp",
        "manual_section": "maintenance",
        "procedure": "belt_replacement"
    }
)

engine.index_document(compressor_manual)

# Index maintenance logs
maintenance_log1 = Document(
    id="maint_log_xyz500_001_2024_01",
    content="""
    Date: 2024-01-15
    Equipment: Compressor XYZ-500 (Serial: 123456)
    Maintenance Type: Preventive
    Technician: John Doe
    
    Work Performed:
    - Replaced compressor belt (Part #: BLT-XYZ-500)
    - Adjusted belt tension to 55 lbs
    - Checked alignment - OK
    - Tested operation - Normal
    
    Next Service: 2024-07-15 or 2000 hours
    """,
    doc_type=DocumentType.MAINTENANCE_LOG,
    industry=IndustryNiche.MANUFACTURING,
    title="XYZ-500 Belt Replacement - Jan 2024",
    source="maintenance_system",
    created_at=datetime(2024, 1, 15),
    metadata={
        "equipment_model": "XYZ-500",
        "serial_number": "123456",
        "maintenance_type": "preventive",
        "technician": "John Doe",
        "parts_replaced": ["BLT-XYZ-500"]
    }
)

engine.index_document(maintenance_log1)

maintenance_log2 = Document(
    id="maint_log_xyz500_002_2024_03",
    content="""
    Date: 2024-03-20
    Equipment: Compressor XYZ-500 (Serial: 123456)
    Maintenance Type: Emergency
    Technician: Jane Smith
    Issue: Loud squealing noise during operation
    
    Diagnosis: Belt slipped off pulley, belt worn
    
    Work Performed:
    - Emergency shutdown
    - Removed old belt (found worn and cracked)
    - Installed new belt (Part #: BLT-XYZ-500)
    - Checked pulley alignment - Found misalignment
    - Adjusted alignment
    - Adjusted tension to 55 lbs
    - Tested operation - Normal
    
    Root Cause: Pulley misalignment caused premature belt wear
    Recommendation: Check alignment monthly
    """,
    doc_type=DocumentType.MAINTENANCE_LOG,
    industry=IndustryNiche.MANUFACTURING,
    title="XYZ-500 Emergency Repair - Mar 2024",
    source="maintenance_system",
    created_at=datetime(2024, 3, 20),
    metadata={
        "equipment_model": "XYZ-500",
        "serial_number": "123456",
        "maintenance_type": "emergency",
        "technician": "Jane Smith",
        "issue": "squealing_noise",
        "parts_replaced": ["BLT-XYZ-500"]
    }
)

engine.index_document(maintenance_log2)

# Query 1: How-to question
query1 = RAGQuery(
    query="How do I replace the belt on compressor XYZ-500?",
    industry=IndustryNiche.MANUFACTURING,
    doc_types=[DocumentType.MANUAL],
    top_k=3
)

results1 = engine.retrieve(query1)
print("=== Query: How do I replace the belt? ===")
for result in results1:
    print(f"\nDocument: {result.document.title}")
    print(f"Content: {result.document.content}")

# Query 2: Historical maintenance
query2 = RAGQuery(
    query="When was the last belt replacement on XYZ-500?",
    industry=IndustryNiche.MANUFACTURING,
    doc_types=[DocumentType.MAINTENANCE_LOG],
    metadata_filters={"equipment_model": "XYZ-500"},
    top_k=5
)

results2 = engine.retrieve(query2)
print("\n\n=== Query: Last belt replacement? ===")
for result in results2:
    print(f"\nDate: {result.document.created_at}")
    print(f"Summary: {result.document.title}")
    print(f"Details: {result.document.content[:200]}...")

# Query 3: Troubleshooting
query3 = RAGQuery(
    query="Compressor making squealing noise, what should I check?",
    industry=IndustryNiche.MANUFACTURING,
    doc_types=[DocumentType.MANUAL, DocumentType.MAINTENANCE_LOG],
    metadata_filters={"equipment_model": "XYZ-500"},
    top_k=3
)

results3 = engine.retrieve(query3)
print("\n\n=== Query: Squealing noise troubleshooting ===")
for result in results3:
    print(f"\nDocument: {result.document.title}")
    print(f"Type: {result.document.doc_type.value}")
    print(f"Relevant info: {result.document.content[:300]}...")
```

---

## Example 3: Multi-Industry Knowledge Base

**Scenario:** Create a cross-industry safety guidelines database

```python
# Create generic RAG engine for cross-industry content
engine = create_universal_rag_engine(
    industry=IndustryNiche.GENERIC,
    collection_name="deepiri_safety_guidelines"
)

# Index manufacturing safety guideline
manufacturing_safety = Document(
    id="safety_manufacturing_fire_2024",
    content="""
    Fire Safety Requirements for Manufacturing Facilities
    (OSHA 29 CFR 1910.157)
    
    Requirements:
    - Fire extinguishers must be accessible within 75 feet of any point
    - Monthly inspections required
    - Annual professional servicing required
    - Employees must be trained annually
    - Emergency evacuation plan must be posted
    - Emergency exits must be clearly marked and unobstructed
    
    Types of Extinguishers:
    - Class A: Ordinary combustibles (wood, paper)
    - Class B: Flammable liquids (oil, gasoline)
    - Class C: Electrical equipment
    - Class D: Combustible metals
    """,
    doc_type=DocumentType.SAFETY_GUIDELINE,
    industry=IndustryNiche.MANUFACTURING,
    title="Fire Safety - Manufacturing Facilities",
    source="osha.gov",
    metadata={
        "regulation": "OSHA 29 CFR 1910.157",
        "topic": "fire_safety"
    }
)

engine.index_document(manufacturing_safety)

# Index construction safety guideline
construction_safety = Document(
    id="safety_construction_fall_2024",
    content="""
    Fall Protection Requirements for Construction
    (OSHA 29 CFR 1926.501)
    
    Fall protection required when working:
    - 6 feet or more above a lower level
    - On scaffolding
    - On roofs
    - Near unprotected edges
    - Near holes or openings
    
    Acceptable Fall Protection Systems:
    - Guardrail systems
    - Safety net systems
    - Personal fall arrest systems (harness + lanyard)
    
    Requirements:
    - Inspect equipment before each use
    - Anchor points must support 5000 lbs per worker
    - Training required before working at heights
    """,
    doc_type=DocumentType.SAFETY_GUIDELINE,
    industry=IndustryNiche.CONSTRUCTION,
    title="Fall Protection - Construction Sites",
    source="osha.gov",
    metadata={
        "regulation": "OSHA 29 CFR 1926.501",
        "topic": "fall_protection"
    }
)

engine.index_document(construction_safety)

# Cross-industry query
query = RAGQuery(
    query="What are the fire extinguisher requirements?",
    doc_types=[DocumentType.SAFETY_GUIDELINE],
    top_k=5
)

results = engine.retrieve(query)
print("=== Query: Fire extinguisher requirements ===")
for result in results:
    print(f"\nIndustry: {result.document.industry.value}")
    print(f"Document: {result.document.title}")
    print(f"Content: {result.document.content[:300]}...")

# Industry-specific query
query_mfg = RAGQuery(
    query="Safety requirements for manufacturing facilities",
    industry=IndustryNiche.MANUFACTURING,
    doc_types=[DocumentType.SAFETY_GUIDELINE],
    top_k=3
)

results_mfg = engine.retrieve(query_mfg)
print("\n\n=== Manufacturing-specific safety query ===")
for result in results_mfg:
    print(f"\nDocument: {result.document.title}")
    print(f"Requirements: {result.document.content}")
```

---

## Example 4: Batch Processing with Document Processors

**Scenario:** Bulk import regulations with automatic chunking

```python
from deepiri_modelkit.rag.processors import get_processor

# Create regulation processor
processor = get_processor(
    doc_type=DocumentType.REGULATION,
    chunk_size=1000,
    chunk_overlap=200
)

# Long regulation document
long_regulation = """
Section 1: General Requirements
All manufacturing facilities must comply with OSHA standards...
[... 10,000 words of regulation text ...]

Section 2: Equipment Safety
Machinery must be equipped with appropriate safety guards...
[... more content ...]

Section 3: Employee Training
Employees must receive annual safety training...
[... more content ...]
"""

# Process document (auto-chunks)
chunks = processor.process(
    raw_content=long_regulation,
    metadata={
        "id": "osha_1910_base",
        "industry": "manufacturing",
        "title": "OSHA 1910 - General Industry Standards",
        "source": "osha.gov",
        "regulation_number": "29 CFR 1910"
    }
)

# Index all chunks
engine = create_universal_rag_engine(industry=IndustryNiche.MANUFACTURING)
result = engine.index_documents(chunks)

print(f"Indexed {result['indexed_count']} document chunks")

# Query will retrieve relevant chunks
query = RAGQuery(
    query="What are the employee training requirements?",
    industry=IndustryNiche.MANUFACTURING,
    doc_types=[DocumentType.REGULATION]
)

results = engine.retrieve(query)
print("\n=== Query Results ===")
for result in results:
    print(f"\nChunk {result.document.chunk_index + 1}/{result.document.total_chunks}")
    print(f"Content: {result.document.content[:200]}...")
```

---

## Example 5: API Integration (HTTP Requests)

**Scenario:** Index and query via REST API

```bash
# 1. Index a document
curl -X POST http://localhost:8000/api/v1/universal-rag/index \
  -H "Content-Type: application/json" \
  -d '{
    "id": "manual_hvac_system_2024",
    "content": "HVAC System Maintenance Manual: Regular filter replacement required every 3 months. Check refrigerant levels annually. Inspect ductwork for leaks...",
    "doc_type": "manual",
    "industry": "property_management",
    "title": "HVAC Maintenance Manual",
    "source": "facilities/manuals/hvac.pdf",
    "metadata": {
      "equipment_type": "hvac",
      "building": "main_office"
    }
  }'

# 2. Search
curl -X POST http://localhost:8000/api/v1/universal-rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How often should I replace HVAC filters?",
    "industry": "property_management",
    "doc_types": ["manual", "procedure"],
    "top_k": 3
  }'

# 3. Generate answer
curl -X POST http://localhost:8000/api/v1/universal-rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What maintenance is required for HVAC systems?",
    "industry": "property_management",
    "doc_types": ["manual"],
    "top_k": 5
  }'

# 4. Get statistics
curl http://localhost:8000/api/v1/universal-rag/stats/property_management
```

---

## Running the Examples

1. **Start Cyrex service:**
```bash
cd deepiri-platform/diri-cyrex
docker-compose up
```

2. **Run Python examples:**
```bash
python examples/rag_insurance_example.py
python examples/rag_manufacturing_example.py
```

3. **Test API:**
```bash
# Use curl commands above or Postman
# API docs: http://localhost:8000/docs
```

---

## Next Steps

- Integrate with LLM for full generation
- Add more document types
- Implement custom processors
- Monitor retrieval performance

