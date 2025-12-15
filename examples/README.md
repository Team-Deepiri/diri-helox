# Helox Examples

The `examples/` folder contains **runnable example scripts** that demonstrate how to use Helox for LLM training.

## Purpose

These examples serve as:
1. **Learning Resources** - Show how to use Helox features
2. **Integration Guides** - Demonstrate Cyrex and Synapse integration
3. **Quick Start Templates** - Copy and modify for your use case
4. **Testing Scripts** - Verify your setup is working

## Files

### `complete_training_example.py`
**Purpose**: Complete end-to-end training example with all 38 features

**Shows**:
- Full training pipeline setup
- All features initialization
- Cyrex RAG integration
- Synapse event publishing
- Data safety checks
- Dataset versioning
- Complete training loop

**Usage**:
```bash
python examples/complete_training_example.py
```

**Best for**: Understanding the complete training workflow

---

### `integration_example.py`
**Purpose**: Integration examples with Deepiri platform

**Shows**:
- How to collect training data from Deepiri features
- Integration with Cyrex services
- Data collection patterns

**Best for**: Understanding Deepiri platform integration

---

### `deepiri_integration_example.py`
**Purpose**: Deepiri-specific integration patterns

**Shows**:
- Prompt-to-tasks engine integration
- Tier 1/2/3 ability system integration
- Gamification system data collection

**Best for**: Deepiri-specific use cases

---

## When to Use Examples

### Use `complete_training_example.py` when:
- ✅ You want to see all features working together
- ✅ You're learning how Helox works
- ✅ You need a template to start from
- ✅ You want to verify your setup

### Use Production Scripts when:
- ✅ You're running actual training jobs
- ✅ You need CLI arguments and configuration
- ✅ You want production-grade error handling

**Production Script**: `scripts/train_with_full_features.py`  
**Example Script**: `examples/complete_training_example.py`

## Creating Your Own Example

To create a custom example:

```python
#!/usr/bin/env python3
"""
My custom training example.
"""

import asyncio
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.unified_training_orchestrator import UnifiedTrainingOrchestrator
# ... your code ...

async def main():
    # Your example code here
    pass

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices

1. **Keep examples simple** - Focus on one concept
2. **Add comments** - Explain what each section does
3. **Include error handling** - Show proper error handling patterns
4. **Document assumptions** - Note what needs to be set up first

## Examples vs Scripts

| Examples | Scripts |
|----------|---------|
| Educational | Production |
| Demonstrative | Operational |
| Copy & modify | Run directly |
| Show concepts | Execute tasks |

## Quick Reference

- **Learn**: Read `complete_training_example.py`
- **Run Training**: Use `scripts/train_with_full_features.py`
- **Test Integration**: Run `complete_training_example.py`
- **Custom Use Case**: Copy example and modify

