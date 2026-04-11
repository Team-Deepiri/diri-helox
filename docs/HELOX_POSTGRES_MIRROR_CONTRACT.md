# Helox Postgres Mirror Contract (Option B)

This document defines the durable Postgres mirror used by Helox when running
the live Redis + historical backfill pipeline.

## Why

- Redis stream remains the primary live source.
- Postgres provides replay/backfill for historical training.
- Helox consumes both with weighted composition.

## Canonical Table

```sql
CREATE SCHEMA IF NOT EXISTS cyrex;

CREATE TABLE IF NOT EXISTS cyrex.helox_training_samples (
    record_id TEXT PRIMARY KEY,
    stream_type TEXT NOT NULL CHECK (stream_type IN ('raw', 'structured')),
    producer TEXT NOT NULL DEFAULT 'language_intelligence',

    -- raw payload fields
    text TEXT,

    -- structured payload fields
    instruction TEXT,
    input_text TEXT,
    output_text TEXT,
    category TEXT,

    quality_score DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_helox_training_samples_created_at
    ON cyrex.helox_training_samples (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_helox_training_samples_stream_type
    ON cyrex.helox_training_samples (stream_type);

CREATE INDEX IF NOT EXISTS idx_helox_training_samples_quality
    ON cyrex.helox_training_samples (quality_score);

CREATE INDEX IF NOT EXISTS idx_helox_training_samples_producer
    ON cyrex.helox_training_samples (producer);
```

## Cyrex Producer Requirement

Whenever Cyrex publishes to Redis streams:

- `pipeline.helox-training.raw`
- `pipeline.helox-training.structured`

it should also upsert the same logical record into
`cyrex.helox_training_samples` with matching `record_id`.

This makes Redis the low-latency source and Postgres the durable audit/replay
source.

## Helox Consumer Defaults

`PostgresDataSource` defaults to:

- `table="cyrex.helox_training_samples"`
- `min_quality=0.4`

and supports optional filters:

- `stream_type` (`raw` or `structured`)
- `producer` (for producer-specific pipelines)
