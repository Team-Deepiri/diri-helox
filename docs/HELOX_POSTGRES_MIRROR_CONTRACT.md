# Helox Postgres Training Contract

This document defines the Cyrex-owned durable Postgres table used by Helox when
running the live Redis + historical backfill pipeline.

## Why

- Redis streams remain the primary live source.
- Cyrex Postgres provides replay/backfill for historical training.
- Helox consumes both with weighted composition.
- This is not a Helox mirror table; Cyrex owns the schema and write path.

## Canonical Table

```sql
CREATE SCHEMA IF NOT EXISTS cyrex;

CREATE TABLE IF NOT EXISTS cyrex.helox_training_samples (
    record_id TEXT PRIMARY KEY,
    stream_type TEXT NOT NULL CHECK (stream_type IN ('raw', 'structured')),
    producer TEXT NOT NULL DEFAULT 'cyrex_realtime_pipeline',

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

Helox keeps `StreamDataSource` and `PostgresDataSource` as reusable low-level
primitives. The production Cyrex -> Helox lane should use the domain-specific
aliases on top of those primitives:

- `cyrex_training_stream` / `CyrexTrainingStreamSource`
- `cyrex_training_postgres` / `CyrexTrainingPostgresSource`

That keeps the generic data-source layer reusable while making the Cyrex
producer, Redis channels, and durable replay table explicit in configs.

`CyrexTrainingPostgresSource` defaults to:

- Cyrex split database DSN (`localhost:5434/cyrex_db` locally, or injected
  `POSTGRES_DSN` / `CYREX_POSTGRES_DSN` in deployment)
- `table="cyrex.helox_training_samples"`
- `producer="cyrex_realtime_pipeline"`
- `min_quality=0.4`

and supports optional filters:

- `stream_type` (`raw` or `structured`)
- `producer` (for producer-specific pipelines)

`CyrexTrainingStreamSource` reads:

- `pipeline.helox-training.raw`
- `pipeline.helox-training.structured`

These sources feed the config-driven `DynamicTrainingPipeline`, which already
wraps trainer callbacks with `deepiri-training-orchestrator` for reproducible
seeding, checkpoints, experiment tracking, and model-ready publication. This PR
does not create a separate training orchestrator; it hardens the Cyrex-produced
sample ingestion path used by that pipeline.
