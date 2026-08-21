-- Align dataset_versions.storage_backend default with deepiri-dataset-processor v0.3.2:
-- only the 'local' backend is implemented; new rows must not default to 's3'.
ALTER TABLE dataset_versions
    ALTER COLUMN storage_backend SET DEFAULT 'local';
