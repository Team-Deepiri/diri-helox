CREATE TABLE dataset_versions (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    dataset_type VARCHAR(50) NOT NULL,

    storage_path VARCHAR(500) NOT NULL,
    storage_backend VARCHAR(50) DEFAULT 's3',

    total_samples INTEGER NOT NULL,
    file_count INTEGER NOT NULL,
    total_size_bytes INTEGER NOT NULL,

    data_checksum VARCHAR(64) NOT NULL,
    metadata_checksum VARCHAR(64) NOT NULL,

    parent_version VARCHAR(50),
    change_summary TEXT,
    change_type VARCHAR(50),

    quality_score VARCHAR(20),
    validation_status VARCHAR(50) DEFAULT 'PENDING',
    validation_errors JSONB,

    tags JSONB DEFAULT '[]',
    dataset_metadata JSONB DEFAULT '{}',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),

    UNIQUE(dataset_name, version, dataset_type)
);

CREATE INDEX idx_dataset_versions_name ON dataset_versions(dataset_name);
CREATE INDEX idx_dataset_versions_type ON dataset_versions(dataset_type);
CREATE INDEX idx_dataset_versions_created ON dataset_versions(created_at);
