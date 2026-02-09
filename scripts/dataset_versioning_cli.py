"""
CLI for dataset versioning operations
"""
import click
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.dataset_versioning import (
    DatasetVersionManager,
    DatasetType,
)

@click.group()
def dataset():
    """Dataset versioning commands"""
    pass

@dataset.command()
@click.option("--name", required=True, help="Dataset name")
@click.option("--type", type=click.Choice([dt.value for dt in DatasetType]), required=True)
@click.option("--path", type=click.Path(exists=True), required=True)
@click.option("--version", help="Version string (auto-increments if not provided)")
@click.option("--parent", help="Parent version")
@click.option("--summary", help="Change summary")
@click.option("--tags", help="Comma-separated tags")
def create(name, type, path, version, parent, summary, tags):
    """Create a new dataset version"""
    manager = DatasetVersionManager(
        db_url="sqlite:///./dataset_versions.db",
        storage_backend="local"
    )

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    metadata = manager.create_version(
        dataset_name=name,
        dataset_type=DatasetType(type),
        data_path=Path(path),
        version=version,
        parent_version=parent,
        change_summary=summary,
        tags=tag_list,
    )

    click.echo(f"Created version {metadata.version} of {name}")
    click.echo(f"  Storage: {metadata.storage_path}")
    click.echo(f"  Samples: {metadata.total_samples}")
    click.echo(f"  Checksum: {metadata.data_checksum[:16]}...")

@dataset.command(name="list")
@click.option("--name", required=True)
@click.option("--version", help="Specific version (default: latest)")
def list_versions_cmd(name, version):
    """List dataset versions"""
    manager = DatasetVersionManager(
        db_url="sqlite:///./dataset_versions.db"
    )

    if version:
        metadata = manager.get_version(name, version)
        if metadata:
            click.echo(f"Version {metadata.version}:")
            click.echo(f"  Created: {metadata.created_at}")
            click.echo(f"  Samples: {metadata.total_samples}")
            click.echo(f"  Change: {metadata.change_type}")
        else:
            click.echo(f"Version {version} not found")
    else:
        versions = manager.list_versions(dataset_name=name)
        click.echo(f"Versions for {name}:")
        for v in versions:
            click.echo(f"  {v.version} - {v.created_at} - {v.total_samples} samples")

@dataset.command()
@click.option("--name", required=True)
@click.option("--version1", required=True)
@click.option("--version2", required=True)
def compare(name, version1, version2):
    """Compare two dataset versions"""
    manager = DatasetVersionManager(
        db_url="sqlite:///./dataset_versions.db"
    )

    comparison = manager.compare_versions(name, version1, version2)

    click.echo(f"Comparison: {version1} vs {version2}")
    click.echo(f"  Sample difference: {comparison['sample_difference']:+d}")
    click.echo(f"  File difference: {comparison['file_difference']:+d}")
    click.echo(f"  Size difference: {comparison['size_difference_bytes']:+d} bytes")
    click.echo(f"  Change type: {comparison['change_type']}")

if __name__ == "__main__":
    dataset()
