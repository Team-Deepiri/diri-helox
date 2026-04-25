try:
    from deepiri_modelkit.data.validation import DatasetValidator, validate_dataset_quality
except ImportError:
    DatasetValidator = None

    def validate_dataset_quality(*args, **kwargs):
        raise ImportError("deepiri_modelkit is not installed")
__all__ = ["DatasetValidator", "validate_dataset_quality"]
