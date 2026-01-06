from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

# Default label ID range for classification models (31 categories: 0-30)
DEFAULT_MIN_LABEL_ID = 0
DEFAULT_MAX_LABEL_ID = 30


@dataclass
class ProcessedData:
    """
    Standardized output format for processed data.
    This class holds the result of any preprocessing stage.
    """
    data: Any  # The actual processed data (can be dict, list, DataFrame, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata about processing
    quality_metrics: Dict[str, float] = field(default_factory=dict)  # Quality scores (completeness, accuracy, etc.)
    schema_version: Optional[str] = None  # Schema version used for validation
    
    
@dataclass
class ValidationResult:
    """
    Results from validation checks.
    This class holds the outcome of data validation operations.
    """
    is_valid: bool  # Whether the data passed validation
    errors: List[str] = field(default_factory=list)  # List of error messages if validation failed
    warnings: List[str] = field(default_factory=list)  # List of warning messages (non-critical issues)
    quality_scores: Dict[str, float] = field(default_factory=dict)  # Quality dimension scores (completeness, accuracy, etc.)
    
    
@dataclass
class StageResult:
    """
    Results from stage execution.
    This class holds the complete result of running a preprocessing stage.
    """
    success: bool  # Whether the stage executed successfully
    processed_data: Optional[ProcessedData] = None  # The processed data (if successful)
    validation_result: Optional[ValidationResult] = None  # Validation results (if validation was performed)
    execution_time: Optional[float] = None  # Time taken to execute in seconds
    error: Optional[str] = None  # Error message if the stage failed
    stage_name: Optional[str] = None  # Name of the stage that executed
    
class PreprocessingStage(ABC):
    """
    Abstract base class for all preprocessing stages.
    All preprocessing stages (DataLoading, DataCleaning, etc.) must inherit from this class.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessing stage.
        
        Args:
            name: Name of this stage (e.g., "DataLoading", "DataCleaning")
            config: Optional configuration dictionary for this stage
        """
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    def process(self, data: Any) -> StageResult:
        """
        Process the input data.
        This method must be implemented by all child classes.
        
        Args:
            data: Input data to process
            
        Returns:
            StageResult: Result of the processing stage
        """
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate the input data.
        This method must be implemented by all child classes.
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult: Validation results
        """
        pass
    
    def get_dependencies(self) -> List[str]:
        """
        Get list of stage names this stage depends on.
        Stages listed here must run before this stage.
        
        Returns:
            List of stage names that must run before this stage
        """
        return []
    
    def get_name(self) -> str:
        """
        Get the name of this stage.
        
        Returns:
            Name of the stage
        """
        return self.name
    
    def _extract_data_and_metadata(self, data: Any) -> tuple[Any, Dict[str, Any]]:
        """
        Extract actual data and metadata from ProcessedData or raw data.
        
        Args:
            data: Input data (ProcessedData instance or raw data)
            
        Returns:
            Tuple of (actual_data, metadata)
        """
        if isinstance(data, ProcessedData):
            return data.data, data.metadata
        else:
            return data, {}
    
    def _extract_data(self, data: Any) -> Any:
        """
        Extract actual data from ProcessedData or raw data (metadata not needed).
        
        Args:
            data: Input data (ProcessedData instance or raw data)
            
        Returns:
            Actual data (without metadata wrapper)
        """
        if isinstance(data, ProcessedData):
            return data.data
        else:
            return data
    
    def _process_items(
        self, 
        actual_data: Any, 
        process_func: Callable[[Any], Any],
        allow_empty: bool = False
    ) -> Any:
        """
        Process data items, handling both lists and single items.
        
        Args:
            actual_data: Data to process (list or single item)
            process_func: Function to process a single item
            allow_empty: Whether to allow empty data (default: False)
            
        Returns:
            Processed data (list or single item, same structure as input)
        """
        if actual_data is None:
            if allow_empty:
                return None
            raise ValueError(f"Cannot process None data in stage '{self.name}'")
        
        if isinstance(actual_data, list):
            if len(actual_data) == 0 and not allow_empty:
                raise ValueError(f"Cannot process empty list in stage '{self.name}'")
            return [process_func(item) for item in actual_data]
        else:
            return process_func(actual_data)
    
    def _check_data_not_none(self, actual_data: Any, context: str = "validate") -> None:
        """
        Standard check: None data always fails (raises error).
        
        This ensures consistent behavior between process() and validate().
        
        Args:
            actual_data: Data to check
            context: Context for error message (default: "validate")
            
        Raises:
            ValueError: If data is None
        """
        if actual_data is None:
            raise ValueError(f"Cannot {context} None data in stage '{self.name}'")
    
    def _validate_items(
        self,
        actual_data: Any,
        validate_func: Callable[[Any], tuple[List[str], List[str]]],
        empty_error: Optional[str] = None
    ) -> tuple[List[str], List[str]]:
        """
        Validate data items (list or single item) using a diagnostic function.
        
        Args:
            actual_data: Data to validate (list or single item)
            validate_func: Function that takes an item and returns (errors, warnings)
            empty_error: Error message for empty data (None = fail on empty, consistent with process())
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Standard check: empty data fails (consistent with process() behavior)
        if actual_data == [] or actual_data == {}:
            error_msg = empty_error or f"Cannot validate empty data in stage '{self.name}'"
            errors.append(error_msg)
            return errors, warnings
        
        # Validate data structure
        if isinstance(actual_data, list):
            # Check each item (we already handled empty list case above)
            for i, item in enumerate(actual_data):
                item_errors, item_warnings = validate_func(item)
                # Prefix errors/warnings with item index
                for err in item_errors:
                    errors.append(f"Item at index {i}: {err}")
                for warn in item_warnings:
                    warnings.append(f"Item at index {i}: {warn}")
        
        elif isinstance(actual_data, dict):
            # Validate single dictionary
            item_errors, item_warnings = validate_func(actual_data)
            errors.extend(item_errors)
            warnings.extend(item_warnings)
        
        else:
            # Data is neither dict nor list
            errors.append(f"Data must be a dictionary or list, got {type(actual_data).__name__}")
        
        return errors, warnings
    
    def _create_validation_result(
        self,
        errors: List[str],
        warnings: List[str]
    ) -> ValidationResult:
        """
        Create a ValidationResult from errors and warnings.
        
        Args:
            errors: List of error messages
            warnings: List of warning messages
            
        Returns:
            ValidationResult instance
        """
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _create_result(
        self, 
        processed_data: Any,
        original_metadata: Dict[str, Any],
        metadata_updates: Dict[str, Any],
        success: bool = True,
        error: Optional[str] = None
    ) -> StageResult:
        """
        Create a StageResult with processed data wrapped in ProcessedData.
        
        Args:
            processed_data: The processed data
            original_metadata: Metadata from previous stage
            metadata_updates: Additional metadata to add
            success: Whether processing was successful
            error: Error message if processing failed
            
        Returns:
            StageResult instance
        """
        if success:
            processed_data_obj = ProcessedData(
                data=processed_data,
                metadata={**original_metadata, **metadata_updates}
            )
            return StageResult(
                success=True,
                processed_data=processed_data_obj,
                stage_name=self.name
            )
        else:
            return StageResult(
                success=False,
                error=error,
                stage_name=self.name
            )
    
    