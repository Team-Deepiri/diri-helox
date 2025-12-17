from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    
    
    