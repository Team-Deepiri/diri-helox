"""
Preprocessing stage implementations.

This module contains concrete implementations of PreprocessingStage for various
data preprocessing operations. These stages wrap and extend the existing
preprocessing components.
"""

from typing import Any, Dict, List, Optional
from .base import PreprocessingStage, StageResult, ProcessedData, ValidationResult


class DataLoadingStage(PreprocessingStage):
    """
    Stage for loading data from various sources (JSONL, JSON, databases, etc.).
    This is typically the first stage in a preprocessing pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="data_loading", config=config)
        self.source = config.get("source") if config else None
        self.format = config.get("format", "jsonl") if config else "jsonl"
    
    def get_dependencies(self) -> List[str]:
        """Data loading has no dependencies - it's the first stage."""
        return []
    
    def process(self, data: Any) -> StageResult:
        """
        Load data from the configured source.
        
        Args:
            data: Optional initial data (usually None for loading stage)
            
        Returns:
            StageResult with loaded data
        """
        try:
            # TODO: Implement actual data loading based on source and format
            # For now, if data is provided, pass it through
            if data is not None:
                processed_data = ProcessedData(
                    data=data,
                    metadata={"source": self.source, "format": self.format}
                )
            else:
                # In a real implementation, load from source
                raise NotImplementedError(
                    "Data loading from source not yet implemented. "
                    "Please provide initial_data to the pipeline."
                )
            
            return StageResult(
                success=True,
                processed_data=processed_data,
                stage_name=self.name
            )
        except Exception as e:
            return StageResult(
                success=False,
                error=f"Data loading failed: {str(e)}",
                stage_name=self.name
            )
    
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate that the data source is accessible and format is supported.
        
        Args:
            data: Data to validate (usually None for loading stage)
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        # Check if source is configured
        if not self.source:
            warnings.append("No data source configured")
        
        # Validate format
        supported_formats = ["jsonl", "json", "csv", "parquet"]
        if self.format not in supported_formats:
            errors.append(f"Unsupported format: {self.format}. Supported: {supported_formats}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


class DataCleaningStage(PreprocessingStage):
    """
    Stage for cleaning data.
    This stage is responsible for cleaning the data to remove noise and outliers.
    """
    
    def __init__(self, config : Optional[Dict[str, Any]] = None):
        super().__init__(name="data_cleaning", config=config)
        
    def get_dependencies(self) -> List[str]:
        return ["data_loading"]
        
    def process(self, data: Any) -> StageResult:
        """
        Clean the input data by removing spaces, nulls, and duplicates.
        
        Args:
            data: Data from previous stage (usually ProcessedData or dict/list)
            
        Returns:
            StageResult with cleaned data
        """
        try:
            # Step 1: Extract actual data if it's ProcessedData
            if isinstance(data, ProcessedData):
                actual_data = data.data
                original_metadata = data.metadata
            else:
                actual_data = data
                original_metadata = {}
            
            # Step 2: Handle None or empty data
            if actual_data is None:
                raise ValueError("Cannot clean None data")
            
            # Step 3: Process list or single item
            if isinstance(actual_data, list):
                # Process each item in the list
                cleaned_items = []
                for item in actual_data:
                    cleaned_item = self._clean_single_item(item)
                    cleaned_items.append(cleaned_item)
                
                # Step 4: Remove duplicates from list
                cleaned_data = self._remove_duplicates(cleaned_items)
            else:
                # Process single item
                cleaned_data = self._clean_single_item(actual_data)
            
            # Step 5: Wrap cleaned data in ProcessedData
            processed_data = ProcessedData(
                data=cleaned_data,
                metadata={
                    **original_metadata,
                    "cleaned": True,
                    "cleaning_stage": self.name
                }
            )
            
            # Step 6: Return successful result
            return StageResult(
                success=True,
                processed_data=processed_data,
                stage_name=self.name
            )
            
        except Exception as e:
            return StageResult(
                success=False,
                error=f"Data cleaning failed: {str(e)}",
                stage_name=self.name
            )
    
    def _clean_single_item(self, item: Dict) -> Dict:
        """Clean a single data item by removing spaces and nulls."""
        if not isinstance(item, dict):
            return item
        
        cleaned = item.copy()
        
        # Remove extra spaces from text field
        if "text" in cleaned and isinstance(cleaned["text"], str):
            cleaned["text"] = " ".join(cleaned["text"].split())
        
        # Remove null/empty values
        cleaned = {k: v for k, v in cleaned.items() 
                   if v is not None and v != ""}
        
        return cleaned
    
    def _remove_duplicates(self, items: List[Dict]) -> List[Dict]:
        """Remove duplicate items from a list."""
        seen = []
        unique_items = []
        for item in items:
            # Convert dict to tuple for comparison
            item_tuple = tuple(sorted(item.items()))
            if item_tuple not in seen:
                seen.append(item_tuple)
                unique_items.append(item)
        return unique_items
     
    
    
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate that the data is cleanable and has required fields.
        
        Args:
            data: Data to validate (can be ProcessedData, dict, list, or None)
            
        Returns:
            ValidationResult with validation errors and warnings
        """
        errors = []
        warnings = []
        
        # Extract actual data if it's ProcessedData
        if isinstance(data, ProcessedData):
            actual_data = data.data
        else:
            actual_data = data
        
        # Check if data is None
        if actual_data is None:
            errors.append("Cannot validate None data")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings
            )
        
        # Check if data is empty
        if actual_data == [] or actual_data == {}:
            warnings.append("Data is empty - nothing to clean")
        
        # Validate data structure
        if isinstance(actual_data, list):
            # Validate list of items
            if len(actual_data) == 0:
                warnings.append("List is empty")
            else:
                # Check first few items to see if they're valid
                for i, item in enumerate(actual_data[:5]):  # Check first 5 items
                    if not isinstance(item, dict):
                        errors.append(f"Item at index {i} is not a dictionary")
                    elif "text" not in item:
                        warnings.append(f"Item at index {i} missing 'text' field")
        
        elif isinstance(actual_data, dict):
            # Validate single dictionary
            if "text" not in actual_data:
                warnings.append("Data missing 'text' field - may not be cleanable")
            
            # Check if text field is valid
            if "text" in actual_data:
                if not isinstance(actual_data["text"], str):
                    warnings.append("'text' field is not a string")
                elif actual_data["text"].strip() == "":
                    warnings.append("'text' field is empty or only whitespace")
        
        else:
            # Data is neither dict nor list
            errors.append(f"Data must be a dictionary or list, got {type(actual_data).__name__}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    
    
    
    