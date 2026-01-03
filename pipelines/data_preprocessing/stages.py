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
        
        
class DataValidationStage(PreprocessingStage):
    """
    Stage for validating data.
    This stage is responsible for validating the data to ensure it is clean and ready for processing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="data_validation", config=config)
        self.required_fields = config.get("required_fields", ["text", "label"]) if config else ["text", "label"]
    
    def get_dependencies(self) -> List[str]:
        return ["data_cleaning"]
    
    def process(self, data: Any) -> StageResult:
        """
        Validate the input data by checking structure and required fields.
        
        Args:
            data: Data from previous stage (usually ProcessedData or dict/list)
            
        Returns:
            StageResult with validated data
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
                raise ValueError("Cannot validate None data")
            
            # Step 3: Validate list or single item
            if isinstance(actual_data, list):
                # Validate each item in the list
                validated_items = []
                for item in actual_data:
                    validated_item = self._validate_single_item(item)
                    validated_items.append(validated_item)
                validated_data = validated_items
            else:
                # Validate single item
                validated_data = self._validate_single_item(actual_data)
            
            # Step 4: Wrap validated data in ProcessedData
            processed_data = ProcessedData(
                data=validated_data,
                metadata={
                    **original_metadata,
                    "validated": True,
                    "validation_stage": self.name
                }
            )
            
            # Step 5: Return successful result
            return StageResult(
                success=True,
                processed_data=processed_data,
                stage_name=self.name
            )
            
        except Exception as e:
            return StageResult(
                success=False,
                error=f"Data validation failed: {str(e)}",
                stage_name=self.name
            )
    
    def _validate_single_item(self, item: Dict) -> Dict:
        """Validate a single data item by checking required fields."""
        if not isinstance(item, dict):
            raise ValueError(f"Data item must be a dictionary, got {type(item).__name__}")
        
        # Check required fields
        for field in self.required_fields:
            if field not in item:
                raise ValueError(f"Required field '{field}' is missing")
        
        # Check text field is not empty
        if "text" in item:
            if not isinstance(item["text"], str):
                raise ValueError("'text' field must be a string")
            if item["text"].strip() == "":
                raise ValueError("'text' field cannot be empty")
        
        # Return the item as-is (validation passed)
        return item
    
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate that the data has required fields and meets validation criteria.
        
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
            warnings.append("Data is empty - nothing to validate")
            return ValidationResult(
                is_valid=True,  # Empty is valid but warning-worthy
                errors=errors,
                warnings=warnings
            )
        
        # Validate data structure
        if isinstance(actual_data, list):
            # Validate list of items
            if len(actual_data) == 0:
                warnings.append("List is empty")
            else:
                # Check each item for validation issues
                for i, item in enumerate(actual_data):
                    item_errors, item_warnings = self._validate_item_diagnostics(item)
                    # Prefix errors/warnings with item index
                    for err in item_errors:
                        errors.append(f"Item at index {i}: {err}")
                    for warn in item_warnings:
                        warnings.append(f"Item at index {i}: {warn}")
        
        elif isinstance(actual_data, dict):
            # Validate single dictionary
            item_errors, item_warnings = self._validate_item_diagnostics(actual_data)
            errors.extend(item_errors)
            warnings.extend(item_warnings)
        
        else:
            # Data is neither dict nor list
            errors.append(f"Data must be a dictionary or list, got {type(actual_data).__name__}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_item_diagnostics(self, item: Dict) -> tuple[List[str], List[str]]:
        """
        Validate a single item and return errors and warnings (non-destructive).
        
        Args:
            item: Data item to validate
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check if item is a dictionary
        if not isinstance(item, dict):
            errors.append(f"Item must be a dictionary, got {type(item).__name__}")
            return errors, warnings
        
        # Check required fields
        for field in self.required_fields:
            if field not in item:
                errors.append(f"Required field '{field}' is missing")
        
        # Check text field validity
        if "text" in item:
            if not isinstance(item["text"], str):
                errors.append("'text' field must be a string")
            elif item["text"].strip() == "":
                errors.append("'text' field cannot be empty")
            elif len(item["text"].strip()) < 3:
                warnings.append("'text' field is very short (less than 3 characters)")
        
        # Check label field if present
        if "label" in item:
            if item["label"] is None:
                warnings.append("'label' field is None")
            elif isinstance(item["label"], str) and item["label"].strip() == "":
                warnings.append("'label' field is empty string")
        
        return errors, warnings
    
class DataRoutingStage(PreprocessingStage):
    """
    Stage for routing data and mapping labels to integer IDs.
    This stage converts string labels to numeric IDs for model training.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="data_routing", config=config)
        # Get label mapping from config, or use default empty dict
        self.label_mapping = config.get("label_mapping", {}) if config else {}
    
    def get_dependencies(self) -> List[str]:
        return ["data_validation"]
    
    def process(self, data: Any) -> StageResult:
        """
        Map string labels to integer IDs for model training.
        
        Args:
            data: Data from previous stage (usually ProcessedData or dict/list)
            
        Returns:
            StageResult with data containing label_id field
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
                raise ValueError("Cannot route None data")
            
            # Step 3: Process list or single item
            if isinstance(actual_data, list):
                # Process each item in the list
                routed_items = []
                for item in actual_data:
                    routed_item = self._map_label_to_id(item)
                    routed_items.append(routed_item)
                routed_data = routed_items
            else:
                # Process single item
                routed_data = self._map_label_to_id(actual_data)
            
            # Step 4: Wrap routed data in ProcessedData
            processed_data = ProcessedData(
                data=routed_data,
                metadata={
                    **original_metadata,
                    "routed": True,
                    "routing_stage": self.name
                }
            )
            
            # Step 5: Return successful result
            return StageResult(
                success=True,
                processed_data=processed_data,
                stage_name=self.name
            )
            
        except Exception as e:
            return StageResult(
                success=False,
                error=f"Data routing failed: {str(e)}",
                stage_name=self.name
            )
    
    def _map_label_to_id(self, item: Dict) -> Dict:
        """
        Map label to integer ID for a single data item.
        
        Converts string labels to numeric IDs using the label mapping.
        If label is already numeric, validates it's in range [0, 30].
        Adds 'label_id' field to the data dictionary.
        
        Args:
            item: Data dictionary with 'label' field
            
        Returns:
            Data dictionary with 'label_id' field added
        """
        if not isinstance(item, dict):
            raise ValueError(f"Data item must be a dictionary, got {type(item).__name__}")
        
        # Create a copy to avoid modifying original
        routed_item = item.copy()
        
        # Check if label field exists
        if "label" not in routed_item:
            raise ValueError("Required field 'label' is missing")
        
        label = routed_item["label"]
        
        # Handle string labels - map to ID
        if isinstance(label, str):
            label_id = self._get_label_id(label)
            routed_item["label_id"] = label_id
        
        # Handle numeric labels - validate range
        elif isinstance(label, (int, float)):
            label_id = int(label)
            # Validate range [0, 30]
            if label_id < 0 or label_id > 30:
                raise ValueError(f"Label ID {label_id} is out of valid range [0, 30]")
            routed_item["label_id"] = label_id
        
        # Handle invalid label types
        else:
            raise ValueError(f"Label must be string or numeric, got {type(label).__name__}")
        
        return routed_item
    
    def _get_label_id(self, label_str: str) -> int:
        """
        Get integer ID for a string label using the label mapping.
        
        Args:
            label_str: String label (e.g., "debugging")
            
        Returns:
            Integer ID for the label
        """
        label_str = label_str.strip().lower()  # Normalize label
        
        # Use the label mapping from config, or default mapping
        if label_str in self.label_mapping:
            return self.label_mapping[label_str]
        else:
            raise ValueError(f"Unknown label '{label_str}'. Valid labels: {list(self.label_mapping.keys())}")
    
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate that labels can be routed/mapped to integer IDs.
        
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
        
        # Check if label mapping is configured
        if not self.label_mapping:
            warnings.append("Label mapping is empty - string labels cannot be mapped")
        
        # Check if data is empty
        if actual_data == [] or actual_data == {}:
            warnings.append("Data is empty - nothing to route")
            return ValidationResult(
                is_valid=True,  # Empty is valid but warning-worthy
                errors=errors,
                warnings=warnings
            )
        
        # Validate data structure
        if isinstance(actual_data, list):
            # Validate list of items
            if len(actual_data) == 0:
                warnings.append("List is empty")
            else:
                # Check each item for routing issues
                for i, item in enumerate(actual_data):
                    item_errors, item_warnings = self._validate_label_routing(item)
                    # Prefix errors/warnings with item index
                    for err in item_errors:
                        errors.append(f"Item at index {i}: {err}")
                    for warn in item_warnings:
                        warnings.append(f"Item at index {i}: {warn}")
        
        elif isinstance(actual_data, dict):
            # Validate single dictionary
            item_errors, item_warnings = self._validate_label_routing(actual_data)
            errors.extend(item_errors)
            warnings.extend(item_warnings)
        
        else:
            # Data is neither dict nor list
            errors.append(f"Data must be a dictionary or list, got {type(actual_data).__name__}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_label_routing(self, item: Dict) -> tuple[List[str], List[str]]:
        """
        Validate that a label can be routed/mapped (non-destructive).
        
        Args:
            item: Data item to validate
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check if item is a dictionary
        if not isinstance(item, dict):
            errors.append(f"Item must be a dictionary, got {type(item).__name__}")
            return errors, warnings
        
        # Check if label field exists
        if "label" not in item:
            errors.append("Required field 'label' is missing for routing")
            return errors, warnings
        
        label = item["label"]
        
        # Handle string labels - check if they can be mapped
        if isinstance(label, str):
            label_str = label.strip().lower()
            if label_str == "":
                errors.append("Label is empty string")
            elif not self.label_mapping:
                errors.append(f"Label '{label}' cannot be mapped - label mapping is empty")
            elif label_str not in self.label_mapping:
                valid_labels = list(self.label_mapping.keys())
                errors.append(f"Label '{label}' not found in mapping. Valid labels: {valid_labels}")
        
        # Handle numeric labels - check if in valid range
        elif isinstance(label, (int, float)):
            label_id = int(label)
            if label_id < 0 or label_id > 30:
                errors.append(f"Numeric label {label_id} is out of valid range [0, 30]")
            else:
                warnings.append(f"Label is already numeric ({label_id}) - routing will validate range only")
        
        # Handle invalid label types
        elif label is None:
            errors.append("Label is None - cannot be routed")
        else:
            errors.append(f"Label must be string or numeric, got {type(label).__name__}")
        
        return errors, warnings
    

class LabelValidationStage(PreprocessingStage):
    """
    Stage for label-specific validation.
    This stage validates label format, category, and numeric range.
    Runs after routing, so it validates the label_id field.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="label_validation", config=config)
        # Valid label ID range [0, 30] for 31 categories
        self.min_label_id = config.get("min_label_id", 0) if config else 0
        self.max_label_id = config.get("max_label_id", 30) if config else 30
    
    def get_dependencies(self) -> List[str]:
        return ["data_routing"]
    
    def process(self, data: Any) -> StageResult:
        """
        Validate labels by checking label_id format and range.
        
        Args:
            data: Data from previous stage (usually ProcessedData or dict/list)
            
        Returns:
            StageResult with validated data
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
                raise ValueError("Cannot validate None data")
            
            # Step 3: Validate list or single item
            if isinstance(actual_data, list):
                # Validate each item in the list
                validated_items = []
                for item in actual_data:
                    validated_item = self._validate_label(item)
                    validated_items.append(validated_item)
                validated_data = validated_items
            else:
                # Validate single item
                validated_data = self._validate_label(actual_data)
            
            # Step 4: Wrap validated data in ProcessedData
            processed_data = ProcessedData(
                data=validated_data,
                metadata={
                    **original_metadata,
                    "label_validated": True,
                    "label_validation_stage": self.name
                }
            )
            
            # Step 5: Return successful result
            return StageResult(
                success=True,
                processed_data=processed_data,
                stage_name=self.name
            )
            
        except Exception as e:
            return StageResult(
                success=False,
                error=f"Label validation failed: {str(e)}",
                stage_name=self.name
            )
    
    def _validate_label(self, item: Dict) -> Dict:
        """
        Validate label_id for a single data item.
        
        Since this runs after routing, it validates the label_id field
        that routing created.
        
        Args:
            item: Data dictionary with 'label_id' field
            
        Returns:
            Data dictionary (validated, unchanged if valid)
        """
        if not isinstance(item, dict):
            raise ValueError(f"Data item must be a dictionary, got {type(item).__name__}")
        
        # Check if label_id exists (created by routing stage)
        if "label_id" not in item:
            raise ValueError("Required field 'label_id' is missing (should be created by routing stage)")
        
        label_id = item["label_id"]
        
        # Validate label_id is numeric
        if not isinstance(label_id, (int, float)):
            raise ValueError(f"label_id must be numeric, got {type(label_id).__name__}")
        
        label_id = int(label_id)
        
        # Validate label_id is in valid range [0, 30]
        if label_id < self.min_label_id or label_id > self.max_label_id:
            raise ValueError(
                f"Label ID {label_id} is out of valid range "
                f"[{self.min_label_id}, {self.max_label_id}]"
            )
        
        # Return the item as-is (validation passed)
        return item
    
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate that labels can be validated (non-destructive).
        
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
            warnings.append("Data is empty - nothing to validate")
            return ValidationResult(
                is_valid=True,
                errors=errors,
                warnings=warnings
            )
        
        # Validate data structure
        if isinstance(actual_data, list):
            # Validate list of items
            if len(actual_data) == 0:
                warnings.append("List is empty")
            else:
                # Check each item for validation issues
                for i, item in enumerate(actual_data):
                    item_errors, item_warnings = self._validate_label_diagnostics(item)
                    # Prefix errors/warnings with item index
                    for err in item_errors:
                        errors.append(f"Item at index {i}: {err}")
                    for warn in item_warnings:
                        warnings.append(f"Item at index {i}: {warn}")
        
        elif isinstance(actual_data, dict):
            # Validate single dictionary
            item_errors, item_warnings = self._validate_label_diagnostics(actual_data)
            errors.extend(item_errors)
            warnings.extend(item_warnings)
        
        else:
            # Data is neither dict nor list
            errors.append(f"Data must be a dictionary or list, got {type(actual_data).__name__}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_label_diagnostics(self, item: Dict) -> tuple[List[str], List[str]]:
        """
        Validate label_id and return errors and warnings (non-destructive).
        
        Args:
            item: Data item to validate
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check if item is a dictionary
        if not isinstance(item, dict):
            errors.append(f"Item must be a dictionary, got {type(item).__name__}")
            return errors, warnings
        
        # Check if label_id exists
        if "label_id" not in item:
            errors.append("Required field 'label_id' is missing (should be created by routing stage)")
            return errors, warnings
        
        label_id = item["label_id"]
        
        # Validate label_id is numeric
        if not isinstance(label_id, (int, float)):
            errors.append(f"label_id must be numeric, got {type(label_id).__name__}")
            return errors, warnings
        
        label_id = int(label_id)
        
        # Validate label_id is in valid range
        if label_id < self.min_label_id or label_id > self.max_label_id:
            errors.append(
                f"Label ID {label_id} is out of valid range "
                f"[{self.min_label_id}, {self.max_label_id}]"
            )
        
        return errors, warnings


class DataTransformationStage(PreprocessingStage):
    """
    Stage for transforming data.
    This stage normalizes text and performs final transformations before model training.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="data_transformation", config=config)
        # Config options for text normalization
        self.normalize_lowercase = config.get("normalize_lowercase", True) if config else True
        self.normalize_trim = config.get("normalize_trim", True) if config else True
    
    def get_dependencies(self) -> List[str]:
        return ["label_validation"]
    
    def process(self, data: Any) -> StageResult:
        """
        Transform data by normalizing text field.
        
        Args:
            data: Data from previous stage (usually ProcessedData or dict/list)
            
        Returns:
            StageResult with transformed data
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
                raise ValueError("Cannot transform None data")
            
            # Step 3: Process list or single item
            if isinstance(actual_data, list):
                # Process each item in the list
                transformed_items = []
                for item in actual_data:
                    transformed_item = self._transform_single_item(item)
                    transformed_items.append(transformed_item)
                transformed_data = transformed_items
            else:
                # Process single item
                transformed_data = self._transform_single_item(actual_data)
            
            # Step 4: Wrap transformed data in ProcessedData
            processed_data = ProcessedData(
                data=transformed_data,
                metadata={
                    **original_metadata,
                    "transformed": True,
                    "transformation_stage": self.name
                }
            )
            
            # Step 5: Return successful result
            return StageResult(
                success=True,
                processed_data=processed_data,
                stage_name=self.name
            )
            
        except Exception as e:
            return StageResult(
                success=False,
                error=f"Data transformation failed: {str(e)}",
                stage_name=self.name
            )
    
    def _transform_single_item(self, item: Dict) -> Dict:
        """
        Transform a single data item by normalizing text field.
        
        Args:
            item: Data dictionary with 'text' field
            
        Returns:
            Data dictionary with normalized text field
        """
        if not isinstance(item, dict):
            raise ValueError(f"Data item must be a dictionary, got {type(item).__name__}")
        
        # Create a copy to avoid modifying original
        transformed_item = item.copy()
        
        # Normalize text field if it exists
        if "text" in transformed_item and isinstance(transformed_item["text"], str):
            text = transformed_item["text"]
            
            # Apply normalization based on config
            if self.normalize_trim:
                text = text.strip()
            
            if self.normalize_lowercase:
                text = text.lower()
            
            transformed_item["text"] = text
        
        return transformed_item
    
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate that data can be transformed.
        
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
            warnings.append("Data is empty - nothing to transform")
            return ValidationResult(
                is_valid=True,
                errors=errors,
                warnings=warnings
            )
        
        # Validate data structure
        if isinstance(actual_data, list):
            # Validate list of items
            if len(actual_data) == 0:
                warnings.append("List is empty")
            else:
                # Check each item for transformation issues
                for i, item in enumerate(actual_data):
                    item_errors, item_warnings = self._validate_transformation_diagnostics(item)
                    # Prefix errors/warnings with item index
                    for err in item_errors:
                        errors.append(f"Item at index {i}: {err}")
                    for warn in item_warnings:
                        warnings.append(f"Item at index {i}: {warn}")
        
        elif isinstance(actual_data, dict):
            # Validate single dictionary
            item_errors, item_warnings = self._validate_transformation_diagnostics(actual_data)
            errors.extend(item_errors)
            warnings.extend(item_warnings)
        
        else:
            # Data is neither dict nor list
            errors.append(f"Data must be a dictionary or list, got {type(actual_data).__name__}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_transformation_diagnostics(self, item: Dict) -> tuple[List[str], List[str]]:
        """
        Validate that an item can be transformed (non-destructive).
        
        Args:
            item: Data item to validate
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check if item is a dictionary
        if not isinstance(item, dict):
            errors.append(f"Item must be a dictionary, got {type(item).__name__}")
            return errors, warnings
        
        # Check if text field exists
        if "text" not in item:
            warnings.append("'text' field is missing - nothing to normalize")
            return errors, warnings
        
        # Check if text field is valid
        text = item["text"]
        if not isinstance(text, str):
            warnings.append(f"'text' field is not a string (got {type(text).__name__}) - cannot normalize")
        elif text.strip() == "":
            warnings.append("'text' field is empty or only whitespace")
        
        return errors, warnings

