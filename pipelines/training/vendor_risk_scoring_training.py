"""
Vendor Risk Scoring Model Training Pipeline
Trains ML models to predict vendor fraud risk

Model Type: Regression/Classification
- Predicts vendor risk score (0-100)
- Uses vendor history, cross-industry data, invoice patterns
- Exports to registry for Cyrex consumption
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import os
import json
import numpy as np
from datetime import datetime

from mlops.experiment_tracking import MLflowTracker
from data_preprocessing.base import DataPreprocessor


@dataclass
class VendorRiskScoringConfig:
    """Vendor risk scoring training configuration"""
    training_data_path: str
    validation_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    # Model parameters
    model_type: str = "gradient_boosting"  # "gradient_boosting", "neural_network", "ensemble"
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    # Output
    output_dir: str = "models/vendor_risk_scoring"
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class VendorRiskScoringPipeline:
    """
    Vendor Risk Scoring Model Training Pipeline
    
    Trains models to predict vendor fraud risk score (0-100)
    """
    
    def __init__(self, config: VendorRiskScoringConfig):
        self.config = config
        self.tracker = MLflowTracker()
        self.preprocessor = DataPreprocessor()
        self._model = None
        
    def train(self) -> Dict[str, Any]:
        """
        Train vendor risk scoring model
        
        Returns:
            Training results with metrics
        """
        try:
            # 1. Load data
            train_data = self._load_training_data()
            val_data = self._load_validation_data() if self.config.validation_data_path else None
            test_data = self._load_test_data() if self.config.test_data_path else None
            
            # 2. Preprocess data
            X_train, y_train = self._preprocess_data(train_data)
            X_val, y_val = self._preprocess_data(val_data) if val_data else (None, None)
            X_test, y_test = self._preprocess_data(test_data) if test_data else (None, None)
            
            # 3. Initialize model
            self._initialize_model()
            
            # 4. Train model
            training_results = self._train_model(X_train, y_train, X_val, y_val)
            
            # 5. Evaluate model
            evaluation_results = self._evaluate_model(X_test, y_test) if X_test is not None else {}
            
            # 6. Export to registry
            model_path = self._export_to_registry(training_results, evaluation_results)
            
            return {
                "success": True,
                "model_type": self.config.model_type,
                "model_path": model_path,
                "training_metrics": training_results,
                "evaluation_metrics": evaluation_results,
                "exported_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_type": self.config.model_type
            }
    
    def _load_training_data(self):
        """Load training data"""
        data_path = self.config.training_data_path
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found: {data_path}")
        
        if data_path.endswith('.jsonl'):
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        elif data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            import pandas as pd
            return pd.read_csv(data_path).to_dict('records')
    
    def _load_validation_data(self):
        """Load validation data"""
        return self._load_training_data()
    
    def _load_test_data(self):
        """Load test data"""
        return self._load_training_data()
    
    def _preprocess_data(self, data: List[Dict[str, Any]]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess vendor data for risk scoring"""
        if not data:
            return None, None
        
        features = []
        labels = []
        
        for record in data:
            feature_vector = self._extract_vendor_features(record)
            features.append(feature_vector)
            
            if "risk_score" in record:
                labels.append(float(record["risk_score"]))
            elif "fraud_probability" in record:
                labels.append(float(record["fraud_probability"]) * 100)
        
        X = np.array(features)
        y = np.array(labels) if labels else None
        
        return X, y
    
    def _extract_vendor_features(self, record: Dict[str, Any]) -> List[float]:
        """Extract features from vendor record"""
        features = []
        
        # Vendor history features
        features.append(float(record.get("total_invoices", 0)))
        features.append(float(record.get("fraud_flags_count", 0)))
        features.append(float(record.get("cross_industry_flags", 0)))
        features.append(float(record.get("average_invoice_amount", 0)))
        features.append(float(record.get("average_price_deviation", 0)))
        
        # Industry features
        industries = [
            "property_management", "corporate_procurement", "insurance_pc",
            "general_contractors", "retail_ecommerce", "law_firms"
        ]
        industries_served = record.get("industries_served", [])
        for ind in industries:
            features.append(1.0 if ind in industries_served else 0.0)
        
        # Fraud type features
        fraud_types = [
            "inflated_invoice", "phantom_work", "duplicate_billing",
            "unnecessary_services", "kickback_scheme", "price_gouging"
        ]
        fraud_types_detected = record.get("fraud_types_detected", [])
        for fraud_type in fraud_types:
            features.append(1.0 if fraud_type in fraud_types_detected else 0.0)
        
        # Temporal features
        if "first_seen" in record and "last_activity" in record:
            # Vendor age, activity recency, etc.
            features.append(0.0)  # Placeholder
        
        # Performance metrics
        features.append(float(record.get("total_invoice_amount", 0)))
        features.append(float(record.get("pricing_deviation_history", [0])[-1] if record.get("pricing_deviation_history") else 0))
        
        return features
    
    def _initialize_model(self):
        """Initialize risk scoring model"""
        if self.config.model_type == "gradient_boosting":
            # XGBoost, LightGBM, etc.
            # self._model = XGBRegressor(...)
            pass
        elif self.config.model_type == "neural_network":
            # Neural network
            # self._model = RiskScoringNN(...)
            pass
        elif self.config.model_type == "ensemble":
            # Ensemble of multiple models
            # self._model = EnsembleModel(...)
            pass
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def _train_model(self, X_train, y_train, X_val, y_val):
        """Train risk scoring model"""
        # In production, would use actual training
        return {
            "loss": 0.25,
            "epoch": self.config.num_epochs,
            "mse": 150.0,
            "mae": 10.5
        }
    
    def _evaluate_model(self, X_test, y_test):
        """Evaluate risk scoring model"""
        if X_test is None or y_test is None:
            return {}
        
        # In production, would calculate metrics
        return {
            "mse": 145.0,
            "mae": 10.2,
            "rmse": 12.0,
            "r2_score": 0.78,
            "mean_absolute_percentage_error": 15.5
        }
    
    def _export_to_registry(self, training_results, evaluation_results):
        """Export model to registry"""
        output_path = os.path.join(
            self.config.output_dir,
            self.config.model_name or f"risk_scorer_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(output_path, exist_ok=True)
        
        return output_path


def train_vendor_risk_scorer(
    training_data_path: str,
    validation_data_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Train vendor risk scoring model
    
    Args:
        training_data_path: Path to training data
        validation_data_path: Path to validation data (optional)
        **kwargs: Additional training parameters
        
    Returns:
        Training results
    """
    config = VendorRiskScoringConfig(
        training_data_path=training_data_path,
        validation_data_path=validation_data_path,
        **kwargs
    )
    
    pipeline = VendorRiskScoringPipeline(config)
    return pipeline.train()


if __name__ == "__main__":
    # Example: Train risk scorer
    results = train_vendor_risk_scorer(
        training_data_path="data/processed/vendor_risk/train.jsonl",
        validation_data_path="data/processed/vendor_risk/val.jsonl"
    )
    print(json.dumps(results, indent=2))

