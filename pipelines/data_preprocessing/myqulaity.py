from typing import  Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
import pandas as pd 




@dataclass
class QualityMetrics:
    dimension: str
    metric_name : str
    value : float
    threshold : float
    result : bool
    details : Dict[str, Any]
    
@dataclass
class QulaityReport:
    timestamp: datetime
    dataset_id : str
    dimension_score : Dict[str, Any]
    overall_score : float
    summary : Dict[str, Any]
    recommendations : List[str]
    metrics : List[QualityMetrics]
    
    def to_dict(self):
        return {
            "dataset_id" : self.dataset_id,
            "timestamp" : self.timestamp,
            "dimension_score" : self.dimension_score,
            "overall_score" : self.overall_score,
            "summary" : self.summary,
            "recommendations" : self.recommendations,
            "metrics" : [
                
                
                {
                    "dimensions" : m.dimension,
                    "overall_score" : m.overall_score,
                    "summary" : m.summary,
                    "recommendations" : m.recommendations,
                    "metrics" : m.metrics,
                    "details" : m.details,
                } 
                for m in self.metrics
                ]
        }
        
class StatisticalValidator:
    
    @staticmethod
    def validate_using_iqr(data: np.ndarray, multiplier :  float = 1.5):
        q1, q3 = np.percentile(data, [25,75])
        iqr = q3-q1
        lowerbound = q1-multiplier*iqr
        upperbound = q3+multiplier*iqr
        outliers = (data < lowerbound) | (data > upperbound)
        details = {
            "q1" : float(q1),
            "q3" : float(q3),
            "iqr" : float(iqr),
            "lowerbound" : float(lowerbound),
            "upperbound" : float(upperbound),
            "outlier_count" : int(outliers.sum()),
        }
    
    
    @staticmethod
    def validating_using_zscore(data: np.ndarray, threashold : float = 3.0):
        z_score = np.abs(stats.zscore(data))
        outliers = z_score > threashold
        details = {
            "z_score" : float(z_score),
            "threashold" : float(threashold),
            "outlier_count" : int(outliers.sum()),
        }
        return outliers, details
    
    @staticmethod
    def validation_using_isolation_forest(
        data : np.ndarray,
        random_state : int = 42,
        contamination: float = 0.1
    ):
        if data.dim == 1:
            data_2d = data.reshape(-1,1)
        else:
            data_2d = data
            
        clf = IsolationForest(contamination=contamination, random_state=random_state)
        outliers_predictions = clf.predict(data_2d)
        outliers = outliers_predictions == -1
        details = {
            "contamination" : float(contamination),
            "outlier_count" : int(outliers.sum()),
            "outlier_percentage" : float(outliers.sum() / len(data) * 100),
            "scores" : clf.score_samples(data_2d).tolist(),
        }
        return outliers, details
    
    
    class QualityChecker(StatisticalValidator):
        def __init__(self, data: Union[pd.DataFrame, List[Dict], Dict],
                    completeness_threshold: float = 0.95,
                    consistency_threshold: float = 0.90,
                    validity_threshold: float = 1.0,
                    uniqueness_threshold: float = 0.98,
                    accuracy_threshold: float = 0.90,
                    integrity_threshold: float = 1.0):  
            
            self.completeness_threshold = completeness_threshold
            self.consistency_threshold = consistency_threshold
            self.validity_threshold = validity_threshold
            self.uniqueness_threshold = uniqueness_threshold
            self.accuracy_threshold = accuracy_threshold
            self.integrity_threshold = integrity_threshold
            self.statvalidator  = StatisticalValidator()
            
            
            metrics = []
            df = self.data_to_dataframe(data)
            
            completeness_score = self.check_completeness_score(df)
            metrics.extend(completeness_score)
            
            consistensy_score = self.check_consistecy_score(df)
            metrics.extend(consistensy_score)
            
            validity_score = self.check_validity_score(df)
            metrics.extend(validity_score)
            
            uniqueness_score = self.check_uniqueness_score(df)
            metrics.extend(uniqueness_score)
            
            accuracy_score = self.check_accuracy_score(df)
            metrics.extend(accuracy_score)
            
            
            
            dimensios_score = self.calculate_dimension_scores(metrics)
            
            
        def data_to_dataframe(data: Union[pd.DataFrame, List[Dict]], Dict):
            if isinstance(data, pd.DataFrame):
                return data
            
            elif isinstance(data, list):
                
                if len(data) == 0:
                    raise ValueError("Data is empty")
                
                if isinstance(data[0], dict):
                    return pd.DataFrame(data)
                else:
                    raise ValueError("Data must be a list of dictionaries")
            
            elif isinstance(data, dict):
                return pd.DataFrame([data])
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")    
            
            
            
        def check_completeness_level(data: pd.DataFrame):
            
            metrics = []
            
            if data is None:
                metrics.append(QualityMetrics(
                    dimension = "completeness",
                    metric_name = "empty_dataset",
                    value = 0.0,
                    threshold = self.completeness_threshold,
                    result = False,
                    details = {
                        "message" : "Dataset is empty"
                    }
                )) 
                return metrics
            
            total_cells = data.size
            missing_cells = data.isnull().sum()
            
            completesness_score = 1 - (missing_cells/total_cells)
            
            metrics.append(QualityMetrics(
                dimension:"completeness",
                metrics_name:"overall_completeness",
                threshold_level : self.completeness_score,
                result : completenss_score>
            ))
        