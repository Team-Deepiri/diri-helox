"""
Dataset Monitoring and Logging Utilities
Provides monitoring, alerting, and logging for dataset versioning operations
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import statistics

from deepiri_modelkit.logging import get_logger

logger = get_logger("helox.dataset_monitoring")


class DatasetMonitor:
    """
    Monitors dataset versioning operations and provides insights.

    Features:
    - Operation metrics and performance tracking
    - Dataset health monitoring
    - Usage analytics
    - Alerting for data quality issues
    """

    def __init__(self, log_dir: str = "./logs/dataset_monitoring"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Metrics storage
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.alerts_file = self.log_dir / "alerts.jsonl"

        # In-memory metrics for quick access
        self.current_metrics = {
            "total_versions_created": 0,
            "total_datasets_tracked": 0,
            "average_version_creation_time": 0,
            "validation_errors_today": 0,
            "last_health_check": None,
            "storage_usage_bytes": 0
        }

        self._load_metrics()

    def log_version_creation(self, operation_data: Dict[str, Any]):
        """Log dataset version creation operation."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "version_creation",
            "dataset_name": operation_data.get("dataset_name"),
            "version": operation_data.get("version"),
            "dataset_type": operation_data.get("dataset_type"),
            "total_samples": operation_data.get("total_samples"),
            "file_count": operation_data.get("file_count"),
            "creation_time_seconds": operation_data.get("creation_time", 0),
            "change_type": operation_data.get("change_type"),
            "quality_score": operation_data.get("quality_score"),
            "storage_path": operation_data.get("storage_path"),
            "created_by": operation_data.get("created_by")
        }

        self._write_log_entry(self.metrics_file, log_entry)
        self.current_metrics["total_versions_created"] += 1

        logger.info("Version creation logged",
                   dataset=operation_data.get("dataset_name"),
                   version=operation_data.get("version"))

    def log_validation_result(self, validation_data: Dict[str, Any]):
        """Log dataset validation results."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "validation",
            "dataset_name": validation_data.get("dataset_name"),
            "version": validation_data.get("version"),
            "is_valid": validation_data.get("is_valid"),
            "quality_score": validation_data.get("quality_score"),
            "error_count": len(validation_data.get("errors", [])),
            "warning_count": len(validation_data.get("warnings", [])),
            "validation_time_seconds": validation_data.get("validation_time", 0)
        }

        self._write_log_entry(self.metrics_file, log_entry)

        if not validation_data.get("is_valid", True):
            self.current_metrics["validation_errors_today"] += 1

        # Check for alerts
        if validation_data.get("quality_score", 1.0) < 0.7:
            self._create_alert("low_quality_score", {
                "dataset_name": validation_data.get("dataset_name"),
                "version": validation_data.get("version"),
                "quality_score": validation_data.get("quality_score"),
                "errors": validation_data.get("errors", [])
            })

        logger.info("Validation result logged",
                   dataset=validation_data.get("dataset_name"),
                   valid=validation_data.get("is_valid"),
                   quality=validation_data.get("quality_score"))

    def log_training_usage(self, training_data: Dict[str, Any]):
        """Log dataset usage in training."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "training_usage",
            "dataset_name": training_data.get("dataset_name"),
            "dataset_version": training_data.get("dataset_version"),
            "model_name": training_data.get("model_name"),
            "training_duration_seconds": training_data.get("training_duration", 0),
            "final_loss": training_data.get("final_loss"),
            "experiment_id": training_data.get("experiment_id"),
            "output_model_path": training_data.get("output_model_path")
        }

        self._write_log_entry(self.metrics_file, log_entry)

        logger.info("Training usage logged",
                   dataset=training_data.get("dataset_name"),
                   version=training_data.get("dataset_version"),
                   model=training_data.get("model_name"))

    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_versions": self.current_metrics["total_versions_created"],
                "datasets_tracked": self.current_metrics["total_datasets_tracked"],
                "validation_errors_today": self.current_metrics["validation_errors_today"],
                "storage_usage_gb": self.current_metrics["storage_usage_bytes"] / (1024**3)
            },
            "performance": self._analyze_performance(),
            "quality_trends": self._analyze_quality_trends(),
            "alerts": self._get_recent_alerts(),
            "recommendations": self._generate_recommendations()
        }

        self.current_metrics["last_health_check"] = report["timestamp"]
        return report

    def get_usage_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get usage analytics for the specified period."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        analytics = {
            "period_days": days,
            "version_creations": [],
            "training_runs": [],
            "validation_runs": [],
            "popular_datasets": {},
            "quality_distribution": {}
        }

        # Read logs and filter by date
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_date = datetime.fromisoformat(entry["timestamp"])

                        if entry_date >= cutoff_date:
                            if entry["operation"] == "version_creation":
                                analytics["version_creations"].append(entry)
                                dataset = entry.get("dataset_name", "unknown")
                                analytics["popular_datasets"][dataset] = analytics["popular_datasets"].get(dataset, 0) + 1

                            elif entry["operation"] == "training_usage":
                                analytics["training_runs"].append(entry)

                            elif entry["operation"] == "validation":
                                analytics["validation_runs"].append(entry)
                                quality = entry.get("quality_score", 0)
                                quality_bucket = f"{int(quality * 10) / 10:.1f}"
                                analytics["quality_distribution"][quality_bucket] = analytics["quality_distribution"].get(quality_bucket, 0) + 1

                    except json.JSONDecodeError:
                        continue

        return analytics

    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze system performance metrics."""
        creation_times = []
        validation_times = []

        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry["operation"] == "version_creation":
                            if "creation_time_seconds" in entry:
                                creation_times.append(entry["creation_time_seconds"])
                        elif entry["operation"] == "validation":
                            if "validation_time_seconds" in entry:
                                validation_times.append(entry["validation_time_seconds"])
                    except json.JSONDecodeError:
                        continue

        return {
            "avg_version_creation_time": statistics.mean(creation_times) if creation_times else 0,
            "avg_validation_time": statistics.mean(validation_times) if validation_times else 0,
            "total_operations": len(creation_times) + len(validation_times),
            "creation_times": creation_times[-10:],  # Last 10
            "validation_times": validation_times[-10:]  # Last 10
        }

    def _analyze_quality_trends(self) -> Dict[str, Any]:
        """Analyze quality trends over time."""
        quality_scores = []

        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry["operation"] == "validation" and "quality_score" in entry:
                            quality_scores.append(entry["quality_score"])
                    except json.JSONDecodeError:
                        continue

        if not quality_scores:
            return {"trend": "insufficient_data"}

        recent_scores = quality_scores[-20:]  # Last 20 validations
        avg_quality = statistics.mean(recent_scores) if recent_scores else 0

        # Simple trend analysis
        if len(recent_scores) >= 10:
            first_half = recent_scores[:len(recent_scores)//2]
            second_half = recent_scores[len(recent_scores)//2:]

            first_avg = statistics.mean(first_half)
            second_avg = statistics.mean(second_half)

            if second_avg > first_avg + 0.05:
                trend = "improving"
            elif second_avg < first_avg - 0.05:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "average_quality": avg_quality,
            "trend": trend,
            "total_validations": len(quality_scores),
            "quality_distribution": {
                "excellent": len([s for s in quality_scores if s >= 0.9]),
                "good": len([s for s in quality_scores if 0.7 <= s < 0.9]),
                "fair": len([s for s in quality_scores if 0.5 <= s < 0.7]),
                "poor": len([s for s in quality_scores if s < 0.5])
            }
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on current state."""
        recommendations = []

        # Check for frequent validation errors
        if self.current_metrics["validation_errors_today"] > 5:
            recommendations.append("High validation error rate detected. Review data quality processes.")

        # Check quality trends
        quality_analysis = self._analyze_quality_trends()
        if quality_analysis.get("trend") == "declining":
            recommendations.append("Dataset quality is declining. Consider reviewing data sources and annotation processes.")

        # Check performance
        performance = self._analyze_performance()
        if performance.get("avg_version_creation_time", 0) > 300:  # 5 minutes
            recommendations.append("Version creation is slow. Consider optimizing storage or processing.")

        # General recommendations
        if self.current_metrics["total_versions_created"] == 0:
            recommendations.append("No dataset versions created yet. Start versioning your datasets for reproducibility.")

        if not recommendations:
            recommendations.append("System operating normally. Continue regular monitoring.")

        return recommendations

    def _create_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Create an alert for monitoring."""
        alert_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "alert_type": alert_type,
            "severity": "warning",  # Could be "info", "warning", "error"
            "data": alert_data,
            "resolved": False
        }

        self._write_log_entry(self.alerts_file, alert_entry)

        logger.warning("Alert created",
                      type=alert_type,
                      data=alert_data)

    def _get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        alerts = []
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        if self.alerts_file.exists():
            with open(self.alerts_file, 'r') as f:
                for line in f:
                    try:
                        alert = json.loads(line.strip())
                        alert_time = datetime.fromisoformat(alert["timestamp"])
                        if alert_time >= cutoff_time:
                            alerts.append(alert)
                    except json.JSONDecodeError:
                        continue

        return alerts[-10:]  # Return last 10 alerts

    def _write_log_entry(self, log_file: Path, entry: Dict[str, Any]):
        """Write a log entry to file."""
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def _load_metrics(self):
        """Load current metrics from log files."""
        # This is a simplified version - in production you'd aggregate from logs
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Count operations from logs
                        version_count = sum(1 for line in lines if '"operation": "version_creation"' in line)
                        validation_count = sum(1 for line in lines if '"operation": "validation"' in line and '"is_valid": false' in line)

                        self.current_metrics["total_versions_created"] = version_count
                        self.current_metrics["validation_errors_today"] = validation_count
            except Exception as e:
                logger.warning("Failed to load metrics from log", error=str(e))


# Convenience functions
def log_version_creation(dataset_name: str, version: str, **kwargs):
    """Convenience function to log version creation."""
    monitor = DatasetMonitor()
    monitor.log_version_creation({
        "dataset_name": dataset_name,
        "version": version,
        **kwargs
    })

def log_validation_result(dataset_name: str, version: str, **kwargs):
    """Convenience function to log validation results."""
    monitor = DatasetMonitor()
    monitor.log_validation_result({
        "dataset_name": dataset_name,
        "version": version,
        **kwargs
    })

def get_health_report():
    """Get current health report."""
    monitor = DatasetMonitor()
    return monitor.get_health_report()

def get_usage_analytics(days: int = 30):
    """Get usage analytics."""
    monitor = DatasetMonitor()
    return monitor.get_usage_analytics(days)


# Example usage
if __name__ == "__main__":
    # Test the monitoring system
    monitor = DatasetMonitor()

    # Log some test data
    monitor.log_version_creation({
        "dataset_name": "test_dataset",
        "version": "1.0.0",
        "total_samples": 1000,
        "creation_time": 45.2
    })

    monitor.log_validation_result({
        "dataset_name": "test_dataset",
        "version": "1.0.0",
        "is_valid": True,
        "quality_score": 0.95,
        "validation_time": 12.3
    })

    # Get reports
    health = monitor.get_health_report()
    analytics = monitor.get_usage_analytics(days=7)

    print("Health Report:")
    print(json.dumps(health, indent=2))

    print("\nUsage Analytics:")
    print(json.dumps(analytics, indent=2))
