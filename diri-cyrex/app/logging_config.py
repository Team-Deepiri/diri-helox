"""
Comprehensive logging configuration for the Python backend.
"""
import logging
import structlog
import sys
from pathlib import Path
from pythonjsonlogger import jsonlogger
from typing import Any, Dict


def configure_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class RequestLogger:
    """Middleware for logging HTTP requests with structured data."""
    
    def __init__(self, logger_name: str = "cyrex.requests"):
        self.logger = get_logger(logger_name)
    
    def log_request(self, request_id: str, method: str, path: str, 
                   status_code: int, duration_ms: float, 
                   user_id: str = None, **kwargs) -> None:
        """
        Log HTTP request details.
        
        Args:
            request_id: Unique request identifier
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration_ms: Request duration in milliseconds
            user_id: Optional user identifier
            **kwargs: Additional context data
        """
        self.logger.info(
            "HTTP request completed",
            request_id=request_id,
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            user_id=user_id,
            **kwargs
        )


class ErrorLogger:
    """Specialized logger for error tracking."""
    
    def __init__(self, logger_name: str = "cyrex.errors"):
        self.logger = get_logger(logger_name)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """
        Log error with context.
        
        Args:
            error: Exception instance
            context: Additional context data
        """
        self.logger.error(
            "Error occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {}
        )
    
    def log_api_error(self, error: Exception, request_id: str, 
                     endpoint: str, user_id: str = None) -> None:
        """
        Log API-specific errors.
        
        Args:
            error: Exception instance
            request_id: Request identifier
            endpoint: API endpoint
            user_id: Optional user identifier
        """
        self.logger.error(
            "API error occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            request_id=request_id,
            endpoint=endpoint,
            user_id=user_id
        )

