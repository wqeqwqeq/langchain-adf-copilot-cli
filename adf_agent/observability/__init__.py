"""
Observability module for ADF Agent.

Provides MLflow tracking and other observability integrations.
"""

from .mlflow_setup import setup_mlflow_tracking

__all__ = ["setup_mlflow_tracking"]
