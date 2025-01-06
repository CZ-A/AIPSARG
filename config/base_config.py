# trading_bot/config/base_config.py
from abc import ABC, abstractmethod
from typing import Dict

class BaseConfig(ABC):
    """Abstract base class for all configurations."""

    @abstractmethod
    def load_config(self) -> Dict:
        """Loads the application configuration.

        Returns:
            Dict: Application configuration.
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validates the configuration.
        
        Returns:
            bool: True if the config is valid, False otherwise
        """
        pass
