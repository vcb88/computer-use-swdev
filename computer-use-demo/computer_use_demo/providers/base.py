"""
Базовые классы для провайдеров AI API
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class AIProvider(ABC):
    """Абстрактный базовый класс для провайдеров AI"""
    
    @abstractmethod
    async def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Получить завершение от модели
        
        Args:
            messages: Список сообщений в формате провайдера
            tools: Список инструментов в формате провайдера
            model: Имя модели
            max_tokens: Максимальное количество токенов в ответе
            **kwargs: Дополнительные параметры специфичные для провайдера
            
        Returns:
            Dict[str, Any]: Ответ от модели в стандартизированном формате
        """
        pass

    @abstractmethod
    def prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Конвертировать сообщения в формат провайдера
        
        Args:
            messages: Список сообщений в стандартном формате
            
        Returns:
            List[Dict[str, Any]]: Сообщения в формате провайдера
        """
        pass

    @abstractmethod
    def prepare_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Конвертировать инструменты в формат провайдера
        
        Args:
            tools: Список инструментов в стандартном формате
            
        Returns:
            List[Dict[str, Any]]: Инструменты в формате провайдера
        """
        pass

    @abstractmethod
    def parse_response(self, response: Any) -> Dict[str, Any]:
        """
        Преобразовать ответ провайдера в стандартный формат
        
        Args:
            response: Ответ от API провайдера
            
        Returns:
            Dict[str, Any]: Ответ в стандартизированном формате
        """
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """
        Получить количество токенов в тексте
        
        Args:
            text: Текст для подсчета токенов
            
        Returns:
            int: Количество токенов
        """
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """
        Получить название модели по умолчанию
        
        Returns:
            str: Название модели по умолчанию для данного провайдера
        """
        pass