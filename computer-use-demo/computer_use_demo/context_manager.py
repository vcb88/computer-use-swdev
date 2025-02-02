"""
Менеджер контекста для управления сообщениями в диалоге с Claude.
Обеспечивает сохранение базового контекста и управление размером контекстного окна.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import tiktoken
import json

@dataclass
class MessageContext:
    """Структура для хранения сообщения и его метаданных"""
    content: dict  # Оригинальное сообщение
    tokens: int  # Количество токенов
    timestamp: datetime
    is_core: bool = False  # Флаг для базового контекста

class ContextManager:
    def __init__(self, max_tokens: int = 40000, core_context: Optional[List[dict]] = None):
        """
        Инициализация менеджера контекста
        
        Args:
            max_tokens: Максимальное количество токенов в контексте
            core_context: Базовый контекст, который должен сохраняться всегда
        """
        self.max_tokens = max_tokens
        self.current_tokens = 0
        self.messages = deque()
        self.core_context = []
        self.core_tokens = 0
        self.encoding = tiktoken.encoding_for_model("cl100k_base")  # Используем токенизатор, похожий на Claude
        
        # Инициализация базового контекста
        if core_context:
            for msg in core_context:
                self.add_core_message(msg)
    
    def _count_tokens(self, message: dict) -> int:
        """
        Подсчет токенов в сообщении с использованием tiktoken
        
        Args:
            message: Сообщение для подсчета токенов
            
        Returns:
            int: Количество токенов в сообщении
        """
        # Преобразуем сообщение в строку для подсчета токенов
        if isinstance(message.get('content'), list):
            # Для сложных сообщений с несколькими блоками контента
            tokens = 0
            for block in message['content']:
                if isinstance(block, dict):
                    # Для текстовых блоков и результатов инструментов
                    if block.get('type') == 'text':
                        tokens += len(self.encoding.encode(block.get('text', '')))
                    elif block.get('type') == 'tool_use':
                        # Подсчитываем токены для инструментов
                        tool_str = json.dumps(block, ensure_ascii=False)
                        tokens += len(self.encoding.encode(tool_str))
                    elif block.get('type') == 'tool_result':
                        if isinstance(block.get('content'), str):
                            tokens += len(self.encoding.encode(block['content']))
                        else:
                            tool_str = json.dumps(block, ensure_ascii=False)
                            tokens += len(self.encoding.encode(tool_str))
                elif isinstance(block, str):
                    tokens += len(self.encoding.encode(block))
            return tokens
        else:
            # Для простых текстовых сообщений
            text = str(message.get('content', ''))
            return len(self.encoding.encode(text))
    
    def add_core_message(self, message: dict) -> None:
        """
        Добавление сообщения в базовый контекст
        
        Args:
            message: Сообщение для добавления в базовый контекст
        """
        tokens = self._count_tokens(message)
        ctx = MessageContext(
            content=message,
            tokens=tokens,
            timestamp=datetime.now(),
            is_core=True
        )
        self.core_context.append(ctx)
        self.core_tokens += tokens
        
    def add_message(self, message: dict) -> List[MessageContext]:
        """
        Добавление нового сообщения с поддержанием лимита токенов
        
        Args:
            message: Новое сообщение для добавления
            
        Returns:
            List[MessageContext]: Список удаленных сообщений
        """
        tokens = self._count_tokens(message)
        removed = []
        
        # Проверяем, нужно ли освободить место для нового сообщения
        available_tokens = self.max_tokens - self.core_tokens
        current_dynamic_tokens = sum(m.tokens for m in self.messages)
        
        # Удаляем старые сообщения, если нужно освободить место
        while current_dynamic_tokens + tokens > available_tokens and self.messages:
            removed_msg = self.messages.popleft()  # Удаляем самое старое сообщение
            current_dynamic_tokens -= removed_msg.tokens
            removed.append(removed_msg)
            
        # Добавляем новое сообщение
        ctx = MessageContext(
            content=message,
            tokens=tokens,
            timestamp=datetime.now()
        )
        self.messages.append(ctx)
        
        return removed
    
    def get_current_context(self) -> List[dict]:
        """
        Получение текущего контекста для API запроса
        
        Returns:
            List[dict]: Список всех актуальных сообщений
        """
        return [msg.content for msg in self.core_context + list(self.messages)]
    
    def get_token_stats(self) -> Dict[str, int]:
        """
        Получение статистики по токенам
        
        Returns:
            Dict[str, int]: Статистика использования токенов
        """
        dynamic_tokens = sum(m.tokens for m in self.messages)
        return {
            'core_tokens': self.core_tokens,
            'dynamic_tokens': dynamic_tokens,
            'total_tokens': self.core_tokens + dynamic_tokens,
            'available_tokens': self.max_tokens - (self.core_tokens + dynamic_tokens)
        }
    
    def clear_dynamic_context(self) -> List[MessageContext]:
        """
        Очистка динамического контекста (всех сообщений кроме базовых)
        
        Returns:
            List[MessageContext]: Список удаленных сообщений
        """
        removed = list(self.messages)
        self.messages.clear()
        return removed