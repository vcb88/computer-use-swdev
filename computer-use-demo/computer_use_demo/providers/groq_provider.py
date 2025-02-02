"""
Реализация провайдера Groq
"""

from typing import Any, Dict, List, Optional
import tiktoken
from groq import AsyncGroq

from .base import AIProvider

class GroqProvider(AIProvider):
    """Провайдер для работы с Groq API"""
    
    def __init__(self, api_key: str):
        """
        Инициализация провайдера
        
        Args:
            api_key: API ключ Groq
        """
        self.client = AsyncGroq(api_key=api_key)
        # Используем токенизатор llama2, так как Groq использует модели на его основе
        self.encoding = tiktoken.encoding_for_model("gpt-4")  # Временно используем gpt-4 токенизатор
    
    def get_default_model(self) -> str:
        return "mixtral-8x7b-32768"
    
    def get_token_count(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def prepare_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Конвертация инструментов в формат Groq (OpenAI-совместимый)"""
        prepared_tools = []
        for tool in tools:
            parameters = {
                "type": "object",
                "properties": tool["parameters"]["properties"],
                "required": tool["parameters"].get("required", [])
            }
            
            prepared_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": parameters
                }
            })
        return prepared_tools
    
    def prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Конвертация сообщений в формат Groq (OpenAI-совместимый)"""
        prepared_messages = []
        for message in messages:
            content = message["content"]
            
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if block["type"] == "text":
                        text_parts.append(block["text"])
                    elif block["type"] == "tool_result":
                        if isinstance(block["content"], str):
                            text_parts.append(f"Tool result: {block['content']}")
                        elif isinstance(block["content"], list):
                            for content_block in block["content"]:
                                if content_block["type"] == "text":
                                    text_parts.append(f"Tool result: {content_block['text']}")
                content = "\n".join(text_parts)
            
            prepared_messages.append({
                "role": message["role"],
                "content": content
            })
        
        return prepared_messages
    
    async def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """Получение завершения от Groq"""
        prepared_messages = self.prepare_messages(messages)
        prepared_tools = self.prepare_tools(tools)
        
        response = await self.client.chat.completions.create(
            model=model,
            messages=prepared_messages,
            tools=prepared_tools,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return self.parse_response(response)
    
    def parse_response(self, response: Any) -> Dict[str, Any]:
        """Преобразование ответа Groq в стандартный формат"""
        content = []
        
        message = response.choices[0].message
        
        # Обработка обычного текстового ответа
        if message.content:
            content.append({
                "type": "text",
                "text": message.content
            })
        
        # Обработка вызовов инструментов
        if tool_calls := getattr(message, 'tool_calls', None):
            for tool_call in tool_calls:
                content.append({
                    "type": "tool_use",
                    "name": tool_call.function.name,
                    "id": tool_call.id,
                    "input": tool_call.function.arguments
                })
        
        return {
            "role": "assistant",
            "content": content
        }