"""
Реализация провайдера OpenAI
"""

from typing import Any, Dict, List, Optional
import tiktoken
from openai import OpenAI, AsyncOpenAI

from .base import AIProvider

class OpenAIProvider(AIProvider):
    """Провайдер для работы с OpenAI API"""
    
    def __init__(self, api_key: str):
        """
        Инициализация провайдера
        
        Args:
            api_key: API ключ OpenAI
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.encoding = tiktoken.encoding_for_model("gpt-4")
    
    def get_default_model(self) -> str:
        return "gpt-4-turbo-preview"
    
    def get_token_count(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def prepare_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Конвертация инструментов в формат OpenAI function calling"""
        openai_tools = []
        for tool in tools:
            # Конвертируем схему параметров в формат OpenAI
            parameters = {
                "type": "object",
                "properties": tool["parameters"]["properties"],
                "required": tool["parameters"].get("required", [])
            }
            
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": parameters
                }
            })
        return openai_tools
    
    def prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Конвертация сообщений в формат OpenAI"""
        prepared_messages = []
        for message in messages:
            content = message["content"]
            
            # Обработка сообщений с множественными блоками контента
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
        """Получение завершения от OpenAI"""
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
        """Преобразование ответа OpenAI в стандартный формат"""
        content = []
        
        message = response.choices[0].message
        
        # Обработка обычного текстового ответа
        if message.content:
            content.append({
                "type": "text",
                "text": message.content
            })
        
        # Обработка вызовов инструментов
        if tool_calls := message.tool_calls:
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