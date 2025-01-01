"""Chat history logging functionality."""

import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union
from anthropic.types.beta import BetaContentBlockParam

def sanitize_filename(s: str) -> str:
    """Convert string to safe filename."""
    return "".join(c if c.isalnum() else "_" for c in s)

class ChatLogger:
    """Logger for chat history and related data."""
    
    def __init__(self, base_dir: Union[str, Path] = "~/.anthropic/chat_logs"):
        self.base_dir = Path(base_dir).expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[Path] = None
        self.current_log_file: Optional[Path] = None
        self.image_counter = 0
    
    def start_session(self) -> None:
        """Start a new chat session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session = self.base_dir / timestamp
        self.current_session.mkdir(parents=True, exist_ok=True)
        self.current_log_file = self.current_session / "chat.log"
        self.image_counter = 0
        
        # Create empty log file
        self.current_log_file.touch()
    
    def _ensure_session(self) -> None:
        """Ensure a session is started."""
        if self.current_session is None or self.current_log_file is None:
            self.start_session()
    
    def log_message(self, role: str, content: Union[str, BetaContentBlockParam, list[Any]], 
                   metadata: Optional[dict] = None) -> None:
        """Log a chat message with optional metadata."""
        self._ensure_session()
        assert self.current_log_file is not None
        
        timestamp = datetime.now().isoformat()
        
        # Format timestamp for display
        display_timestamp = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        # Process content based on type
        if isinstance(content, str):
            processed_content = f"[{display_timestamp}]\n{content}"
        elif isinstance(content, dict):
            if content.get("type") == "text":
                processed_content = f"[{display_timestamp}]\n{content['text']}"
            elif content.get("type") == "tool_use":
                processed_content = f"[{display_timestamp}]\nTool Use: {content['name']}\nInput: {content['input']}"
            else:
                processed_content = json.dumps(content)
        elif isinstance(content, list):
            # Process list of content blocks
            processed_blocks = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        processed_blocks.append(f"[{display_timestamp}]\n{block['text']}")
                    elif block.get("type") == "tool_use":
                        processed_blocks.append(
                            f"[{display_timestamp}]\nTool Use: {block['name']}\nInput: {block['input']}"
                        )
                    elif block.get("type") == "tool_result":
                        if "error" in block:
                            processed_blocks.append(f"[{display_timestamp}]\nTool Error: {block['error']}")
                        else:
                            processed_blocks.append(f"[{display_timestamp}]\nTool Result: {block.get('content', '')}")
                    else:
                        processed_blocks.append(f"[{display_timestamp}]\n{json.dumps(block)}")
                else:
                    processed_blocks.append(f"[{display_timestamp}]\n{str(block)}")
            processed_content = "\n".join(processed_blocks)
        else:
            processed_content = f"[{display_timestamp}]\n{str(content)}"
        
        # Prepare log entry
        log_entry = {
            "timestamp": timestamp,
            "role": role,
            "content": processed_content
        }
        if metadata:
            log_entry["metadata"] = metadata
            
        # Write to log file
        with self.current_log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def save_image(self, base64_image: str, description: str = "") -> None:
        """Save base64 encoded image to the session directory."""
        self._ensure_session()
        assert self.current_session is not None
        
        self.image_counter += 1
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"image_{timestamp}_{self.image_counter}.png"
        
        # Save image
        image_path = self.current_session / filename
        image_data = base64.b64decode(base64_image)
        image_path.write_bytes(image_data)
        
        # Log image save
        if description:
            self.log_message(
                "system", 
                f"Saved image: {filename} - {description}",
                {"image_path": str(image_path)}
            )
    
    def save_tool_result(self, result: Any, tool_id: str) -> None:
        """Save tool execution result."""
        self._ensure_session()
        
        # Extract relevant information
        metadata = {
            "tool_id": tool_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Process different result types
        if hasattr(result, "error") and result.error:
            self.log_message("tool", f"Error: {result.error}", metadata)
        
        if hasattr(result, "output") and result.output:
            self.log_message("tool", result.output, metadata)
        
        if hasattr(result, "base64_image") and result.base64_image:
            self.save_image(
                result.base64_image,
                f"Tool result image from {tool_id}"
            )
    
    def export_session(self, format: str = "txt") -> Path:
        """Export current session in specified format."""
        self._ensure_session()
        assert self.current_session is not None
        assert self.current_log_file is not None
        
        if format == "txt":
            output_file = self.current_session / "session_export.txt"
            with self.current_log_file.open("r", encoding="utf-8") as f_in:
                with output_file.open("w", encoding="utf-8") as f_out:
                    for line in f_in:
                        entry = json.loads(line)
                        timestamp = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                        f_out.write(f"[{timestamp}] {entry['role'].upper()}:\n{entry['content']}\n\n")
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return output_file
    
    def get_session_path(self) -> Optional[Path]:
        """Get current session path."""
        return self.current_session