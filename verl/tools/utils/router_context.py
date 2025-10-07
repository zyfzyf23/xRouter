# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Any
import threading


class ConversationContextManager:
    """
    Manages conversation histories across different tools based on request_id.
    Thread-safe singleton for sharing context between tools.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConversationContextManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not getattr(self, '_initialized', False):
            self._contexts = {}  # request_id -> list of interactions
            self._lock = threading.Lock()
            self._initialized = True
    
    def clear_context(self, request_id: str) -> None:
        """Clear all conversation context for a request_id."""
        with self._lock:
            if request_id in self._contexts:
                del self._contexts[request_id]
    
    def get_context(self, request_id: str) -> list:
        """Get conversation context list for a request_id."""
        with self._lock:
            if request_id not in self._contexts:
                self._contexts[request_id] = []
            return self._contexts[request_id]
    
    def append_interaction(self, request_id: str, route_to: str, arguments: Dict[str, Any], feedback: str = "", metadata: Dict[str, Any] = {}, tool_id: str = "") -> None:
        """Append a new interaction to the conversation context."""
        with self._lock:
            if request_id not in self._contexts:
                self._contexts[request_id] = []
            self._contexts[request_id].append({
                "route_to": route_to,
                "arguments": arguments,
                "feedback": feedback,
                "metadata": metadata,
                "tool_id": tool_id
            })


# Global singleton instance
conversation_context = ConversationContextManager() 