import json
import time
from typing import Dict

from magic_llm.engine.amazon_adapters.base_provider import AmazonBaseProvider
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel


class ProviderAmazonAnthropic(AmazonBaseProvider):
    """
    Provider for Anthropic Claude models via Amazon Bedrock.
    """
    
    def prepare_data(self, chat: ModelChat, **kwargs) -> str:
        """
        Prepare the request data for Anthropic Claude models.
        
        Args:
            chat: The chat model containing messages
            **kwargs: Additional parameters for the request
            
        Returns:
            A JSON string containing the request body
        """
        body = json.dumps({
            "prompt": chat.generic_chat(format='claude'),
            "max_tokens_to_sample": kwargs.get('max_tokens_to_sample', 1024),
            "temperature": kwargs.get('temperature', 0.5),
            "top_k": kwargs.get('top_k', 250),
            "top_p": kwargs.get('top_p', 1),
            "stop_sequences": kwargs.get('stop_sequences', ["\n\nHuman:"]),
            # "anthropic_version": "bedrock-2023-05-31"
        })
        
        return body
    
    def process_response(self, response: dict) -> Dict:
        """
        Process the response from Anthropic Claude models.
        
        Args:
            response: The response from the model
            
        Returns:
            A dictionary containing the processed response
        """
        return {
            'content': response['completion'],
            'role': 'assistant',
            'usage': UsageModel(
                prompt_tokens=len(response.get('prompt', '')),
                completion_tokens=len(response['completion']),
                total_tokens=len(response.get('prompt', '')) + len(response['completion'])
            )
        }
    
    def format_event_to_chunk(self, event: dict) -> ChatCompletionModel:
        """
        Format a streaming event from Anthropic Claude models to a ChatCompletionModel.
        
        Args:
            event: The event from the streaming response
            
        Returns:
            A ChatCompletionModel containing the formatted event
        """
        chunk = {
            'id': '1',
            'choices':
                [{
                    'delta':
                        {
                            'content': event['completion'],
                            'role': None
                        },
                    'finish_reason': 'stop' if event['stop_reason'] else None,
                    'index': 0
                }],
            'created': int(time.time()),
            'model': self.model,
            'object': 'chat.completion.chunk'
        }
        
        return ChatCompletionModel(**chunk)