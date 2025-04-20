import json
import time
from typing import Dict

from magic_llm.engine.amazon_adapters.base_provider import AmazonBaseProvider
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel


class ProviderAmazonMeta(AmazonBaseProvider):
    """
    Provider for Meta Llama models via Amazon Bedrock.
    """
    
    def prepare_data(self, chat: ModelChat, **kwargs) -> str:
        """
        Prepare the request data for Meta Llama models.
        
        Args:
            chat: The chat model containing messages
            **kwargs: Additional parameters for the request
            
        Returns:
            A JSON string containing the request body
        """
        body = json.dumps({
            "prompt": chat.generic_chat(format='llama2'),
            "max_gen_len": kwargs.get('max_gen_len', 1024),
            "temperature": kwargs.get('temperature', 0.2),
            "top_p": kwargs.get('top_p', 1),
            # "stop_sequences": kwargs.get('stop_sequences', ["[/INST]"]),
        })
        
        return body
    
    def process_response(self, response: dict) -> Dict:
        """
        Process the response from Meta Llama models.
        
        Args:
            response: The response from the model
            
        Returns:
            A dictionary containing the processed response
        """
        return {
            'content': response['generation'],
            'role': 'assistant',
            'usage': UsageModel(
                prompt_tokens=response['prompt_token_count'],
                completion_tokens=response['generation_token_count'],
                total_tokens=response['prompt_token_count'] + response['generation_token_count']
            )
        }
    
    def format_event_to_chunk(self, event: dict) -> ChatCompletionModel:
        """
        Format a streaming event from Meta Llama models to a ChatCompletionModel.
        
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
                            'content': event['generation'],
                            'role': None
                        },
                    'finish_reason': 'stop' if event.get('stop_reason') else None,
                    'index': 0
                }],
            'created': int(time.time()),
            'model': self.model,
            'object': 'chat.completion.chunk'
        }
        
        return ChatCompletionModel(**chunk)