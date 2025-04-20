import json
import time
from typing import Dict

from magic_llm.engine.amazon_adapters.base_provider import AmazonBaseProvider
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel


class ProviderAmazonTitan(AmazonBaseProvider):
    """
    Provider for Amazon Bedrock Titan models.
    """
    
    def prepare_data(self, chat: ModelChat, **kwargs) -> str:
        """
        Prepare the request data for Titan models.
        
        Args:
            chat: The chat model containing messages
            **kwargs: Additional parameters for the request
            
        Returns:
            A JSON string containing the request body
        """
        body = json.dumps({
            "inputText": chat.generic_chat(format='titan'),
            "textGenerationConfig": {
                "maxTokenCount": kwargs.get('maxTokenCount', 4096),
                "temperature": kwargs.get('temperature', 0),
                "topP": kwargs.get('topP', 1),
                "stopSequences": kwargs.get('stopSequences', ['User:']),
            }
        })
        
        return body
    
    def process_response(self, response: dict) -> Dict:
        """
        Process the response from Titan models.
        
        Args:
            response: The response from the model
            
        Returns:
            A dictionary containing the processed response
        """
        return {
            'content': response['results'][0]['outputText'],
            'role': 'assistant',
            'usage': UsageModel(
                prompt_tokens=response['inputTextTokenCount'],
                completion_tokens=response['results'][0]['tokenCount'],
                total_tokens=response['inputTextTokenCount'] + response['results'][0]['tokenCount'],
            )
        }
    
    def format_event_to_chunk(self, event: dict) -> ChatCompletionModel:
        """
        Format a streaming event from Titan models to a ChatCompletionModel.
        
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
                            'content': event['outputText'],
                            'role': None
                        },
                    'finish_reason': 'stop' if event['completionReason'] == 'FINISH' else None,
                    'index': event['index']
                }],
            'created': int(time.time()),
            'model': self.model,
            'object': 'chat.completion.chunk'
        }
        
        return ChatCompletionModel(**chunk)