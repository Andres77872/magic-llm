import json
import time
from abc import ABC
from typing import Dict, Tuple, Optional

import aioboto3
import boto3

from magic_llm.model import ModelChat
from magic_llm.model.ModelAudio import AudioSpeechRequest
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel


class AmazonBaseProvider(ABC):
    def __init__(self,
                 aws_access_key_id: str,
                 aws_secret_access_key: str,
                 region_name: str = 'us-east-1',
                 service_name: str = 'bedrock-runtime',
                 model: str | None = None,
                 **kwargs):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.service_name = service_name
        self.model = model
        self.kwargs = kwargs
        
        self.client = boto3.client(
            service_name=service_name,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        self.aclient = aioboto3.Session().client(
            service_name=service_name,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

    def prepare_data(self, chat: ModelChat, **kwargs) -> str:
        """
        Prepare the request data for the model.
        
        Args:
            chat: The chat model containing messages
            **kwargs: Additional parameters for the request
            
        Returns:
            A JSON string containing the request body
        """
        raise NotImplementedError("Subclasses must implement prepare_data")
    
    def process_response(self, response: dict) -> Dict:
        """
        Process the response from the model.
        
        Args:
            response: The response from the model
            
        Returns:
            A dictionary containing the processed response
        """
        raise NotImplementedError("Subclasses must implement process_response")
    
    def format_event_to_chunk(self, event: dict) -> ChatCompletionModel:
        """
        Format a streaming event to a ChatCompletionModel.
        
        Args:
            event: The event from the streaming response
            
        Returns:
            A ChatCompletionModel containing the formatted event
        """
        raise NotImplementedError("Subclasses must implement format_event_to_chunk")
    
    def audio_speech(self, data: AudioSpeechRequest, **kwargs):
        """
        Generate speech from text.
        
        Args:
            data: The speech request data
            **kwargs: Additional parameters for the request
            
        Returns:
            The audio stream
        """
        response = self.client.synthesize_speech(
            VoiceId=data.voice,
            OutputFormat=data.response_format,
            Text=data.input,
            Engine=data.model,
        )
        return response['AudioStream']