import time

import aioboto3
import boto3
import json

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChatResponse, ModelChat
from magic_llm.model.ModelAudio import AudioSpeechRequest
from magic_llm.model.ModelChatStream import ChatCompletionModel, UsageModel


class EngineAmazon(BaseChat):
    def __init__(self,
                 aws_access_key_id: str,
                 aws_secret_access_key: str,
                 region_name: str = 'us-east-1',
                 service_name: str = 'bedrock-runtime',
                 **kwargs):
        super().__init__(**kwargs)
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
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

    def prepare_data(self, chat: ModelChat, **kwargs):
        if self.model.startswith('amazon.nova'):
            m = chat.get_messages()
            for i in m:
                i['content'] = [{
                    "text": i['content']
                }]
            body = json.dumps({
                "messages": m,
                "inferenceConfig": {
                    "max_new_tokens": kwargs.get('max_new_tokens', 4096),
                    "temperature": kwargs.get('temperature', 1),
                    "topP": kwargs.get('topP', 1)
                }
            })

        elif self.model.startswith('amazon'):
            body = json.dumps({
                "inputText": chat.generic_chat(format='titan'),
                "textGenerationConfig": {
                    "maxTokenCount": kwargs.get('maxTokenCount', 4096),
                    "temperature": kwargs.get('temperature', 0),
                    "topP": kwargs.get('topP', 1),
                    "stopSequences": kwargs.get('stopSequences', ['User:']),
                }
            })
        elif self.model.startswith('anthropic'):
            body = json.dumps({
                "prompt": chat.generic_chat(format='claude'),
                "max_tokens_to_sample": kwargs.get('max_tokens_to_sample', 1024),
                "temperature": kwargs.get('temperature', 0.5),
                "top_k": kwargs.get('top_k', 250),
                "top_p": kwargs.get('top_p', 1),
                "stop_sequences": kwargs.get('stop_sequences', ["\n\nHuman:"]),
                # "anthropic_version": "bedrock-2023-05-31"
            })
        elif self.model.startswith('meta'):
            body = json.dumps({
                "prompt": chat.generic_chat(format='llama2'),
                "max_gen_len": kwargs.get('max_gen_len', 1024),
                "temperature": kwargs.get('temperature', 0.2),
                "top_p": kwargs.get('top_p', 1),
                # "stop_sequences": kwargs.get('stop_sequences', ["[/INST]"]),
            })
        else:
            raise Exception("Unknown model")
        return body

    @BaseChat.async_intercept_generate
    async def async_generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        async with self.aclient as client:
            response = await client.invoke_model(
                body=self.prepare_data(chat, **kwargs),
                modelId=self.model,
                accept='application/json',
                contentType='application/json'
            )

            r = json.loads(await response['body'].read())

            if self.model.startswith('amazon'):
                return ModelChatResponse(**{
                    'content': r['results'][0]['outputText'],
                    'role': 'assistant',
                    'usage': UsageModel(
                        prompt_tokens=r['inputTextTokenCount'],
                        completion_tokens=r['results'][0]['tokenCount'],
                        total_tokens=r['inputTextTokenCount'] + r['results'][0]['tokenCount'],
                    )
                })
            elif self.model.startswith('anthropic'):
                return ModelChatResponse(**{
                    'content': r['completion'],
                    'role': 'assistant',
                    'usage': UsageModel(
                        prompt_tokens=len(chat.generic_chat(format='claude')),
                        completion_tokens=len(r['completion']),
                        total_tokens=len(chat.generic_chat(format='claude')) + len(r['completion'])
                    )
                })
            elif self.model.startswith('meta'):
                return ModelChatResponse(**{
                    'content': r['generation'],
                    'role': 'assistant',
                    'usage': UsageModel(
                        prompt_tokens=r['prompt_token_count'],
                        completion_tokens=r['generation_token_count'],
                        total_tokens=r['prompt_token_count'] + r['generation_token_count']
                    )
                })

    @BaseChat.sync_intercept_generate
    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        response = self.client.invoke_model(body=self.prepare_data(chat, **kwargs),
                                            modelId=self.model,
                                            accept='application/json',
                                            contentType='application/json')

        r = json.loads(response.get('body').read())

        if self.model.startswith('amazon'):
            return ModelChatResponse(**{
                'content': r['results'][0]['outputText'],
                'role': 'assistant',
                'usage': UsageModel(
                    prompt_tokens=r['inputTextTokenCount'],
                    completion_tokens=r['results'][0]['tokenCount'],
                    total_tokens=r['inputTextTokenCount'] + r['results'][0]['tokenCount'],
                )
            })
        elif self.model.startswith('anthropic'):
            return ModelChatResponse(**{
                'content': r['completion'],
                'role': 'assistant',
                'usage': UsageModel(
                    prompt_tokens=len(chat.generic_chat(format='claude')),
                    completion_tokens=len(r['completion']),
                    total_tokens=len(chat.generic_chat(format='claude')) + len(r['completion'])
                )
            })
        elif self.model.startswith('meta'):
            return ModelChatResponse(**{
                'content': r['generation'],
                'role': 'assistant',
                'usage': UsageModel(
                    prompt_tokens=r['prompt_token_count'],
                    completion_tokens=r['generation_token_count'],
                    total_tokens=r['prompt_token_count'] + r['generation_token_count']
                )
            })

    @BaseChat.sync_intercept_stream_generate
    def stream_generate(self, chat: ModelChat, **kwargs):
        response = self.client.invoke_model_with_response_stream(
            body=self.prepare_data(chat, **kwargs),
            modelId=self.model,
            accept='application/json',
            contentType='application/json')
        for event in response.get("body"):
            event = json.loads(event["chunk"]["bytes"])
            chunk = self.format_event_to_chunk(event)
            prompt_tokens = event.get('amazon-bedrock-invocationMetrics', {}).get('inputTokenCount', 0)
            completion_tokens = event.get('amazon-bedrock-invocationMetrics', {}).get('outputTokenCount', 0)
            chunk.usage = UsageModel(**{
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
            })
            yield chunk

    @BaseChat.async_intercept_stream_generate
    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        body = self.prepare_data(chat, **kwargs)

        async with self.aclient as client:
            response = await client.invoke_model_with_response_stream(
                body=body,
                modelId=self.model,
                accept='application/json',
                contentType='application/json')
            async for event in response.get("body"):
                event = json.loads(event["chunk"]["bytes"])
                chunk = self.format_event_to_chunk(event)
                prompt_tokens = event.get('amazon-bedrock-invocationMetrics', {}).get('inputTokenCount', 0)
                completion_tokens = event.get('amazon-bedrock-invocationMetrics', {}).get('outputTokenCount', 0)
                chunk.usage = UsageModel(**{
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens,
                })
                yield chunk

    def format_event_to_chunk(self, event):
        if self.model.startswith('anthropic'):
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
        elif self.model.startswith('amazon.nova'):
            chunk = {
                'id': '1',
                'choices':
                    [{
                        'delta':
                            {
                                'content': event.get('contentBlockDelta', {}).get('delta', {}).get('text'),
                                'role': None
                            },
                        'finish_reason': 'stop' if event.get('messageStop', {}).get(
                            'stopReason') == 'end_turn' else None,
                        'index': event.get('index')
                    }],
                'created': int(time.time()),
                'model': self.model,
                'object': 'chat.completion.chunk'
            }
            if c := chunk.get('metadata', {}).get('usage'):
                chunk['usage'] = UsageModel(prompt_tokens=c['inputTokens'],
                                            completion_tokens=c['outputTokens'],
                                            total_tokens=c['inputTokens'] + c['outputTokens'])

        elif self.model.startswith('amazon'):
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
        elif self.model.startswith('meta'):
            chunk = {
                'id': '1',
                'choices':
                    [{
                        'delta':
                            {
                                'content': event['generation'],
                                'role': None
                            },
                        'finish_reason': 'stop' if event['stop_reason'] else None,
                        'index': 0
                    }],
                'created': int(time.time()),
                'model': self.model,
                'object': 'chat.completion.chunk'
            }
        else:
            raise Exception('Unrecognized')
        return ChatCompletionModel(**chunk)

    def audio_speech(self, data: AudioSpeechRequest, **kwargs):
        response = self.client.synthesize_speech(
            VoiceId=data.voice,
            OutputFormat=data.response_format,
            Text=data.input,
            Engine=data.model,
        )
        return response['AudioStream']
