import time

import boto3
import json

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChatResponse, ModelChat


class EngineAmazon(BaseChat):
    def __init__(self,
                 aws_access_key_id: str,
                 aws_secret_access_key: str,
                 region_name: str = 'us-east-1',
                 service_name: str = 'bedrock-runtime',
                 **kwargs):
        super().__init__(**kwargs)

        self.client = boto3.client(
            service_name=service_name,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

    def prepare_data(self, chat: ModelChat, **kwargs):
        if self.model.startswith('amazon'):
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
            })
        else:
            raise Exception("Unknown model")
        return body

    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
        response = self.client.invoke_model(body=self.prepare_data(chat, **kwargs),
                                            modelId=self.model,
                                            accept='application/json',
                                            contentType='application/json')

        r = json.loads(response.get('body').read())

        # print(r)

        if self.model.startswith('amazon'):
            return ModelChatResponse(**{
                'content': r['results'][0]['outputText'],
                'prompt_tokens': r['inputTextTokenCount'],
                'completion_tokens': r['results'][0]['tokenCount'],
                'total_tokens': r['inputTextTokenCount'] + r['results'][0]['tokenCount'],
                'role': 'assistant'
            })
        elif self.model.startswith('anthropic'):
            return ModelChatResponse(**{
                'content': r['completion'],
                'prompt_tokens': len(chat.generic_chat(format='claude')),
                'completion_tokens': len(r['completion']),
                'total_tokens': len(chat.generic_chat(format='claude')) + len(r['completion']),
                'role': 'assistant'
            })
        elif self.model.startswith('meta'):
            return ModelChatResponse(**{
                'content': r['generation'],
                'prompt_tokens': r['prompt_token_count'],
                'completion_tokens': r['generation_token_count'],
                'total_tokens': r['prompt_token_count'] + r['generation_token_count'],
                'role': 'assistant'
            })

    def stream_generate(self, chat: ModelChat, **kwargs):
        response = self.client.invoke_model_with_response_stream(
            body=self.prepare_data(chat, **kwargs),
            modelId=self.model,
            accept='application/json',
            contentType='application/json')
        for event in response.get("body"):
            event = json.loads(event["chunk"]["bytes"])
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
                stop_reason = event['stop_reason']
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
                stop_reason = event['completionReason']
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
                stop_reason = event['stop_reason']
            else:
                raise Exception('Unrecognized')
            chunk = json.dumps(chunk)
            yield f'data: {chunk}\n'
            yield f'\n'
            if stop_reason:
                yield 'data: [DONE]\n'
            yield f'\n'
