import boto3
import json

from magic_llm.engine.base_chat import BaseChat
from magic_llm.model import ModelChatResponse, ModelChat


class EngineAmazon(BaseChat):
    def __init__(self,
                 model: str,
                 aws_access_key_id: str,
                 aws_secret_access_key: str,
                 region_name: str = 'us-east-1',
                 service_name: str = 'bedrock-runtime',
                 stream: bool = False,
                 **kwargs):
        super().__init__()

        self.model = model
        self.stream = stream
        self.kwargs = kwargs
        self.client = boto3.client(
            service_name=service_name,
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

    def generate(self, chat: ModelChat, **kwargs) -> ModelChatResponse:
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

            print(chat.generic_chat(format='llama2'))

        response = self.client.invoke_model(body=body,
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
