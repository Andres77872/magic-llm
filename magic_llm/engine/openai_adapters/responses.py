"""Responses wire format behind the existing chat/agent interfaces.

Conversation state stays in ModelChat. Encrypted reasoning and output items are
replayed with tool results, so no provider-side storage or shared client state is
needed for an agent's next iteration.
"""
import json
import re
from copy import deepcopy

from magic_llm.engine._usage_factory import build_usage_model
from magic_llm.engine.openai_adapters.openai_base import _is_official_openai_url
from magic_llm.model.ModelChatResponse import ModelChatResponse
from magic_llm.model.ModelChatStream import ChatCompletionModel


def _input_items(messages):
    items = []
    for message in messages:
        role = message['role']
        if role == 'assistant' and message.get('responses_output'):
            items.extend(deepcopy(message['responses_output']))
            continue
        content = message.get('content')
        if role == 'tool':
            items.append({'type': 'function_call_output',
                          'call_id': message['tool_call_id'], 'output': content or ''})
            continue
        if content is not None:
            if isinstance(content, list):
                parts = []
                for part in content:
                    kind = part.get('type')
                    if kind == 'text':
                        parts.append({'type': 'input_text', 'text': part['text']})
                    elif kind == 'image_url':
                        image = part['image_url']
                        parts.append({'type': 'input_image', 'image_url': image['url'],
                                      'detail': image.get('detail', 'auto')})
                    elif kind in {'input_text', 'input_image', 'input_file'}:
                        parts.append(deepcopy(part))
                    else:
                        raise ValueError(f'Unsupported Responses content type: {kind}')
                content = parts
            items.append({'role': role, 'content': content})
        for call in message.get('tool_calls') or []:
            function = call['function']
            items.append({'type': 'function_call', 'call_id': call['id'],
                          'name': function['name'], 'arguments': function['arguments']})
    return items


def _usage(response):
    usage = response.get('usage') or {}
    return build_usage_model(
        prompt_tokens=usage.get('input_tokens'),
        completion_tokens=usage.get('output_tokens'),
        total_tokens=usage.get('total_tokens'),
        cached_tokens_read=(usage.get('input_tokens_details') or {}).get('cached_tokens'),
        reasoning_tokens=(usage.get('output_tokens_details') or {}).get('reasoning_tokens'),
        provider_request_id=response.get('id'), service_tier=response.get('service_tier'),
    )


def _finish_reason(response):
    if response.get('error') or response.get('status') in {'failed', 'cancelled'}:
        error = response.get('error') or {}
        raise ValueError(f"Responses API failed: {error.get('message') or response.get('status')}")
    if response.get('status') == 'incomplete':
        reason = (response.get('incomplete_details') or {}).get('reason')
        return 'length' if reason == 'max_output_tokens' else 'content_filter'
    if any(item.get('type') == 'function_call' for item in response.get('output') or []):
        return 'tool_calls'
    return 'stop'


class OpenAIResponsesAdapter:
    def __init__(self, provider):
        self.provider = provider

    def transform_request(self, chat, **kwargs):
        raw, headers = self.provider.transform_request(chat, **kwargs)
        data = json.loads(raw)
        data.pop('messages', None)
        data['input'] = _input_items(chat.get_messages())
        # HTTP/client controls are not model request fields.
        for name in ('stream_options', 'timeout', 'retries', 'executor'):
            data.pop(name, None)
        for name in ('max_completion_tokens', 'max_tokens'):
            if name in data:
                data.setdefault('max_output_tokens', data.pop(name))
        default_reasoning = self.provider.kwargs.get('reasoning')
        if isinstance(default_reasoning, dict) and isinstance(data.get('reasoning'), dict):
            # A per-call summary should not erase the client's default effort.
            data['reasoning'] = {**default_reasoning, **data['reasoning']}
        if 'reasoning_effort' in data:
            effort = data.pop('reasoning_effort')
            reasoning = dict(data.get('reasoning') or {})
            if effort is not None:
                # Per-call options beat client defaults even across the two
                # spellings. Keep other reasoning fields (e.g. summary) intact.
                if ('reasoning_effort' in kwargs
                        and 'effort' not in (kwargs.get('reasoning') or {})):
                    reasoning['effort'] = effort
                else:
                    reasoning.setdefault('effort', effort)
            if reasoning:
                data['reasoning'] = reasoning
        if 'response_format' in data:
            format_ = data.pop('response_format')
            if format_.get('type') == 'json_schema':
                format_ = {'type': 'json_schema', **format_['json_schema']}
            data.setdefault('text', {}).setdefault('format', format_)
        if 'verbosity' in data:
            data.setdefault('text', {}).setdefault('verbosity', data.pop('verbosity'))
        if 'tools' in data:
            data['tools'] = [
                {'type': 'function', 'strict': False, **tool['function']}
                if tool.get('type') == 'function' and 'function' in tool else tool
                for tool in data['tools']
            ]
        choice = data.get('tool_choice')
        if isinstance(choice, dict) and choice.get('type') == 'function' and 'function' in choice:
            data['tool_choice'] = {'type': 'function', 'name': choice['function']['name']}
        if data.pop('n', 1) != 1:
            raise ValueError('Responses supports only one generation (n=1)')
        data.setdefault('store', False)
        include = list(data.get('include') or [])
        if 'reasoning.encrypted_content' not in include:
            include.append('reasoning.encrypted_content')
        # GPT-6 defaults to reasoning, including when effort is omitted.
        # Keep this model-specific rule off OpenAI-compatible providers.
        if (_is_official_openai_url(self.provider.base_url)
                and re.match(r'^gpt-6(?:-|$)', data.get('model') or '')
                and (data.get('reasoning') or {}).get('effort') != 'none'):
            for name in ('temperature', 'top_p', 'top_logprobs', 'logprobs'):
                data.pop(name, None)
            include = [value for value in include if value != 'message.output_text.logprobs']
        data['include'] = include
        return json.dumps(data).encode('utf-8'), headers

    def transform_response(self, raw):
        finish = _finish_reason(raw)
        text, refusal, annotations, calls = [], [], [], []
        for item in raw.get('output') or []:
            if item.get('type') == 'message':
                for part in item.get('content') or []:
                    if part.get('type') == 'output_text':
                        text.append(part.get('text', ''))
                        annotations.extend(part.get('annotations') or [])
                    elif part.get('type') == 'refusal':
                        refusal.append(part.get('refusal', ''))
            elif item.get('type') == 'function_call' and finish == 'tool_calls':
                calls.append({'id': item['call_id'], 'type': 'function',
                              'function': {'name': item['name'], 'arguments': item['arguments']}})
        return ModelChatResponse(
            id=raw['id'], object='chat.completion', created=raw.get('created_at', 0),
            model=raw.get('model') or self.provider.model,
            choices=[{'index': 0, 'finish_reason': finish, 'message': {
                'role': 'assistant', 'content': ''.join(text) or None,
                'tool_calls': calls or None, 'refusal': ''.join(refusal) or None,
                'annotations': annotations,
            }}], usage=_usage(raw), provider_request_id=raw['id'],
            service_tier=raw.get('service_tier'), responses_output=deepcopy(raw.get('output') or []),
        )

    def transform_stream_chunk(self, raw, context):
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        if isinstance(raw, str):
            if not raw.startswith('data:') or raw[5:].strip() == '[DONE]':
                return None
            raw = json.loads(raw[5:])
        kind = raw.get('type')
        response = raw.get('response') or {}
        if response:
            context.update(id=response['id'], model=response.get('model'), created=response.get('created_at'))
        if kind in {'error', 'response.failed'}:
            error = response.get('error') or raw.get('error') or raw
            raise ValueError(f"Responses API failed: {error.get('message', 'unknown error')}")
        delta, finish, usage, output = {}, None, None, None
        if kind == 'response.output_text.delta':
            delta['content'] = raw['delta']
        elif kind == 'response.refusal.delta':
            delta['refusal'] = raw['delta']
        elif kind in {'response.reasoning_summary_text.delta', 'response.reasoning_text.delta'}:
            delta['reasoning_content'] = raw['delta']
        elif kind == 'response.output_item.added' and raw['item']['type'] == 'function_call':
            item = raw['item']
            context['has_tool_calls'] = True
            delta['tool_calls'] = [{'index': raw['output_index'], 'id': item['call_id'],
                                    'type': 'function', 'function': {
                                        'name': item['name'], 'arguments': item.get('arguments', '')}}]
        elif kind == 'response.function_call_arguments.delta':
            delta['tool_calls'] = [{'index': raw['output_index'], 'function': {'arguments': raw['delta']}}]
        elif kind in {'response.completed', 'response.incomplete'}:
            if kind == 'response.incomplete' and context.get('has_tool_calls'):
                raise ValueError('Responses ended with incomplete tool calls; increase max_output_tokens')
            finish = _finish_reason(response)
            usage = _usage(response)
            output = deepcopy(response.get('output') or [])
            context['terminal'] = True
        else:
            # Ignore SSE event headers, lifecycle events and done snapshots;
            # snapshots must not duplicate streamed text or tool arguments.
            return None
        return ChatCompletionModel(
            id=context.get('id') or raw.get('response_id') or '',
            model=context.get('model') or self.provider.model,
            created=context.get('created'), choices=[{'index': 0, 'delta': delta, 'finish_reason': finish}],
            usage=usage, responses_output=output,
        )
