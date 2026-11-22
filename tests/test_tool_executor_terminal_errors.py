"""Host-only terminal failures escape tool results and stop native turns."""
import asyncio
import json

import pytest

from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.types import CanonicalToolCall
from test_agent_loop_control import Control, Provider, answer, execute, loop_for, tool


class Denied(RuntimeError):
    pass


class DeadlineDenied(TimeoutError):
    pass


def call(name='operation'):
    return CanonicalToolCall(id='runtime-call', name=name, arguments={})


@pytest.mark.parametrize('error_type', [Denied, DeadlineDenied])
def test_sync_explicit_terminal_failure_propagates(error_type):
    executor = ToolExecutor()
    executor.propagate_errors(error_type)
    def operation(): raise error_type('terminal')
    executor.register('operation', operation)
    with pytest.raises(error_type): executor.execute(call())


@pytest.mark.asyncio
@pytest.mark.parametrize('error_type', [Denied, DeadlineDenied])
@pytest.mark.parametrize('sync_callable', [False, True])
async def test_async_explicit_terminal_failure_propagates(error_type, sync_callable):
    executor = ToolExecutor()
    executor.propagate_errors(error_type)
    def operation_sync(): raise error_type('terminal')
    async def operation_async(): raise error_type('terminal')
    executor.register('operation', operation_sync if sync_callable else operation_async)
    with pytest.raises(error_type): await executor.execute_async(call())


@pytest.mark.asyncio
async def test_parallel_terminal_failure_cancels_and_joins_sibling():
    executor = ToolExecutor()
    executor.propagate_errors(Denied)
    entered, joined = asyncio.Event(), asyncio.Event()
    async def held():
        entered.set()
        try: await asyncio.Event().wait()
        finally: joined.set()
    async def denied():
        await entered.wait()
        raise Denied('authority revoked')
    executor.register_many([('held', held), ('denied', denied)])
    with pytest.raises(Denied):
        await executor.execute_parallel_async([call('held'), call('denied')])
    assert joined.is_set()


@pytest.mark.parametrize('error_type', [Denied, DeadlineDenied])
def test_fork_and_options_preserve_private_policy_without_mutating_parent(error_type):
    parent = ToolExecutor()
    parent.propagate_errors(Denied)
    child = parent.with_options({'max_parallel_tools': 1})
    child.propagate_errors(DeadlineDenied)
    sibling = parent.fork()
    def operation(): raise error_type('terminal')
    for executor in [parent, child, sibling]: executor.register('operation', operation)
    with pytest.raises(error_type): child.execute(call())
    for executor in [parent, sibling]:
        if error_type is Denied:
            with pytest.raises(Denied): executor.execute(call())
        else:
            result = executor.execute(call())
            assert result.is_error and result.error_type == 'DeadlineDenied'


@pytest.mark.parametrize('value', ['PermissionError', PermissionError('instance'), BaseException, None, ['Denied']])
def test_policy_rejects_non_exception_classes_without_changing_configuration(value):
    executor = ToolExecutor()
    executor.propagate_errors(Denied)
    with pytest.raises(TypeError): executor.propagate_errors(RuntimeError, value)
    assert executor._propagated_errors == (Denied,)
    with pytest.raises(ValueError): executor.with_options({'propagate_errors': ['Denied']})


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('terminal', [False, True])
async def test_native_loop_cannot_buy_a_turn_after_denial_but_ordinary_error_recovers(stream, terminal):
    executor = ToolExecutor()
    executor.propagate_errors(Denied)
    async def operation():
        if terminal: raise Denied('fence lost')
        raise ValueError('ordinary operation failure')
    provider = Provider([answer(calls=[tool('effect', 'operation')]), answer('recovered')])
    control = Control()
    loop = loop_for(provider, control, tools=[operation], tool_executor=executor)
    if terminal:
        with pytest.raises(Denied): await execute(loop, stream, user_input='task')
        assert len(provider.requests) == len(control.attempts) == 1
        assert not control.candidates
        assert all(boundary != 'tool_results' for boundary, _ in control.saved)
    else:
        await execute(loop, stream, user_input='task')
        assert len(provider.requests) == len(control.attempts) == 2
        results = [m for m in provider.requests[1][0] if m['role'] == 'tool']
        assert len(results) == 1 and 'ordinary operation failure' in results[0]['content']
        assert control.candidates


@pytest.mark.asyncio
async def test_negative_domain_receipt_remains_a_normal_complete_tool_result():
    executor = ToolExecutor()
    executor.propagate_errors(Denied)
    receipt = {'ok':False, 'error':{'code':'unknown_ref','message':'not a visible peer'}}
    async def operation(): return receipt
    executor.register('operation', operation)
    result = await executor.execute_async(call())
    assert not result.is_error and json.loads(result.content) == receipt


@pytest.mark.asyncio
async def test_legacy_default_still_converts_same_exception_into_result():
    executor = ToolExecutor()
    async def operation(): raise Denied('ordinary legacy error')
    executor.register('operation', operation)
    result = await executor.execute_async(call())
    assert result.is_error and result.error_type == 'Denied'
