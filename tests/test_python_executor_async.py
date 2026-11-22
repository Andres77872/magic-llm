import asyncio
import json
import os
import time

import pytest

from magic_llm.util.python_executor import PythonExecutor


@pytest.mark.asyncio
@pytest.mark.parametrize('mode',['subprocess','restricted_builtins','in_process'])
async def test_async_python_preserves_stdout_and_sync_error_contract(mode):
    executor=PythonExecutor(safety_mode=mode,max_output_chars=100)
    assert await executor.execute_async('print(2 * 7)')=='14\n'
    # A successful program may itself print JSON containing an error field.
    assert json.loads(await executor.execute_async('print(\'{"error":"data"}\')'))=={'error':'data'}
    with pytest.raises((RuntimeError,ZeroDivisionError),match='division by zero'):
        await executor.execute_async('1 / 0')
    assert 'division by zero' in json.loads(executor('1 / 0'))['error']


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel',[True,False])
async def test_async_python_timeout_or_cancel_reaps_owned_subprocess(tmp_path,cancel):
    pidfile=tmp_path/'child.pid'
    executor=PythonExecutor(timeout=10 if cancel else .2)
    code=f'from pathlib import Path\nimport os,time\nPath({str(pidfile)!r}).write_text(str(os.getpid()))\ntime.sleep(10)'
    task=asyncio.create_task(executor.execute_async(code))
    for _ in range(100):
        if pidfile.exists(): break
        await asyncio.sleep(.005)
    else:
        task.cancel()
        await asyncio.gather(task,return_exceptions=True)
        pytest.fail('child did not start')
    pid=int(pidfile.read_text())
    if cancel:task.cancel()
    started=time.monotonic()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await asyncio.wait_for(task,2)
    assert time.monotonic()-started < 1, 'cleanup must finish before the outer test timeout'
    with pytest.raises(ProcessLookupError):os.kill(pid,0)


@pytest.mark.asyncio
async def test_async_python_parallel_output_and_truncation_are_isolated():
    executor=PythonExecutor(max_output_chars=4)
    outputs=await asyncio.gather(executor.execute_async('print("abcdef")'),executor.execute_async('print(12)'))
    assert outputs==['abcd\n... [truncated 3 chars]','12\n']


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != 'posix',reason='Owned process groups are POSIX-specific')
@pytest.mark.parametrize('parent_waits',[False,True])
@pytest.mark.parametrize('cancel',[False,True])
async def test_async_python_closes_descendant_pipes_even_after_parent_exit(tmp_path,parent_waits,cancel):
    marker=tmp_path/'descendant-started'
    descendant=f'from pathlib import Path; import time; Path({str(marker)!r}).write_text("ready"); time.sleep(10)'
    code=f'import subprocess,sys,time\nsubprocess.Popen([sys.executable,"-c",{descendant!r}])\n'
    if parent_waits:code+='time.sleep(10)'
    executor=PythonExecutor(timeout=10 if cancel else .2)
    task=asyncio.create_task(executor.execute_async(code))
    for _ in range(100):
        if marker.exists():break
        await asyncio.sleep(.005)
    else:
        task.cancel()
        await asyncio.gather(task,return_exceptions=True)
        pytest.fail('descendant did not start')
    if cancel:task.cancel()
    started=time.monotonic()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await asyncio.wait_for(task,2)
    assert time.monotonic()-started < 1, 'cleanup must finish before the outer test timeout'
