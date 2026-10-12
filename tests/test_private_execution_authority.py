import pytest
from magic_llm.model import ModelChat
from magic_llm.exception.ChatException import RequestValidationError
from magic_llm.util.private_limits import PrivateRequestLimits

@pytest.mark.parametrize('available',[False,None,1,'yes'])
def test_closed_or_invalid_authority_stops_an_already_allocated_client_before_payload_preparation(available):
    limits=PrivateRequestLimits(model='approved',max_input_tokens=8192,max_output_tokens=64,
        max_messages=8,timeout_seconds=10,authorize_request=lambda:available)
    chat=ModelChat();chat.add_user_message('At the gate.')
    with pytest.raises(RequestValidationError) as error:limits.prepare(chat,{'max_tokens':64})
    assert error.value.code == 'EXECUTION_REQUEST_UNAVAILABLE'
    assert str(error.value) == 'Provider request rejected by mandatory validation'
    assert chat.messages==[{'role':'user','content':'At the gate.'}]

def test_authority_is_checked_for_each_attempt_and_not_exposed_in_the_limits_repr():
    available=[True]
    def authority():return available[0]
    limits=PrivateRequestLimits(model='approved',max_input_tokens=8192,max_output_tokens=64,
        max_messages=8,timeout_seconds=10,authorize_request=authority)
    chat=ModelChat();chat.add_user_message('At the gate.')
    limits.prepare(chat,{})
    available[0]=False
    with pytest.raises(RequestValidationError):limits.prepare(chat,{})
    assert 'authorize_request' not in repr(limits) and 'authority' not in repr(limits)

def test_request_authority_must_be_an_in_memory_callable():
    with pytest.raises(ValueError,match='authorization must be callable'):
        PrivateRequestLimits(model='approved',max_input_tokens=8192,max_output_tokens=64,
            max_messages=8,timeout_seconds=10,authorize_request='profile-supplied')
