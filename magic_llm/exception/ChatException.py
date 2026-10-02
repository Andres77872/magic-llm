from copy import deepcopy

class ChatException(Exception):
    """Custom exception class for chat-related errors"""

    def __init__(self, message="A chat error occurred", error_code=None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class RequestValidationError(ChatException):
    """Mandatory input validation failure: never retry, log bodies or fallback.

    The original host error remains available for application error mapping;
    the outward message is safe for routine diagnostics.
    """
    def __init__(self, validation_error):
        self.validation_error = validation_error
        self.outcome = deepcopy(getattr(validation_error, "outcome", None))
        code = getattr(validation_error, 'code', None) or getattr(validation_error, 'error_code', None) or getattr(self.outcome, 'code', None) or 'REQUEST_VALIDATION_FAILED'
        self.code = code
        super().__init__('Provider request rejected by mandatory validation', error_code=code)
