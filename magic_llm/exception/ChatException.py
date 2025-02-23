class ChatException(Exception):
    """Custom exception class for chat-related errors"""

    def __init__(self, message="A chat error occurred", error_code=None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
