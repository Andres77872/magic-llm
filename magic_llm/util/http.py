import json
import logging
from typing import Any, Generator, AsyncGenerator, Optional, Dict, Union, TypeVar, Generic

import aiohttp
import requests
from requests import RequestException

logger = logging.getLogger(__name__)

T = TypeVar('T')


class HttpError(Exception):
    """Base exception for HTTP client errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_content: Optional[bytes] = None):
        self.status_code = status_code
        self.response_content = response_content
        super().__init__(message)


class AsyncHttpClient:
    """
    A reusable HTTP client for making asynchronous requests using aiohttp.
    """

    def __init__(self):
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _ensure_session(self):
        """Ensures the session is initialized."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'with' block or call 'connect()'.")

    async def request(
            self,
            method: str,
            url: str,
            **kwargs,
    ) -> bytes:
        """
        Makes an HTTP request.

        :param method: The HTTP method (e.g., 'POST', 'GET', etc.).
        :param url: The endpoint URL.
        :param kwargs: Additional arguments for aiohttp's request.
        :return: The response content as bytes.
        :raises: HttpError if the request fails or returns a non-200 status code.
        """
        self._ensure_session()
        timeout = aiohttp.ClientTimeout(total=kwargs.pop('timeout', None))
        try:
            async with self.session.request(
                    method,
                    url,
                    timeout=timeout,
                    **kwargs,
            ) as response:
                content = await response.read()
                if response.status != 200:
                    error_msg = f"HTTP {response.status}: {content.decode('utf-8', errors='replace')}"
                    logger.error(f"Request to {url} failed: {error_msg}")
                    raise HttpError(error_msg, response.status, content)
                return content
        except aiohttp.ClientError as e:
            logger.error(f"Request to {url} failed with aiohttp error: {str(e)}")
            raise HttpError(f"aiohttp error: {str(e)}")

    async def post_json(self, url: str, **kwargs) -> Any:
        """
        Sends a POST request with a JSON payload and returns the parsed JSON response.

        :param url: The endpoint URL.
        :param kwargs: Additional arguments for the `request` method.
        :return: The parsed JSON response.
        :raises: HttpError, json.JSONDecodeError
        """
        response = await self.request("POST", url, **kwargs)
        try:
            return json.loads(response.decode('utf-8'))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from {url}: {str(e)}")
            raise

    async def post_raw_binary(self, url: str, **kwargs) -> bytes:
        """
        Sends a POST request and returns the raw binary response.

        :param url: The endpoint URL.
        :param kwargs: Additional arguments for the `request` method.
        :return: The raw binary response content.
        :raises: HttpError
        """
        return await self.request("POST", url, **kwargs)

    async def stream_request(
            self,
            method: str,
            url: str,
            **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Makes a streaming HTTP request.

        :param method: The HTTP method (e.g., 'POST', 'GET', etc.).
        :param url: The endpoint URL.
        :param kwargs: Additional arguments for aiohttp's request.
        :yield: Chunks of data as bytes.
        :raises: HttpError
        """
        self._ensure_session()

        timeout = aiohttp.ClientTimeout(total=kwargs.pop('timeout', None))
        try:
            async with self.session.request(
                    method,
                    url,
                    timeout=timeout,
                    **kwargs
            ) as response:
                if response.status != 200:
                    content = await response.read()
                    error_msg = f"HTTP {response.status}: {content.decode('utf-8', errors='replace')}"
                    logger.error(f"Streaming request to {url} failed: {error_msg}")
                    raise HttpError(error_msg, response.status, content)

                async for chunk in response.content:
                    yield chunk
        except aiohttp.ClientError as e:
            logger.error(f"Streaming request to {url} failed with aiohttp error: {str(e)}")
            raise HttpError(f"aiohttp streaming error: {str(e)}")

    async def post_stream(self, url: str, **kwargs) -> AsyncGenerator[bytes, None]:
        """
        Sends a POST request and returns a streaming response.

        :param url: The endpoint URL.
        :param kwargs: Additional arguments for the request method.
        :yield: Chunks of data as bytes.
        :raises: HttpError
        """
        async for chunk in self.stream_request("POST", url, **kwargs):
            yield chunk


class HttpClient:
    """
    A reusable HTTP client for making synchronous requests using the requests library.
    """

    def __init__(self):
        self.session = None

    def __enter__(self):
        self.session = requests.Session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
            self.session = None

    def _ensure_session(self):
        """Ensures the session is initialized."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'with' block.")

    def request(
            self,
            method: str,
            url: str,
            **kwargs,
    ) -> bytes:
        """
        Makes an HTTP request.

        :param method: The HTTP method (e.g., 'POST', 'GET', etc.).
        :param url: The endpoint URL.
        :param kwargs: Additional arguments for requests.
        :return: The response content as bytes.
        :raises: HttpError, requests.exceptions.RequestException
        """
        self._ensure_session()
        try:
            response = self.session.request(method, url, **kwargs)
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.content.decode('utf-8', errors='replace')}"
                logger.error(f"Request to {url} failed: {error_msg}")
                raise HttpError(error_msg, response.status_code, response.content)
            return response.content
        except RequestException as e:
            # Preserve the original error but add more context if possible
            if hasattr(e, 'response') and hasattr(e.response, 'content'):
                error_message = f"{str(e)}: {e.response.content.decode('utf-8', errors='replace')}"
                logger.error(f"Request to {url} failed: {error_message}")
                raise HttpError(error_message, 
                               getattr(e.response, 'status_code', None),
                               getattr(e.response, 'content', None))
            logger.error(f"Request to {url} failed with requests error: {str(e)}")
            raise HttpError(f"requests error: {str(e)}")

    def post_json(self, url: str, **kwargs) -> Any:
        """
        Sends a POST request with a JSON payload and returns the parsed JSON response.

        :param url: The endpoint URL.
        :param kwargs: Additional arguments for the `request` method.
        :return: The parsed JSON response.
        :raises: HttpError, json.JSONDecodeError
        """
        response = self.request("POST", url, **kwargs)
        try:
            return json.loads(response.decode('utf-8'))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from {url}: {str(e)}")
            raise

    def post_raw_binary(self, url: str, **kwargs) -> bytes:
        """
        Sends a POST request and returns the raw binary response.

        :param url: The endpoint URL.
        :param kwargs: Additional arguments for the `request` method.
        :return: The raw binary response content.
        :raises: HttpError
        """
        return self.request("POST", url, **kwargs)

    def stream_request(
            self,
            method: str,
            url: str,
            **kwargs
    ) -> Generator[str, None, None]:
        """
        Makes a streaming HTTP request.

        :param method: The HTTP method (e.g., 'POST', 'GET', etc.).
        :param url: The endpoint URL.
        :param kwargs: Additional arguments for requests.
        :yield: Lines of text from the response.
        :raises: HttpError
        """
        self._ensure_session()

        # Set stream=True for streaming response
        kwargs['stream'] = True
        if (d := kwargs.get('data')) and isinstance(d, dict):
            kwargs['data'] = json.dumps(d)
        try:
            with self.session.request(method, url, **kwargs) as response:
                if response.status_code != 200:
                    error_msg = f"HTTP {response.status_code}: {response.content.decode('utf-8', errors='replace')}"
                    logger.error(f"Streaming request to {url} failed: {error_msg}")
                    raise HttpError(error_msg, response.status_code, response.content)

                for line in response.iter_lines(decode_unicode=False):
                    if line:
                        yield line.decode('utf-8')
        except RequestException as e:
            # Add context to the error if possible
            if hasattr(e, 'response') and hasattr(e.response, 'content'):
                error_message = f"{str(e)}: {e.response.content.decode('utf-8', errors='replace')}"
                logger.error(f"Streaming request to {url} failed: {error_message}")
                raise HttpError(error_message,
                               getattr(e.response, 'status_code', None),
                               getattr(e.response, 'content', None))
            logger.error(f"Streaming request to {url} failed with requests error: {str(e)}")
            raise HttpError(f"requests streaming error: {str(e)}")
