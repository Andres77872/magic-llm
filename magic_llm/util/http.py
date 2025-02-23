import json
from typing import Any, Generator, AsyncGenerator

import aiohttp
import requests
from requests import RequestException


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
        :return: The aiohttp.ClientResponse object.
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'with' block or call 'connect()'.")
        timeout = aiohttp.ClientTimeout(total=kwargs.pop('timeout', None))
        async with self.session.request(
                method,
                url,
                timeout=timeout,
                **kwargs,
        ) as response:
            if response.status != 200:
                raise Exception(await response.read())
            return await response.read()

    async def post_json(self, url: str, **kwargs) -> Any:
        """
        Sends a POST request with a JSON payload and returns the parsed JSON response.

        :param url: The endpoint URL.
        :param kwargs: Additional arguments for the `request` method.
        :return: The parsed JSON response.
        """
        response = await self.request("POST", url, **kwargs)
        return json.loads(response.decode())

    async def post_raw_binary(self, url: str, **kwargs) -> bytes:
        """
        Sends a POST request and returns the raw binary response.

        :param url: The endpoint URL.
        :param kwargs: Additional arguments for the `request` method.
        :return: The raw binary response content.
        """
        response = await self.request("POST", url, **kwargs)
        return response

    async def stream_request(
            self,
            method: str,
            url: str,
            **kwargs
    ) -> AsyncGenerator[Any, Any]:
        """
        Makes a streaming HTTP request.

        :param method: The HTTP method (e.g., 'POST', 'GET', etc.).
        :param url: The endpoint URL.
        :param kwargs: Additional arguments for aiohttp's request.
        :return: An async generator yielding chunks of data.
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'with' block or call 'connect()'.")

        timeout = aiohttp.ClientTimeout(total=kwargs.pop('timeout', None))
        async with self.session.request(
                method,
                url,
                timeout=timeout,
                **kwargs
        ) as response:
            if response.status != 200:
                raise Exception(await response.read())
            async for chunk in response.content:
                yield chunk

    async def post_stream(self, url: str, **kwargs) -> AsyncGenerator[Any, Any]:
        """
        Sends a POST request and returns a streaming response.

        :param url: The endpoint URL.
        :param kwargs: Additional arguments for the request method.
        :return: An async generator yielding chunks of data.
        """
        async for chunk in self.stream_request("POST", url, **kwargs):
            yield chunk


class HttpClient:
    """
    A reusable HTTP client for making requests using the requests library.
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
        :raises: requests.exceptions.RequestException
        """
        self._ensure_session()
        try:
            response = self.session.request(method, url, **kwargs)
            if response.status_code != 200:
                raise Exception(response.content)
            return response.content
        except RequestException as e:
            # Preserve the original error but add more context if possible
            if hasattr(e.response, 'content'):
                error_message = f"{str(e)}: {e.response.content.decode('utf-8', errors='replace')}"
                raise type(e)(error_message) from None
            raise

    def post_json(self, url: str, **kwargs) -> Any:
        """
        Sends a POST request with a JSON payload and returns the parsed JSON response.

        :param url: The endpoint URL.
        :param kwargs: Additional arguments for the `request` method.
        :return: The parsed JSON response.
        :raises: requests.exceptions.RequestException, json.JSONDecodeError
        """
        response = self.request("POST", url, **kwargs)
        return json.loads(response.decode('utf-8'))

    def post_raw_binary(self, url: str, **kwargs) -> bytes:
        """
        Sends a POST request and returns the raw binary response.

        :param url: The endpoint URL.
        :param kwargs: Additional arguments for the `request` method.
        :return: The raw binary response content.
        :raises: requests.exceptions.RequestException
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
        :raises: requests.exceptions.RequestException
        """
        self._ensure_session()

        # Set stream=True for streaming response
        kwargs['stream'] = True
        if (d := kwargs.get('data')) and isinstance(d, dict):
            kwargs['data'] = json.dumps(d)
        try:
            with self.session.request(method, url, **kwargs) as response:
                if response.status_code != 200:
                    raise Exception(response.content)
                for line in response.iter_lines(decode_unicode=False):
                    if line:
                        yield line.decode('utf-8')
        except RequestException as e:
            # Add context to the error if possible
            if hasattr(e.response, 'content'):
                error_message = f"{str(e)}: {e.response.content.decode('utf-8', errors='replace')}"
                raise type(e)(error_message) from None
            raise
