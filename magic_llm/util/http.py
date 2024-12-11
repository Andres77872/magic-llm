import json

import aiohttp
from typing import Any


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
            response.raise_for_status()
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
    ) -> aiohttp.StreamReader:
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
            response.raise_for_status()
            async for chunk in response.content:
                yield chunk

    async def post_stream(self, url: str, **kwargs) -> aiohttp.StreamReader:
        """
        Sends a POST request and returns a streaming response.

        :param url: The endpoint URL.
        :param kwargs: Additional arguments for the request method.
        :return: An async generator yielding chunks of data.
        """
        async for chunk in self.stream_request("POST", url, **kwargs):
            yield chunk
