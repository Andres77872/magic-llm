import aiohttp


async def async_http_post_json(**kwargs):
    async with aiohttp.ClientSession() as session:
        async with session.post(**kwargs) as response:
            response.raise_for_status()
            return await response.json()


async def async_http_post_raw_binary(**kwargs):
    async with aiohttp.ClientSession() as session:
        async with session.post(**kwargs) as response:
            response.raise_for_status()
            return await response.read()
