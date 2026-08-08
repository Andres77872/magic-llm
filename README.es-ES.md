

# Magic LLM

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Andres77872/magic-llm)

Magic LLM es una biblioteca cliente para Python 3.10+ que expone una única interfaz `MagicLLM` a través de proveedores nativos de LLM y compatibles con OpenAI. La versión `0.1.37` admite chat, streaming, llamadas asíncronas, embeddings, audio específico del proveedor, descubrimiento de modelos, invocación de herramientas (tool calling), agentes estilo ReAct y subagentes respaldados por YAML.

> Magic LLM es una biblioteca cliente. No incluye **una** CLI, servidor REST ni cargador de `.env`. Pasa las credenciales explícitamente a `MagicLLM(...)` desde la configuración de tu propia aplicación.

## Enlaces rápidos

- [Centro de documentación](docs/index.md)
- [Instalación](docs/installation.md)
- [Inicio rápido en 5 minutos](docs/quickstart.md)
- [Guía de proveedores](docs/providers/index.md)
- [Uso de chat](docs/usage/chat.md)
- [Invocación de herramientas y agentes](docs/agents/index.md)
- [Solución de problemas](docs/troubleshooting.md)
- [Configuración de desarrollo](docs/development/setup.md)
- [Pruebas](docs/development/testing.md)

## Características

- Constructor unificado `MagicLLM(engine=..., model=..., private_key=...)`.
- Métodos públicos de chat: `generate`, `stream_generate`, `async_generate` y `async_stream_generate`.
- Motores nativos: `openai`, `google`, `cloudflare`, `amazon`, `cohere`, `anthropic` y `azure`.
- Enrutamiento compatible con OpenAI mediante `engine='openai'` junto con `base_url` para Groq, SambaNova, OpenRouter, Mistral, Fireworks, DeepSeek, DeepInfra, Together y endpoints similares.
- Soporte para streaming y llamadas asíncronas en toda la superficie de chat principal.
- Modelos de respuesta unificados con metadatos de uso y latencia cuando los proveedores los exponen.
- Embeddings, conversión de voz a texto, texto a voz, descubrimiento de modelos, clientes de respaldo, callbacks, invocación de herramientas, agentes ReAct y subagentes.
- La visión significa entrada de imágenes para el chat. La generación de imágenes/salida de imágenes de primera clase no forma parte de la API principal actual; las herramientas de usuario llamadas `generate_image` son herramientas gestionadas por el llamador.

## Instalación

Ruta principal de instalación:

```bash
pip install git+https://github.com/Andres77872/magic-llm.git
```

Si el paquete está publicado en tu entorno, esto también podría funcionar:

```bash
pip install magic-llm
```

Los metadatos del proyecto no declaran una versión mínima de Python, pero el código fuente utiliza sintaxis de Python 3.10+. Utiliza Python 3.10 o una versión posterior.

## Inicio rápido en 5 minutos

```python
from magic_llm import MagicLLM
from magic_llm.model import ModelChat

client = MagicLLM(
    engine="openai",
    model="gpt-4o-mini",
    private_key="sk-your-key",
)

chat = ModelChat(system="You are a concise assistant.")
chat.add_user_message("Explain what a vector embedding is in one paragraph.")

response = client.llm.generate(chat)
print(response.content)
```

El streaming utiliza el mismo objeto de chat:

```python
for chunk in client.llm.stream_generate(chat):
    text = chunk.choices[0].delta.content or ""
    print(text, end="", flush=True)
```

Streaming asíncrono:

```python
async for chunk in client.llm.async_stream_generate(chat):
    text = chunk.choices[0].delta.content or ""
    print(text, end="", flush=True)
```

## Resumen de proveedores

| Familia de proveedor | Motor | Notas |
| --- | --- | --- |
| OpenAI | `openai` | Endpoint oficial de OpenAI por defecto. |
| Proveedores compatibles con OpenAI | `openai` + `base_url` | Groq, SambaNova, OpenRouter, Mistral, Fireworks, DeepSeek, DeepInfra, Together y otros se seleccionan mediante coincidencia de URL. |
| Anthropic | `anthropic` | Soporte nativo para la API de Claude. |
| Google AI Studio | `google` | Formato nativo de solicitudes/respuestas de Gemini. |
| AWS Bedrock | `amazon` | Enruta según el prefijo del modelo de Bedrock. El descubrimiento de modelos no es compatible. |
| Cloudflare Workers AI | `cloudflare` | Requiere `account_id`. La invocación de herramientas y el descubrimiento de modelos no son compatibles. |
| Cohere | `cohere` | Formato nativo de chat de Cohere. La invocación de herramientas no es compatible. |
| Azure | `azure` | Motor solo para voz. Los métodos de generación de chat generan `NotImplementedError`. |

Consulta [providers/index.md](docs/providers/index.md) y las páginas específicas de cada proveedor para obtener credenciales y ejemplos.

## Proveedores compatibles con OpenAI

Utiliza `engine='openai'` y establece `base_url`:

```python
client = MagicLLM(
    engine="openai",
    model="llama-3.1-8b-instant",
    private_key="gsk-your-key",
    base_url="https://api.groq.com/openai/v1",
)
```

Magic LLM detecta las URLs conocidas de los proveedores y aplica adaptadores específicos del proveedor cuando están implementados.

## Puntos importantes a tener en cuenta

- **Argumento oficial de tokens de OpenAI:** solo para `api.openai.com`, `max_tokens` se transforma en `max_completion_tokens`. Si se proporcionan ambos, `max_completion_tokens` prevalece y `max_tokens` se elimina. Otros endpoints compatibles con OpenAI mantienen `max_tokens` sin cambios.
- **Azure es solo para voz:** `generate`, `stream_generate`, `async_generate` y `async_stream_generate` generan `NotImplementedError` para `engine='azure'`.
- **El soporte de audio es específico del proveedor/método:** las rutas TTS/STT no compatibles fallan rápidamente en lugar de devolver `None`; consulta `docs/usage/audio.md` para la matriz síncrona/asíncrona y los requisitos de metadatos de carga de STT.
- **Sin generación de imágenes principal:** Magic LLM admite entrada de visión/imágenes para modelos de chat compatibles, no la generación de salida de texto a imagen.
- **Limitaciones en el descubrimiento de modelos:** Amazon y Cloudflare no admiten el descubrimiento de modelos.
- **Limitaciones en la invocación de herramientas:** Amazon Bedrock, Cohere y Cloudflare no admiten la invocación de herramientas.
- **Los subagentes están deshabilitados por defecto:** llama a `enable_subagents()` antes de `load_subagents()` o obtendrás un paquete vacío.
- **Sin carga de `.env`:** lee los secretos desde tu propio gestor de secretos o capa de entorno, y luego pásalos como argumentos del constructor.

## Agentes y invocación de herramientas

```python
from magic_llm import MagicLLM

client = MagicLLM(engine="openai", model="gpt-4o-mini", private_key="sk-your-key")

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

response = client.run_agent(
    user_input="Use the tool to add 17 and 25, then explain the result.",
    tools=[add],
    max_iterations=4,
)

print(response.content)
```

Las especificaciones de herramientas pueden ser funciones invocables de Python, esquemas JSON al estilo de OpenAI con `tool_functions`, o clases de modelos Pydantic. Consulta [agents/tool-calling.md](docs/agents/tool-calling.md).

## Descubrimiento de modelos

```python
client = MagicLLM(engine="openai", private_key="sk-your-key")

for model in client.list_models():
    print(model.external_id, model.capabilities.chat)
```

El descubrimiento depende del proveedor. Amazon y Cloudflare no son compatibles explícitamente. Consulta [usage/model-discovery.md](docs/usage/model-discovery.md).

## Desarrollo

```bash
git clone https://github.com/Andres77872/magic-llm.git
cd magic-llm
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Comando de prueba utilizado por el proyecto:

```bash
pytest test/ -v
```

No hagas commit de claves reales. Las pruebas de integración utilizan `MAGIC_LLM_KEYS`, `MAGIC_LLM_AUDIO_FILE` y `MAGIC_LLM_IMAGE_B64_FILE`; el registro de carga útil de depuración utiliza `MAGIC_LLM_DEBUG_PAYLOAD` y `MAGIC_LLM_DEBUG_PAYLOAD_FULL`.

Más detalles: [development/setup.md](docs/development/setup.md), [development/testing.md](docs/development/testing.md) y [development/contributing.md](docs/development/contributing.md).
