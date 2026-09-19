"""Architecture tests ensuring Amazon REST adapters remain SDK-client free."""

import ast
import importlib
import inspect

import pytest

from magic_llm.engine.amazon_adapters.amazon_nova import ProviderAmazonNova


AMAZON_MODULES = (
    "magic_llm.engine.amazon_adapters.base_provider",
    "magic_llm.engine.engine_amazon",
    "magic_llm.engine.amazon_adapters.amazon_nova",
    "magic_llm.engine.amazon_adapters.amazon_titan",
    "magic_llm.engine.amazon_adapters.amazon_anthropic",
    "magic_llm.engine.amazon_adapters.amazon_meta",
)
FORBIDDEN_SDKS = {"boto3", "aioboto3"}


def _module_tree(module_name: str) -> ast.Module:
    return ast.parse(inspect.getsource(importlib.import_module(module_name)))


@pytest.mark.parametrize("module_name", AMAZON_MODULES, ids=lambda value: value.rsplit(".", 1)[-1])
def test_amazon_modules_do_not_import_sdk_clients(module_name):
    imported_roots = set()
    for node in ast.walk(_module_tree(module_name)):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots.isdisjoint(FORBIDDEN_SDKS)


@pytest.mark.parametrize("attribute", ["client", "aclient"])
def test_amazon_base_provider_does_not_assign_sdk_client_attributes(attribute):
    assigned_attributes = {
        target.attr
        for node in ast.walk(_module_tree(AMAZON_MODULES[0]))
        for target in (
            list(node.targets) if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign)
            else []
        )
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        )
    }

    assert attribute not in assigned_attributes


@pytest.mark.parametrize("attribute", ["client", "aclient"])
def test_amazon_provider_instances_have_no_sdk_clients(attribute):
    provider = ProviderAmazonNova(
        aws_access_key_id="test-access-key-id",
        aws_secret_access_key="test-secret-access-key",
        region_name="us-east-1",
        model="amazon.nova-lite-v1:0",
    )

    assert getattr(provider, attribute, None) is None
