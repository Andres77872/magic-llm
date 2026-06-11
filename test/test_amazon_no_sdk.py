"""Tests verifying Amazon engine/adapters do not use boto3/aioboto3 SDK clients."""
import ast
import importlib
import inspect

import pytest


class TestAmazonNoSdkImports:
    """Verify no boto3/aioboto3 imports in Amazon-related modules."""

    def test_base_provider_no_boto3_import(self):
        """AmazonBaseProvider module does not import boto3."""
        import magic_llm.engine.amazon_adapters.base_provider as mod
        source = inspect.getsource(mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in ("boto3", "aioboto3"), \
                        f"Found import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith(("boto3", "aioboto3")):
                    pytest.fail(f"Found from import: {node.module}")

    def test_engine_amazon_no_boto3_import(self):
        """EngineAmazon module does not import boto3 or aioboto3."""
        import magic_llm.engine.engine_amazon as mod
        source = inspect.getsource(mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in ("boto3", "aioboto3"), \
                        f"Found import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith(("boto3", "aioboto3")):
                    pytest.fail(f"Found from import: {node.module}")

    def test_amazon_nova_no_boto3_import(self):
        """Amazon Nova adapter does not import boto3 or aioboto3."""
        import magic_llm.engine.amazon_adapters.amazon_nova as mod
        source = inspect.getsource(mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in ("boto3", "aioboto3"), \
                        f"Found import: {alias.name}"

    def test_amazon_titan_no_boto3_import(self):
        """Amazon Titan adapter does not import boto3 or aioboto3."""
        import magic_llm.engine.amazon_adapters.amazon_titan as mod
        source = inspect.getsource(mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in ("boto3", "aioboto3"), \
                        f"Found import: {alias.name}"

    def test_amazon_anthropic_no_boto3_import(self):
        """Amazon Anthropic adapter does not import boto3 or aioboto3."""
        import magic_llm.engine.amazon_adapters.amazon_anthropic as mod
        source = inspect.getsource(mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in ("boto3", "aioboto3"), \
                        f"Found import: {alias.name}"

    def test_amazon_meta_no_boto3_import(self):
        """Amazon Meta adapter does not import boto3 or aioboto3."""
        import magic_llm.engine.amazon_adapters.amazon_meta as mod
        source = inspect.getsource(mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in ("boto3", "aioboto3"), \
                        f"Found import: {alias.name}"


class TestAmazonNoSdkClientAttributes:
    """Verify AmazonBaseProvider does not create SDK client attributes."""

    def test_base_provider_no_client_attribute(self):
        """AmazonBaseProvider has no runtime SDK client assignment."""
        import magic_llm.engine.amazon_adapters.base_provider as mod
        source = inspect.getsource(mod)
        tree = ast.parse(source)
        assignments = []
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == "client"
                ):
                    assignments.append(target.attr)
        assert assignments == []

    def test_base_provider_no_aclient_attribute(self):
        """AmazonBaseProvider has no runtime async SDK client assignment."""
        import magic_llm.engine.amazon_adapters.base_provider as mod
        source = inspect.getsource(mod)
        tree = ast.parse(source)
        assignments = []
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == "aclient"
                ):
                    assignments.append(target.attr)
        assert assignments == []

    def test_base_provider_subclass_no_client(self):
        """AmazonBaseProvider subclass instances have no self.client."""
        from magic_llm.engine.amazon_adapters.amazon_nova import ProviderAmazonNova
        provider = ProviderAmazonNova(
            aws_access_key_id="test",
            aws_secret_access_key="test",
            region_name="us-east-1",
            model="amazon.nova-lite-v1:0",
        )
        assert not hasattr(provider, "client") or provider.client is None

    def test_base_provider_subclass_no_aclient(self):
        """AmazonBaseProvider subclass instances have no self.aclient."""
        from magic_llm.engine.amazon_adapters.amazon_nova import ProviderAmazonNova
        provider = ProviderAmazonNova(
            aws_access_key_id="test",
            aws_secret_access_key="test",
            region_name="us-east-1",
            model="amazon.nova-lite-v1:0",
        )
        assert not hasattr(provider, "aclient")
