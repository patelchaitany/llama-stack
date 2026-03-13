# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from llama_stack.core.storage.datatypes import KVStoreReference


class FeastVectorIOConfig(BaseModel):
    """Configuration for the Feast-backed VectorIO provider."""

    project: str = Field(
        default="llama_stack_vector_io",
        description="Feast project name",
    )
    provider: str = Field(
        default="local",
        description="Feast provider type (e.g., 'local', 'gcp', 'aws')",
    )
    online_store: dict[str, Any] = Field(
        description=(
            "Feast online store configuration passed directly to RepoConfig. "
            "Backend-specific options vary by type (milvus, sqlite, pgvector, etc.). "
            "Common keys: type, path, vector_enabled, embedding_dim, index_type, metric_type."
        ),
    )
    registry: str | dict[str, Any] = Field(
        description=(
            "Feast registry configuration. Either a path string for file-based registry "
            "(e.g., '/tmp/feast/registry.db') or a dict with registry_type, path, "
            "cache_ttl_seconds, etc."
        ),
    )
    offline_store: str | dict[str, Any] | None = Field(
        default="file",
        description=(
            "Feast offline store configuration. Not required for vector_io operations. "
            "Defaults to 'file' (local file-based)."
        ),
    )
    entity_key_serialization_version: int = Field(
        default=3,
        description="Feast entity key serialization version",
    )
    persistence: KVStoreReference = Field(
        description="Config for KV store backend used by Llama Stack for metadata persistence",
    )

    @model_validator(mode="after")
    def _validate_online_store(self) -> "FeastVectorIOConfig":
        if "type" not in self.online_store:
            raise ValueError("online_store must include a 'type' key (e.g., 'milvus', 'sqlite')")
        return self

    def get_repo_path(self) -> str:
        """Derive repo_path from the registry path when file-based, otherwise use a temp directory."""
        if isinstance(self.registry, str):
            return str(Path(self.registry).parent)
        registry_path = self.registry.get("path", "")
        if registry_path and not registry_path.startswith(("postgresql://", "postgresql+", "mysql://", "snowflake://")):
            return str(Path(registry_path).parent)
        return "/tmp/feast_repo"

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "project": "llama_stack_vector_io",
            "provider": "local",
            "online_store": {
                "type": "milvus",
                "path": "${env.FEAST_ONLINE_STORE_PATH:=" + __distro_dir__ + "}/feast_online.db",
                "vector_enabled": True,
                "embedding_dim": 384,
                "metric_type": "COSINE",
            },
            "registry": "${env.FEAST_REGISTRY_PATH:=" + __distro_dir__ + "}/feast_registry.db",
            "persistence": KVStoreReference(
                backend="kv_default",
                namespace="vector_io::feast",
            ).model_dump(exclude_none=True),
        }
