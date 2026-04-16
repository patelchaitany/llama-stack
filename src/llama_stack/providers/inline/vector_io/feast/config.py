# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from llama_stack.core.storage.datatypes import KVStoreReference


class FieldMapping(BaseModel):
    """Maps llama-stack's internal field names to the actual field names in an existing Feast feature view."""

    embedding: str = Field(
        description="Name of the embedding/vector field in the Feast feature view",
    )
    chunk_id: str = Field(
        description="Name of the entity/ID field in the Feast feature view",
    )
    chunk_text: str = Field(
        description="Name of the text content field in the Feast feature view",
    )
    chunk_metadata: str | None = Field(
        default=None,
        description=(
            "Name of the metadata field in the Feast feature view. If None, metadata defaults to empty for all chunks."
        ),
    )


class ExistingFeatureViewConfig(BaseModel):
    """Configuration for connecting to an existing Feast feature view as a read-only (or writable) vector store."""

    feature_view_name: str = Field(
        description="Name of the existing Feast feature view",
    )
    vector_store_id: str | None = Field(
        default=None,
        description="Vector store ID to expose in llama-stack. Defaults to feature_view_name.",
    )
    dimension: int = Field(
        description="Embedding dimension of the vectors in this feature view",
    )
    field_mapping: FieldMapping = Field(
        description="Maps llama-stack field names (embedding, chunk_id, chunk_text, chunk_metadata) to the actual field names in the existing Feast feature view",
    )
    read_only: bool = Field(
        default=True,
        description="If true, writes (insert/delete) are blocked. Recommended for externally managed feature views.",
    )
    embedding_model: str = Field(
        default="unknown",
        description="Identifier of the embedding model used to generate vectors in this feature view. Must match the registered_resources entry if one exists.",
    )
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric for vector similarity search (e.g., 'cosine', 'l2', 'inner_product')",
    )
    write_defaults: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Default values for extra fields in the feature view when writing. "
            "Only used when read_only is false. Keys are Feast field names, values are defaults."
        ),
    )

    @model_validator(mode="after")
    def _validate_write_defaults(self) -> "ExistingFeatureViewConfig":
        if self.write_defaults and self.read_only:
            raise ValueError("write_defaults cannot be set when read_only is true")
        return self


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
    existing_feature_views: list[ExistingFeatureViewConfig] | None = Field(
        default=None,
        description=(
            "List of existing Feast feature views to expose as vector stores. "
            "Each entry maps an existing feature view's fields to llama-stack's expected schema. "
            "These feature views must already exist in the Feast registry."
        ),
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
