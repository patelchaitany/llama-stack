# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import os
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from feast import Entity, FeatureStore, FeatureView, FileSource
from feast import Field as FeastField
from feast.repo_config import RepoConfig
from feast.types import Array, Float32, String
from feast.value_type import ValueType
from numpy.typing import NDArray

from llama_stack.core.storage.kvstore import kvstore_impl
from llama_stack.log import get_logger
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import (
    RERANKER_TYPE_RRF,
    EmbeddingIndex,
    VectorStoreWithIndex,
)
from llama_stack.providers.utils.vector_io import load_embedded_chunk_with_backward_compat
from llama_stack.providers.utils.vector_io.vector_utils import WeightedInMemoryAggregator
from llama_stack_api import (
    ChunkForDeletion,
    DeleteChunksRequest,
    EmbeddedChunk,
    Files,
    Inference,
    InsertChunksRequest,
    QueryChunksRequest,
    QueryChunksResponse,
    VectorIO,
    VectorStore,
    VectorStoreNotFoundError,
    VectorStoresProtocolPrivate,
)

from .config import FeastVectorIOConfig, FieldMapping

logger = get_logger(name=__name__, category="vector_io")


def _sanitize_feast_name(name: str) -> str:
    """Sanitize a name for use as a Feast entity/feature view name."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


class FeastIndex(EmbeddingIndex):
    """
    An EmbeddingIndex backed by a Feast FeatureStore.

    Each llama-stack VectorStore maps to a Feast FeatureView with a vector-indexed
    embedding field. Data is written via write_to_online_store() and queried via
    retrieve_online_documents_v2().

    For existing (externally managed) feature views, a FieldMapping translates between
    the Feast schema and llama-stack's internal field names. External feature views
    are never deleted by llama-stack.
    """

    def __init__(
        self,
        feast_store: FeatureStore,
        feature_view_name: str,
        dimension: int,
        distance_metric: str = "COSINE",
        field_mapping: FieldMapping | None = None,
        read_only: bool = False,
        write_defaults: dict[str, Any] | None = None,
    ):
        self.feast_store = feast_store
        self.feature_view_name = feature_view_name
        self.dimension = dimension
        self.distance_metric = distance_metric
        self.field_mapping = field_mapping
        self.read_only = read_only
        self.write_defaults = write_defaults
        self.is_external = field_mapping is not None

    @property
    def _embedding_field(self) -> str:
        return self.field_mapping.embedding if self.field_mapping else "embedding"

    @property
    def _chunk_id_field(self) -> str:
        return self.field_mapping.chunk_id if self.field_mapping else "chunk_id"

    @property
    def _chunk_text_field(self) -> str:
        return self.field_mapping.chunk_text if self.field_mapping else "chunk_text"

    @property
    def _chunk_metadata_field(self) -> str | None:
        if self.field_mapping:
            return self.field_mapping.chunk_metadata
        return "chunk_metadata"

    def _build_features(self) -> list[str]:
        """Build the Feast feature list using mapped field names."""
        fv = self.feature_view_name
        features = [
            f"{fv}:{self._embedding_field}",
            f"{fv}:{self._chunk_text_field}",
            f"{fv}:{self._chunk_id_field}",
        ]
        if self._chunk_metadata_field:
            features.append(f"{fv}:{self._chunk_metadata_field}")
        return features

    @classmethod
    def create(
        cls,
        config: FeastVectorIOConfig,
        feast_store: FeatureStore,
        store_id: str,
        dimension: int,
    ) -> "FeastIndex":
        """Create a new FeastIndex, registering the entity and feature view in Feast."""
        feature_view_name = _sanitize_feast_name(store_id)
        distance_metric = config.online_store.get(
            "metric_type", config.online_store.get("similarity", "cosine")
        ).lower()

        entity = Entity(
            name=f"{feature_view_name}_chunk",
            join_keys=["chunk_id"],
            value_type=ValueType.STRING,
        )

        repo_path = config.get_repo_path()
        os.makedirs(repo_path, exist_ok=True)
        placeholder_path = os.path.join(repo_path, f"placeholder_{feature_view_name}.parquet")
        if not os.path.exists(placeholder_path):
            placeholder_df = pd.DataFrame(
                {
                    "chunk_id": pd.Series(dtype="str"),
                    "chunk_text": pd.Series(dtype="str"),
                    "chunk_metadata": pd.Series(dtype="str"),
                    "embedding": pd.Series(dtype="object"),
                    "event_timestamp": pd.Series(dtype="datetime64[ns]"),
                }
            )
            placeholder_df.to_parquet(placeholder_path)

        source = FileSource(
            path=placeholder_path,
            timestamp_field="event_timestamp",
        )
        fv = FeatureView(
            name=feature_view_name,
            entities=[entity],
            schema=[
                FeastField(
                    name="embedding",
                    dtype=Array(Float32),
                    vector_index=True,
                    vector_search_metric=distance_metric,
                    vector_length=dimension,
                ),
                FeastField(name="chunk_id", dtype=String),
                FeastField(name="chunk_text", dtype=String),
                FeastField(name="chunk_metadata", dtype=String),
            ],
            source=source,
            ttl=timedelta(days=365),
        )

        feast_store.apply([entity, fv])

        return cls(
            feast_store=feast_store,
            feature_view_name=feature_view_name,
            dimension=dimension,
            distance_metric=distance_metric,
        )

    async def add_chunks(self, embedded_chunks: list[EmbeddedChunk], batch_size: int = 500) -> None:
        if self.read_only:
            raise PermissionError(
                f"Failed to write to feature view '{self.feature_view_name}': it is configured as read-only"
            )

        if not embedded_chunks:
            return

        rows = []
        for chunk in embedded_chunks:
            content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            metadata_json = json.dumps(chunk.metadata) if chunk.metadata else "{}"
            row: dict[str, Any] = {
                self._chunk_id_field: chunk.chunk_id,
                self._chunk_text_field: content,
                self._embedding_field: list(chunk.embedding),
                "event_timestamp": datetime.now(tz=UTC),
            }
            if self._chunk_metadata_field:
                row[self._chunk_metadata_field] = metadata_json
            if self.write_defaults:
                for key, value in self.write_defaults.items():
                    if key not in row:
                        row[key] = value
            rows.append(row)

        df = pd.DataFrame(rows)

        self.feast_store.write_to_online_store(
            feature_view_name=self.feature_view_name,
            df=df,
        )

    async def query_vector(
        self, embedding: NDArray, k: int, score_threshold: float, filters: Any = None
    ) -> QueryChunksResponse:
        query_emb = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
        features = self._build_features()
        distance_metric = self.distance_metric

        def _query():
            return self.feast_store.retrieve_online_documents_v2(
                features=features,
                query=query_emb,
                top_k=k,
                distance_metric=distance_metric,
            )

        response = await asyncio.to_thread(_query)
        return self._parse_feast_response(response, score_threshold)

    async def query_keyword(
        self, query_string: str, k: int, score_threshold: float, filters: Any = None
    ) -> QueryChunksResponse:
        features = self._build_features()

        def _query():
            return self.feast_store.retrieve_online_documents_v2(
                features=features,
                query_string=query_string,
                top_k=k,
            )

        response = await asyncio.to_thread(_query)
        return self._parse_feast_response(response, score_threshold)

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str = RERANKER_TYPE_RRF,
        reranker_params: dict[str, Any] | None = None,
        filters: Any = None,
    ) -> QueryChunksResponse:
        if reranker_params is None:
            reranker_params = {}

        vector_response = await self.query_vector(embedding, k, score_threshold)
        keyword_response = await self.query_keyword(query_string, k, score_threshold)

        vector_scores = {
            ec.chunk_id: score for ec, score in zip(vector_response.chunks, vector_response.scores, strict=False)
        }
        keyword_scores = {
            ec.chunk_id: score for ec, score in zip(keyword_response.chunks, keyword_response.scores, strict=False)
        }

        combined_scores = WeightedInMemoryAggregator.combine_search_results(
            vector_scores, keyword_scores, reranker_type, reranker_params
        )

        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_items = sorted_items[:k]
        filtered_items = [(doc_id, score) for doc_id, score in top_k_items if score >= score_threshold]

        chunk_map = {ec.chunk_id: ec for ec in vector_response.chunks + keyword_response.chunks}

        chunks: list[EmbeddedChunk] = []
        scores: list[float] = []
        for doc_id, score in filtered_items:
            if doc_id in chunk_map:
                chunks.append(chunk_map[doc_id])
                scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        if self.read_only:
            raise PermissionError(
                f"Failed to delete chunks from feature view '{self.feature_view_name}': it is configured as read-only"
            )
        raise NotImplementedError(
            "Feast does not support individual record deletion from the online store. "
            "To remove data, re-create the vector store."
        )

    async def delete(self) -> None:
        if self.is_external:
            logger.info(
                "Skipping deletion of externally managed Feast feature view",
                feature_view=self.feature_view_name,
            )
            return

        fv_name = self.feature_view_name

        def _teardown():
            try:
                fv = self.feast_store.get_feature_view(fv_name)
                self.feast_store.apply(objects=[], objects_to_delete=[fv], partial=False)
            except Exception:
                logger.warning("Failed to delete Feast feature view, it may not exist", feature_view=fv_name)

        await asyncio.to_thread(_teardown)

    def _parse_feast_response(self, response: Any, score_threshold: float) -> QueryChunksResponse:
        """Convert a Feast OnlineResponse into a QueryChunksResponse."""
        result_df = response.to_df()
        chunks: list[EmbeddedChunk] = []
        scores: list[float] = []

        if result_df.empty:
            return QueryChunksResponse(chunks=chunks, scores=scores)

        for _, row in result_df.iterrows():
            score = float(row.get("distance", 0.0))
            similarity = 1.0 / (1.0 + score) if score >= 0 else 1.0

            if similarity < score_threshold:
                continue

            embedding = row.get(self._embedding_field, [])
            if embedding is None:
                embedding = []
            embedding = list(embedding) if not isinstance(embedding, list) else embedding

            if self._chunk_metadata_field:
                metadata_str = row.get(self._chunk_metadata_field, "{}")
            else:
                metadata_str = "{}"
            try:
                metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else {}
            except (json.JSONDecodeError, TypeError):
                metadata = {}

            chunk_data = {
                "content": row.get(self._chunk_text_field, ""),
                "chunk_id": row.get(self._chunk_id_field, ""),
                "metadata": metadata,
                "chunk_metadata": {},
                "embedding": embedding,
                "embedding_model": "unknown",
                "embedding_dimension": len(embedding),
            }

            try:
                embedded_chunk = load_embedded_chunk_with_backward_compat(chunk_data)
            except Exception as e:
                logger.error("Failed to parse Feast result row", error=str(e))
                continue

            chunks.append(embedded_chunk)
            scores.append(similarity)

        return QueryChunksResponse(chunks=chunks, scores=scores)


VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_stores:feast:{VERSION}::"


class FeastVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorStoresProtocolPrivate):
    """
    A VectorIO implementation backed by a Feast FeatureStore.

    Each registered VectorStore maps to a Feast FeatureView with a vector-indexed
    embedding field. Metadata is persisted via llama-stack's KVStore, while vector
    data lives in Feast's online store.
    """

    def __init__(self, config: FeastVectorIOConfig, inference_api: Inference, files_api: Files | None) -> None:
        super().__init__(inference_api=inference_api, files_api=files_api, kvstore=None)
        self.config = config
        self.cache: dict[str, VectorStoreWithIndex] = {}
        self.vector_store_table = None
        self.feast_store: FeatureStore | None = None
        self._external_store_ids: set[str] = set()

    def _build_feast_store(self) -> FeatureStore:
        """Build a Feast FeatureStore from the provider config."""
        registry = self.config.registry
        online_store = self.config.online_store
        offline_store = self.config.offline_store or "file"
        repo_path = self.config.get_repo_path()

        os.makedirs(repo_path, exist_ok=True)

        repo_config = RepoConfig(
            project=self.config.project,
            provider=self.config.provider,
            registry=registry,
            online_store=online_store,
            offline_store=offline_store,
            entity_key_serialization_version=self.config.entity_key_serialization_version,
            repo_path=repo_path,
        )
        return FeatureStore(config=repo_config)

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.persistence)
        self.feast_store = self._build_feast_store()

        if self.config.existing_feature_views:
            for fv_config in self.config.existing_feature_views:
                store_id = fv_config.vector_store_id or fv_config.feature_view_name
                index = FeastIndex(
                    feast_store=self.feast_store,
                    feature_view_name=fv_config.feature_view_name,
                    dimension=fv_config.dimension,
                    distance_metric=fv_config.distance_metric,
                    field_mapping=fv_config.field_mapping,
                    read_only=fv_config.read_only,
                    write_defaults=fv_config.write_defaults,
                )
                vector_store = VectorStore(
                    identifier=store_id,
                    provider_resource_id=fv_config.feature_view_name,
                    provider_id="feast",
                    embedding_model=fv_config.embedding_model,
                    embedding_dimension=fv_config.dimension,
                )
                self.cache[store_id] = VectorStoreWithIndex(vector_store, index, self.inference_api)
                self._external_store_ids.add(store_id)
                logger.info(
                    "Registered existing Feast feature view as vector store",
                    vector_store_id=store_id,
                    feature_view=fv_config.feature_view_name,
                    read_only=fv_config.read_only,
                )

        start_key = VECTOR_DBS_PREFIX
        end_key = f"{VECTOR_DBS_PREFIX}\xff"
        stored_vector_stores = await self.kvstore.values_in_range(start_key, end_key)
        for db_json in stored_vector_stores:
            vector_store = VectorStore.model_validate_json(db_json)
            if vector_store.identifier in self._external_store_ids:
                continue
            feature_view_name = _sanitize_feast_name(vector_store.identifier)
            distance_metric = self.config.online_store.get(
                "metric_type", self.config.online_store.get("similarity", "cosine")
            ).lower()
            index = FeastIndex(
                feast_store=self.feast_store,
                feature_view_name=feature_view_name,
                dimension=vector_store.embedding_dimension,
                distance_metric=distance_metric,
            )
            self.cache[vector_store.identifier] = VectorStoreWithIndex(vector_store, index, self.inference_api)

        await self.initialize_openai_vector_stores()

    async def shutdown(self) -> None:
        await super().shutdown()

    async def list_vector_stores(self) -> list[VectorStore]:
        return [v.vector_store for v in self.cache.values()]

    async def register_vector_store(self, vector_store: VectorStore) -> None:
        if self.kvstore is None or self.feast_store is None:
            raise RuntimeError("Not initialized. Call initialize() first.")

        if vector_store.identifier in self._external_store_ids:
            logger.info(
                "Vector store already managed as existing feature view, skipping registration",
                vector_store_id=vector_store.identifier,
            )
            return

        key = f"{VECTOR_DBS_PREFIX}{vector_store.identifier}"
        await self.kvstore.set(key=key, value=vector_store.model_dump_json())

        def _create_index():
            return FeastIndex.create(
                config=self.config,
                feast_store=self.feast_store,
                store_id=vector_store.identifier,
                dimension=vector_store.embedding_dimension,
            )

        index = await asyncio.to_thread(_create_index)
        self.cache[vector_store.identifier] = VectorStoreWithIndex(vector_store, index, self.inference_api)

    async def _get_and_cache_vector_store_index(self, vector_store_id: str) -> VectorStoreWithIndex | None:
        if vector_store_id in self.cache:
            return self.cache[vector_store_id]

        if self.kvstore is None or self.feast_store is None:
            raise RuntimeError("Not initialized. Call initialize() first.")

        key = f"{VECTOR_DBS_PREFIX}{vector_store_id}"
        vector_store_data = await self.kvstore.get(key)
        if not vector_store_data:
            raise VectorStoreNotFoundError(vector_store_id)

        vector_store = VectorStore.model_validate_json(vector_store_data)
        feature_view_name = _sanitize_feast_name(vector_store_id)
        distance_metric = self.config.online_store.get(
            "metric_type", self.config.online_store.get("similarity", "cosine")
        ).lower()

        index = VectorStoreWithIndex(
            vector_store=vector_store,
            index=FeastIndex(
                feast_store=self.feast_store,
                feature_view_name=feature_view_name,
                dimension=vector_store.embedding_dimension,
                distance_metric=distance_metric,
            ),
            inference_api=self.inference_api,
        )
        self.cache[vector_store_id] = index
        return index

    async def unregister_vector_store(self, vector_store_id: str) -> None:
        if vector_store_id in self._external_store_ids:
            raise ValueError(
                f"Failed to unregister vector store '{vector_store_id}': "
                "it is an externally managed Feast feature view. "
                "Remove it from the 'existing_feature_views' config instead."
            )

        if vector_store_id in self.cache:
            await self.cache[vector_store_id].index.delete()
            del self.cache[vector_store_id]

        if self.kvstore is not None:
            key = f"{VECTOR_DBS_PREFIX}{vector_store_id}"
            await self.kvstore.delete(key)

    async def insert_chunks(self, request: InsertChunksRequest) -> None:
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not index:
            raise VectorStoreNotFoundError(request.vector_store_id)
        await index.insert_chunks(request)

    async def query_chunks(self, request: QueryChunksRequest) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not index:
            raise VectorStoreNotFoundError(request.vector_store_id)
        return await index.query_chunks(request)

    async def delete_chunks(self, request: DeleteChunksRequest) -> None:
        index = await self._get_and_cache_vector_store_index(request.vector_store_id)
        if not index:
            raise VectorStoreNotFoundError(request.vector_store_id)
        await index.index.delete_chunks(request.chunks)
