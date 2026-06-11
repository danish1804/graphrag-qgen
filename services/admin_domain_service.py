from __future__ import annotations

import os
from typing import Any

from services.admin_rebuild_handler import (
    filter_and_skip_duplicates,
    load_chunks_for_files,
    mark_skipped_duplicates,
    store_fingerprints_for_files,
)
from services.admin_upload_handler import (
    assess_before_save,
    duplicate_response_without_save,
    move_temp_to_tested,
    register_new_file_for_rebuild,
    remove_temp_file,
    save_temp_file,
)
from services.document_dedup_service import DocumentDedupService
from utils.registry import load_registry, save_registry


class AdminDomainService:
    """Admin domain workflows extracted from route handlers."""

    def __init__(
        self,
        *,
        tested_dir: str,
        graph_pipeline: Any,
        index_repository: Any,
        runtime_state_module: Any,
        sync_domains_callable: Any,
        dedup_service: DocumentDedupService | None = None,
    ):
        self.tested_dir = tested_dir
        self.graph_pipeline = graph_pipeline
        self.index_repository = index_repository
        self.runtime_state = runtime_state_module
        self.sync_domains = sync_domains_callable
        self.dedup_service = dedup_service or DocumentDedupService(
            tested_dir
        )

    async def upload_document(
        self, *, domain: str, file: Any, valid_domains: set[str]
    ) -> dict[str, Any]:
        if domain not in valid_domains:
            raise ValueError(
                f"Invalid domain '{domain}'. Valid: {valid_domains}"
            )

        original_filename = file.filename
        if not original_filename.lower().endswith((".pdf", ".pptx")):
            raise ValueError("Only PDF and PPTX files supported.")

        content = await file.read()
        ext = os.path.splitext(original_filename)[1]

        existing_path = os.path.join(self.tested_dir, original_filename)
        if os.path.exists(existing_path):
            return self._handle_existing_tested_file(
                domain=domain,
                original_filename=original_filename,
                existing_path=existing_path,
            )

        return self._handle_new_upload(
            domain=domain,
            original_filename=original_filename,
            content=content,
            ext=ext,
        )

    def _handle_new_upload(
        self,
        *,
        domain: str,
        original_filename: str,
        content: bytes,
        ext: str,
    ) -> dict[str, Any]:
        temp_path = save_temp_file(content, suffix=ext)
        try:
            assessment = assess_before_save(
                self.dedup_service, temp_path, domain
            )
            if assessment.should_skip_indexing and assessment.match:
                return duplicate_response_without_save(
                    original_filename=original_filename,
                    domain=domain,
                    assessment=assessment,
                )

            final_filename = self._allocate_filename(ext)
            move_temp_to_tested(temp_path, self.tested_dir, final_filename)
            temp_path = None

            print(
                f"[Admin] New file auto-renamed: {original_filename} "
                f"-> {final_filename}"
            )
            return register_new_file_for_rebuild(
                tested_dir=self.tested_dir,
                domain=domain,
                final_filename=final_filename,
                original_filename=original_filename,
                assessment=assessment,
            )
        finally:
            remove_temp_file(temp_path)

    def _handle_existing_tested_file(
        self,
        *,
        domain: str,
        original_filename: str,
        existing_path: str,
    ) -> dict[str, Any]:
        print(
            f"[Admin] File already in tested/: {original_filename} "
            "— checking dedup before register"
        )

        registry = load_registry()
        domain_files = registry["domains"].get(domain, {}).get("files", {})
        if original_filename in domain_files and domain_files[
            original_filename
        ].get("indexed"):
            return {
                "status": "ok",
                "original_filename": original_filename,
                "saved_as": original_filename,
                "domain": domain,
                "already_indexed": True,
                "saved_to_disk": False,
                "registered": True,
                "message": (
                    f"{original_filename} is already indexed in {domain}. "
                    "No rebuild needed."
                ),
            }

        assessment = assess_before_save(
            self.dedup_service, existing_path, domain
        )
        if assessment.should_skip_indexing and assessment.match:
            return duplicate_response_without_save(
                original_filename=original_filename,
                domain=domain,
                assessment=assessment,
            )

        return register_new_file_for_rebuild(
            tested_dir=self.tested_dir,
            domain=domain,
            final_filename=original_filename,
            original_filename=original_filename,
            assessment=assessment,
        )

    def rebuild_domain(
        self, domain: str, valid_domains: set[str]
    ) -> dict[str, Any]:
        if domain not in valid_domains:
            raise ValueError(f"Invalid domain. Valid: {valid_domains}")

        from utils.registry import load_registry, mark_as_indexed
        from graphrag.graphrag_domain_indexer import build_domain_profiles

        registry = load_registry()
        domain_files = registry["domains"].get(domain, {}).get("files", {})
        new_files = [
            filename
            for filename, info in domain_files.items()
            if not info["indexed"]
        ]

        if not new_files:
            return self._empty_rebuild_response(domain)

        to_index, skipped = filter_and_skip_duplicates(
            self.dedup_service, domain, new_files
        )
        mark_skipped_duplicates(domain, skipped)

        if not to_index:
            return self._all_duplicates_response(domain, skipped)

        all_new_chunks, all_new_sources = load_chunks_for_files(
            self.tested_dir, to_index
        )
        if not all_new_chunks:
            raise ValueError("No chunks loaded from new files.")

        summary, elapsed_seconds = (
            self.index_repository.incremental_update_domain(
                domain=domain,
                new_chunks=all_new_chunks,
                new_sources=all_new_sources,
            )
        )

        for pdf_file in to_index:
            mark_as_indexed(pdf_file, domain=domain)
        store_fingerprints_for_files(
            self.tested_dir, domain, to_index
        )

        self.index_repository.reload_domain(domain)
        build_domain_profiles(self.graph_pipeline.embedder)

        return self._success_rebuild_response(
            domain, to_index, skipped, summary, elapsed_seconds
        )

    def add_domain(
        self, *, domain_name: str, description: str, keywords: list[str]
    ) -> dict[str, Any]:
        from utils.registry import ensure_domain

        valid_domains = self.sync_domains()
        if domain_name in valid_domains:
            raise ValueError(f"Domain '{domain_name}' already exists.")

        self.runtime_state.add_runtime_domain(
            domain_name=domain_name,
            description=description,
            keywords=keywords,
        )
        ensure_domain(domain_name, description)
        self.sync_domains()

        print(f"[Admin] New domain registered: {domain_name}")
        return {
            "status": "ok",
            "domain": domain_name,
            "description": description,
            "message": (
                f"Domain '{domain_name}' registered in runtime state. "
                f"Upload PDFs to data/tested/ then call POST /admin/rebuild-domain "
                f"to build the knowledge graph."
            ),
        }

    def _allocate_filename(self, ext: str) -> str:
        registry = load_registry()
        next_num = registry["last_source_number"] + 1
        final_filename = f"source_{next_num}{ext}"
        final_path = os.path.join(self.tested_dir, final_filename)

        while os.path.exists(final_path):
            next_num += 1
            final_filename = f"source_{next_num}{ext}"
            final_path = os.path.join(self.tested_dir, final_filename)

        registry["last_source_number"] = next_num
        save_registry(registry)
        return final_filename

    def _empty_rebuild_response(self, domain: str) -> dict[str, Any]:
        return {
            "status": "ok",
            "domain": domain,
            "update_type": "incremental",
            "message": "No new documents found. All files already indexed.",
            "new_nodes_added": 0,
            "new_edges_added": 0,
            "skipped_duplicates": [],
        }

    def _all_duplicates_response(
        self, domain: str, skipped: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "status": "ok",
            "domain": domain,
            "update_type": "incremental",
            "new_files": [],
            "skipped_duplicates": skipped,
            "new_nodes_added": 0,
            "new_edges_added": 0,
            "elapsed_seconds": 0.0,
            "message": (
                f"No graph changes: {len(skipped)} file(s) matched existing "
                "indexed content (hash or similarity). See skipped_duplicates."
            ),
        }

    def _success_rebuild_response(
        self,
        domain: str,
        to_index: list[str],
        skipped: list[dict[str, Any]],
        summary: dict[str, Any],
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        return {
            "status": "ok",
            "domain": domain,
            "update_type": "incremental",
            "new_files": to_index,
            "skipped_duplicates": skipped,
            "new_nodes_added": summary["new_nodes_added"],
            "nodes_updated": summary["nodes_updated"],
            "new_edges_added": summary["new_edges_added"],
            "graph_nodes": summary["graph_nodes"],
            "graph_edges": summary["graph_edges"],
            "elapsed_seconds": elapsed_seconds,
            "message": (
                f"Domain '{domain}' updated. {summary['new_nodes_added']} "
                f"new concepts added from {len(to_index)} file(s). "
                f"{len(skipped)} duplicate(s) skipped."
            ),
        }
