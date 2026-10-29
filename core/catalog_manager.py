"""Zero-training SKU Catalog Onboarding and DINOv2 Embedding Cache.

Implements Stage 1 of ReturnShield AI:
- SKU profile and Bill of Materials (BOM) management.
- DINOv2 (Vision Transformer) feature extraction (global CLS + patch tokens).
- Fast local vector embedding caching and cosine similarity search.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from core.schemas import BOMItem, SKUOnboardRequest, SKUProfile

logger = logging.getLogger("returnshield.catalog_manager")


class DINOv2EmbeddingEngine:
    """Extracts foundation visual embeddings using pre-trained DINOv2 models."""

    _instance: Optional["DINOv2EmbeddingEngine"] = None

    def __init__(self, model_name: str = "dinov2_vits14", device: Optional[str] = None):
        """Initializes the DINOv2 backbone and transforms.

        Args:
            model_name: Torch Hub model identifier (e.g. 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitb14_reg').
            device: 'cuda' or 'cpu'. Defaults to auto-detecting CUDA.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[torch.nn.Module] = None
        self._transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @classmethod
    def get_shared_engine(cls, model_name: str = "dinov2_vits14", device: Optional[str] = None) -> "DINOv2EmbeddingEngine":
        """Singleton accessor to prevent redundant model weights loading across calls."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name=model_name, device=device)
        return cls._instance

    @property
    def model(self) -> torch.nn.Module:
        """Lazy-loads the DINOv2 model on first invocation."""
        if self._model is None:
            logger.info("Loading DINOv2 model '%s' on %s...", self.model_name, self.device)
            try:
                self._model = torch.hub.load(
                    "facebookresearch/dinov2",
                    self.model_name,
                    trust_repo=True,
                )
            except Exception as exc:
                logger.warning("Torch hub load encountered %s; retrying with local cache...", exc)
                self._model = torch.hub.load(
                    "facebookresearch/dinov2",
                    self.model_name,
                    source="local",
                )
            self._model.to(self.device)
            self._model.eval()
            logger.info("DINOv2 model '%s' loaded successfully.", self.model_name)
        return self._model

    def _prepare_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """Converts diverse image formats into a standard RGB PIL Image."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Reference image not found at path: {path}")
            image = Image.open(path).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                image_input = np.clip(image_input * 255.0, 0, 255).astype(np.uint8)
            if image_input.ndim == 2:
                image = Image.fromarray(image_input, mode="L").convert("RGB")
            elif image_input.shape[2] == 4:
                image = Image.fromarray(image_input, mode="RGBA").convert("RGB")
            else:
                image = Image.fromarray(image_input, mode="RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")
        return image

    def extract_features(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts L2-normalized global embedding and patch token embeddings.

        Returns:
            Tuple of:
                global_embedding: 1D np.ndarray float32 of shape (embedding_dim,)
                patch_embeddings: 2D np.ndarray float32 of shape (num_patches, embedding_dim)
        """
        pil_img = self._prepare_image(image_input)
        tensor = self._transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 1. Global CLS embedding
            global_feat = self.model(tensor)
            global_feat = global_feat / global_feat.norm(dim=-1, keepdim=True)
            global_np = global_feat.squeeze(0).cpu().numpy().astype(np.float32)

            # 2. Patch tokens
            intermediate = self.model.forward_features(tensor)
            if isinstance(intermediate, dict) and "x_norm_patchtokens" in intermediate:
                patch_tokens = intermediate["x_norm_patchtokens"]
            elif hasattr(intermediate, "x_norm_patchtokens"):
                patch_tokens = intermediate.x_norm_patchtokens
            else:
                patch_tokens = intermediate
            patch_np = patch_tokens.squeeze(0).cpu().numpy().astype(np.float32)

        return global_np, patch_np


class CatalogManager:
    """Manages zero-training SKU onboarding, BOM definitions, and vector embedding cache."""

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        storage_dir: Optional[Union[str, Path]] = None,
        model_name: str = "dinov2_vits14",
        device: Optional[str] = None,
    ):
        base_dir = Path(storage_dir or Path(__file__).resolve().parent.parent / "storage")
        self.storage_dir = Path(base_dir)
        self.catalog_db_path = Path(db_path or self.storage_dir / "catalog_db.json")
        self.embeddings_dir = self.storage_dir / "embeddings"
        self.images_dir = self.storage_dir / "catalog_images"

        # Ensure directory structure exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.device = device
        self._embedding_engine: Optional[DINOv2EmbeddingEngine] = None

        # In-memory caches for fast sub-millisecond retrieval
        self._profiles: Dict[str, SKUProfile] = {}
        self._global_embeddings: Dict[str, np.ndarray] = {}
        self._patch_embeddings: Dict[str, np.ndarray] = {}

        self._load_catalog()

    @property
    def embedding_engine(self) -> DINOv2EmbeddingEngine:
        """Returns the shared embedding engine instance."""
        if self._embedding_engine is None:
            self._embedding_engine = DINOv2EmbeddingEngine.get_shared_engine(
                model_name=self.model_name,
                device=self.device,
            )
        return self._embedding_engine

    def _load_catalog(self) -> None:
        """Loads SKU profiles from JSON database and embedding cache from disk."""
        if self.catalog_db_path.exists():
            try:
                with open(self.catalog_db_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                for sku_id, item in raw_data.items():
                    try:
                        profile = SKUProfile.model_validate(item)
                        self._profiles[sku_id] = profile
                    except Exception as err:
                        logger.error("Failed to parse SKU profile '%s': %s", sku_id, err)
            except Exception as err:
                logger.error("Error reading catalog DB '%s': %s", self.catalog_db_path, err)
        else:
            self._save_catalog()

        # Load embedding files into cache
        for emb_path in self.embeddings_dir.glob("*.npz"):
            sku_id = emb_path.stem
            try:
                data = np.load(emb_path)
                if "global_embedding" in data:
                    self._global_embeddings[sku_id] = data["global_embedding"]
                if "patch_embeddings" in data:
                    self._patch_embeddings[sku_id] = data["patch_embeddings"]
            except Exception as err:
                logger.warning("Failed loading cached embedding for '%s': %s", sku_id, err)

    def _save_catalog(self) -> None:
        """Persists registered SKU profiles to catalog_db.json."""
        serializable = {
            sku_id: profile.model_dump(mode="json")
            for sku_id, profile in self._profiles.items()
        }
        with open(self.catalog_db_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def onboard_sku(
        self,
        profile_data: Union[SKUProfile, SKUOnboardRequest, Dict[str, Any]],
        reference_images: Optional[List[Union[str, Path, Image.Image, np.ndarray]]] = None,
    ) -> SKUProfile:
        """Registers a new SKU or updates an existing SKU profile with DINOv2 embeddings.

        Args:
            profile_data: SKUProfile, SKUOnboardRequest, or dictionary with SKU metadata.
            reference_images: Optional list of 1-3 catalog reference photos (file paths or PIL Images).

        Returns:
            The fully registered and persisted SKUProfile.
        """
        if isinstance(profile_data, SKUProfile):
            profile = profile_data
        elif isinstance(profile_data, SKUOnboardRequest):
            profile = SKUProfile(**profile_data.model_dump())
        elif isinstance(profile_data, dict):
            profile = SKUProfile.model_validate(profile_data)
        else:
            raise TypeError(f"Invalid profile_data type: {type(profile_data)}")

        sku_id = profile.sku_id
        sku_img_dir = self.images_dir / sku_id
        sku_img_dir.mkdir(parents=True, exist_ok=True)

        saved_image_paths: List[str] = list(profile.reference_image_paths)

        # Process and store reference images if provided
        computed_globals: List[np.ndarray] = []
        computed_patches: List[np.ndarray] = []

        if reference_images:
            for idx, img_item in enumerate(reference_images):
                target_filename = f"ref_{idx + 1}.png"
                target_path = sku_img_dir / target_filename

                if isinstance(img_item, (str, Path)):
                    src_path = Path(img_item)
                    if src_path.resolve() != target_path.resolve():
                        img_obj = Image.open(src_path).convert("RGB")
                        img_obj.save(target_path, "PNG")
                    saved_path_str = str(target_path)
                elif isinstance(img_item, Image.Image):
                    img_item.convert("RGB").save(target_path, "PNG")
                    saved_path_str = str(target_path)
                elif isinstance(img_item, np.ndarray):
                    pil_img = self.embedding_engine._prepare_image(img_item)
                    pil_img.save(target_path, "PNG")
                    saved_path_str = str(target_path)
                else:
                    continue

                if saved_path_str not in saved_image_paths:
                    saved_image_paths.append(saved_path_str)

                # Extract DINOv2 feature representations
                global_vec, patch_vecs = self.embedding_engine.extract_features(target_path)
                computed_globals.append(global_vec)
                computed_patches.append(patch_vecs)

        elif saved_image_paths:
            for p_str in saved_image_paths:
                p = Path(p_str)
                if p.exists():
                    global_vec, patch_vecs = self.embedding_engine.extract_features(p)
                    computed_globals.append(global_vec)
                    computed_patches.append(patch_vecs)

        # If reference embeddings were extracted, aggregate and cache them
        if computed_globals:
            # Mean normalized global embedding
            stacked_globals = np.stack(computed_globals, axis=0)
            avg_global = np.mean(stacked_globals, axis=0)
            norm = np.linalg.norm(avg_global)
            if norm > 1e-8:
                avg_global = avg_global / norm

            # Average or representative patch tokens
            stacked_patches = np.stack(computed_patches, axis=0)
            avg_patches = np.mean(stacked_patches, axis=0)

            # Save to disk as compressed numpy archive
            emb_file = self.embeddings_dir / f"{sku_id}.npz"
            np.savez_compressed(
                emb_file,
                global_embedding=avg_global,
                patch_embeddings=avg_patches,
                sku_id=sku_id,
            )

            # Cache in-memory
            self._global_embeddings[sku_id] = avg_global
            self._patch_embeddings[sku_id] = avg_patches
            logger.info("Cached DINOv2 embedding vector for SKU '%s' (dim=%d)", sku_id, len(avg_global))

        # Update profile with reference paths
        profile.reference_image_paths = saved_image_paths
        self._profiles[sku_id] = profile
        self._save_catalog()

        logger.info("Successfully onboarded SKU '%s' (%s, MSRP: $%.2f)", sku_id, profile.title, profile.msrp)
        return profile

    def get_sku(self, sku_id: str) -> Optional[SKUProfile]:
        """Retrieves registered SKU profile by SKU ID."""
        return self._profiles.get(sku_id)

    def list_skus(self) -> List[SKUProfile]:
        """Lists all registered SKU profiles."""
        return list(self._profiles.values())

    def delete_sku(self, sku_id: str) -> bool:
        """Deletes SKU profile and cached embedding assets."""
        if sku_id not in self._profiles:
            return False

        del self._profiles[sku_id]
        self._global_embeddings.pop(sku_id, None)
        self._patch_embeddings.pop(sku_id, None)

        emb_file = self.embeddings_dir / f"{sku_id}.npz"
        if emb_file.exists():
            emb_file.unlink(missing_ok=True)

        self._save_catalog()
        return True

    def get_sku_embedding(self, sku_id: str) -> Optional[np.ndarray]:
        """Returns cached L2-normalized 1D DINOv2 global feature vector."""
        return self._global_embeddings.get(sku_id)

    def get_sku_patch_embeddings(self, sku_id: str) -> Optional[np.ndarray]:
        """Returns cached 2D DINOv2 patch tokens array."""
        return self._patch_embeddings.get(sku_id)

    def compute_similarity(
        self,
        sku_id: str,
        query_image: Union[str, Path, Image.Image, np.ndarray],
    ) -> float:
        """Computes cosine similarity between query image and cached SKU reference embedding.

        Returns:
            Cosine similarity score between -1.0 and 1.0 (typically 0.0 to 1.0).
        """
        ref_vec = self.get_sku_embedding(sku_id)
        if ref_vec is None:
            raise KeyError(f"No cached DINOv2 embedding found for SKU '{sku_id}'. Please onboard reference images first.")

        query_vec, _ = self.embedding_engine.extract_features(query_image)
        # Both vectors are L2-normalized, so dot product equals cosine similarity
        cosine_sim = float(np.dot(ref_vec, query_vec))
        return max(-1.0, min(1.0, cosine_sim))

    def search_catalog(
        self,
        query_image: Union[str, Path, Image.Image, np.ndarray],
        top_k: int = 5,
    ) -> List[Tuple[SKUProfile, float]]:
        """Identifies closest matching SKUs in the catalog using DINOv2 cosine similarity.

        Returns:
            List of tuples (SKUProfile, similarity_score) sorted in descending score order.
        """
        if not self._global_embeddings:
            return []

        query_vec, _ = self.embedding_engine.extract_features(query_image)

        scored: List[Tuple[SKUProfile, float]] = []
        for sku_id, ref_vec in self._global_embeddings.items():
            profile = self._profiles.get(sku_id)
            if profile is None:
                continue
            sim = float(np.dot(ref_vec, query_vec))
            scored.append((profile, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def verify_bom_checklist(
        self,
        sku_id: str,
        detected_item_ids: List[str],
    ) -> Tuple[List[BOMItem], List[BOMItem], float]:
        """Evaluates detected accessory IDs against SKU BOM requirements.

        Returns:
            Tuple of:
                present_items: List[BOMItem]
                missing_items: List[BOMItem]
                total_deduction_penalty: float ($)
        """
        profile = self.get_sku(sku_id)
        if not profile:
            raise KeyError(f"SKU '{sku_id}' not found in catalog.")

        detected_set = set(detected_item_ids)
        present: List[BOMItem] = []
        missing: List[BOMItem] = []
        total_deduction = 0.0

        for item in profile.bom:
            if item.item_id in detected_set:
                present.append(item)
            else:
                missing.append(item)
                total_deduction += item.deduction_fee

        return present, missing, total_deduction

