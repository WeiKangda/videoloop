# Modified from Qwen-MM-Plugins for VideoLoop; see README.md in this folder.
"""Load the built memory of one video."""

from __future__ import annotations

import logging
import os

from .embeddings import NPZ_INDEX_FILENAME, load_retrieval_index
from .schema import HierarchicalGraphMemory
from .toolkit import MemoryToolkit

GRAPH_FILENAME = "graph_memory.json"

logger = logging.getLogger(__name__)


def load_toolkit(memory_dir: str) -> MemoryToolkit:
    """A MemoryToolkit over <memory_dir>/graph_memory.json and its retrieval index."""
    graph_path = os.path.join(memory_dir, GRAPH_FILENAME)
    index_path = os.path.join(memory_dir, NPZ_INDEX_FILENAME)
    if not os.path.exists(graph_path) or not os.path.exists(index_path):
        raise RuntimeError(f"No complete memory in {memory_dir}: run scripts/build_video_memory.py")
    memory = HierarchicalGraphMemory.load(graph_path)
    toolkit = MemoryToolkit(memory, load_retrieval_index(index_path))
    logger.info(f"Video memory loaded: {len(memory.macro_events)} macros, {len(memory.super_events)} supers")
    return toolkit
