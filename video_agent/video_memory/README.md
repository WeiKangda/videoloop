# Video memory, from Qwen-MM-Plugins

This folder comes from the `video-memory` capability of [QwenLM/Qwen-MM-Plugins](https://github.com/QwenLM/Qwen-MM-Plugins), at commit `acad0c92c33b3c76f4904a179279590e805548b7`, under the Apache License 2.0. A copy of that license is in [LICENSE](LICENSE). How to run it is in [docs/video_index_pilot.md](../../docs/video_index_pilot.md).

## Files

| File | From upstream | Changed? |
|---|---|---|
| `schema.py`, `time_system.py` | `qwen_mm_plugins_video_memory/` | No |
| `prompts.py`, `time_utils.py` | `skill/script/build_memory/` | No |
| `toolkit.py` | `qwen_mm_plugins_video_memory/` | One note names `analyze_frames` instead of `read_video` |
| `build_graph.py` | `skill/script/build_memory/` | See below |
| `llm_client.py` | `skill/script/build_memory/` | Calls go through VideoLoop's `openai_completion` instead of DashScope |
| `embeddings.py` | `qwen_mm_plugins_video_memory/` | See below |
| `loader.py` | `qwen_mm_plugins_video_memory/` | Only `load_toolkit`, with no MCP cache and no time-system detection |
| `tool_schemas.py` | `qwen_mm_plugins_video_memory/tools/` | The tools as OpenAI-style schemas, with no `video_path` |

The workflow in the `video_memory_preamble` of `configs/prompts.yaml` is shortened from upstream's `skill/SKILL.md`.

## What changed in the build

- **Model route.** Every model call goes through the route VideoLoop already uses, so wiring one model wires both. Upstream calls DashScope directly.
- **Frames only.** The route to Claude takes no video, so the branch that uploads clips to OSS is gone. Upstream's branch for base64 frames is the one kept.
- **At most 100 frames per segment**, where upstream allows 250, because Claude takes at most 100 images in a request. A request is also kept under `main_agent.max_batch_mb`.
- **Frame labels count from the start of the segment.** Upstream labels frames with the time in the whole video while its prompt asks for times relative to the segment.
- **Transcripts come from `scripts/transcribe_videoindex.py`.** Upstream calls cloud ASR.
- **Each finished subgraph is saved at once**, so a stopped build doesn't pay for it again. Upstream does this only in its worker process.
- **A build with more than 10% of its segments empty stops** before aggregation. Upstream carries on with a warning.
- **The file name alone goes in `graph_memory.json`**, not the full path of the video.

## What changed in retrieval

- **Dense vectors are optional.** Upstream embeds with DashScope's `qwen3-vl-embedding`. Here the embedding function is whatever `video_memory.embed_backend` or `VIDEOLOOP_EMBED_BACKEND` names, as `module:function`.
- **With no backend, search is BM25 alone**, which upstream already falls back to when embedding fails. This is the default, and it matches words, not meaning.
- **NPZ index only.** The zvec backend is gone.

## Left as upstream

`label` of a segment stays `scene_0012` and its `summary` stays empty: nothing in upstream's pipeline fills them, since segments come from frame differences. So `get_macro_events` and `search_by_time` show little beyond time ranges and key entities, and the aggregation step sees no summaries. This is kept here so that the starting point is upstream's behavior.
