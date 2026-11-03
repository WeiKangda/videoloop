# Modified from Qwen-MM-Plugins for VideoLoop; see README.md in this folder.
"""The memory tools as the agent sees them, one for each of upstream's MCP tools.

The descriptions are upstream's, with no video_path argument, because a question's memory is fixed
when it starts, and with analyze_frames in place of read_video.
"""


def _tool(name: str, description: str, properties: dict = None, required: list = None) -> dict:
    parameters = {"type": "object", "properties": properties or {}}
    if required:
        parameters["required"] = required
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


VIDEO_MEMORY_TOOLS = [
    _tool(
        "get_summary",
        "Video-level root summary: title, description, themes, key entities, emotional tone. "
        "When to use: you need a quick overview before drilling down.",
    ),
    _tool(
        "get_super_events",
        "List all SuperEvents (high-level narrative arcs) with super_id, label, time_range, key_entities, "
        "description and num_macros. When to use: you need the overall structure or to pick the relevant story arc.",
    ),
    _tool(
        "get_macro_events",
        "Get Macro Event list (detailed events within a Super Event). Each macro event has macro_id, time_range, "
        "label, key_entities, summary. When to use: You know which SuperEvent is relevant but need to find the "
        "right MacroEvent within it. If super_id is omitted, lists ALL macro events.",
        {
            "super_id": {"type": "string", "description": "Filter by super event ID, e.g. 'super_01'. Optional — omit to list all."},
            "macro_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific macro IDs to retrieve, e.g. ['macro_0001', 'macro_0002'].",
            },
        },
    ),
    _tool(
        "get_subgraph",
        "Get detailed subgraph for a specific Macro Event. THIS IS THE MOST INFORMATION-RICH TOOL. Returns: "
        "Entities (name, type, attributes, visual features), Events (label, description, time), OCR texts "
        "(on-screen text), Key relations (CAUSAL/SPATIAL + key semantic links). When to use: Already identified "
        "the relevant macro_id, need full details. Always identify the macro_id first via search or navigation "
        "before calling this.",
        {"macro_id": {"type": "string", "description": "The macro_id to get the subgraph for (e.g. 'macro_0001')."}},
        ["macro_id"],
    ),
    _tool(
        "search_nodes",
        "Search over Entity and Event nodes (excludes OCRText — use search_ocr_text for on-screen text). Returns "
        "top-k results with relevance scores and ego-graph context. When to use: You need to find specific events, "
        "actions, or entities but don't know which macro to look at. Query Construction Guide: Write a DESCRIPTIVE "
        "STATEMENT about the target event, NOT a question. Extract key entities and actions from the question "
        "(person names, action types, objects). Extract SHARED information from all options — keywords/phrases that "
        "appear across multiple options are the most reliable retrieval signal. EXCLUDE speculative differing "
        "details from options — if options disagree on a specific detail (e.g., different numbers, different "
        "names), do NOT pick one; instead describe the event type neutrally. Good: 'A player scores with an "
        "alley-oop dunk' — descriptive, captures the shared event type. Bad: 'Which player scored the alley-oop?' — "
        "question format, poor match with event descriptions.",
        {
            "query": {"type": "string", "description": "Descriptive statement for search (NOT a question). Combine key entities, actions, and shared option info into a complete sentence."},
            "top_k": {"type": "integer", "description": "Number of results to return (default: 10)."},
            "node_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by node type (lowercase). Valid values: 'event', 'entity', 'on_screen_text'. e.g. ['event'], ['entity'].",
            },
        },
        ["query"],
    ),
    _tool(
        "search_ocr_text",
        "Search over OCR text nodes only (on-screen text: scores, names, graphics, jersey numbers, broadcast "
        "overlays). When to use: Question asks about on-screen information. Do NOT use search_nodes for these — "
        "OCR text is excluded from search_nodes.",
        {
            "query": {"type": "string", "description": "Search text, e.g. 'scoreboard', 'third quarter score', 'jersey number'."},
            "top_k": {"type": "integer", "description": "Number of results to return (default: 10)."},
        },
        ["query"],
    ),
    _tool(
        "search_asr_text",
        "Search over ASR transcript text (speech, dialogue, narration). When to use: Question asks about spoken "
        "content, dialogue, narration, or any audio/verbal information in the video. Returns matching transcript "
        "segments with timestamps and macro event context.",
        {
            "query": {"type": "string", "description": "Search text for spoken content, e.g. 'discussing weather', 'mentions recipe'."},
            "top_k": {"type": "integer", "description": "Number of results to return (default: 10)."},
        },
        ["query"],
    ),
    _tool(
        "search_by_time",
        "Find macro events covering a time range. Returns MacroEvent list with time ranges, parent SuperEvent "
        "labels, and key entities. When to use: Question references a specific time or time range in the video. "
        "Use numeric seconds, or start_time/end_time in 'HH:MM:SS' format. Invalid timestamps or reversed ranges "
        "return an error. Note: returns macro event summaries, not subgraph details. Always follow up with "
        "get_subgraph if you need entity/event details.",
        {
            "start_sec": {"type": "number", "description": "Non-negative start in seconds. Default: 0."},
            "end_sec": {"type": "number", "description": "Non-negative end in seconds, at or after the start. Default: 600."},
            "start_time": {"type": "string", "description": "Window start as 'HH:MM:SS'. Overrides start_sec."},
            "end_time": {"type": "string", "description": "Window end as 'HH:MM:SS'. Overrides end_sec."},
        },
    ),
    _tool(
        "enumerate_events",
        "Enumerate ALL matching event instances in time order — built for COUNTING and 'how many times / list every "
        "occurrence' questions. Unlike search_nodes (top-k best matches), this returns every matching node, "
        "deduplicated and sorted by start time, in a compact format. Protocol for counting: 1) Call this with a "
        "descriptive statement of the target event. 2) Review the time-ordered list; merge adjacent entries that "
        "describe the same real occurrence. 3) Verify borderline or ambiguous entries with analyze_frames at their "
        "time ranges. 4) If the expected count seems higher than returned matches, retry with alternative phrasings "
        "of the event. Never count from a single top-10 search — recurring events are spread across the whole video.",
        {
            "query": {"type": "string", "description": "Descriptive statement of the recurring event to enumerate (NOT a question)."},
            "min_cosine": {"type": "number", "description": "Minimum cosine similarity to keep a match (default 0.5), when the memory has vectors. Lower it if instances seem missing."},
            "max_results": {"type": "integer", "description": "Maximum matches to return (cap 300)."},
            "node_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Node types to enumerate (default ['event']). Use ['entity'] to enumerate objects/people.",
            },
        },
        ["query"],
    ),
]

VIDEO_MEMORY_TOOL_NAMES = [t["function"]["name"] for t in VIDEO_MEMORY_TOOLS]
