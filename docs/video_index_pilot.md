# 用 VideoLoop 跑 Video-Index v1.1

在 Video-Index v1.1 的 150 道题上跑 VideoLoop，模型用 Claude Opus 5.5，再和另一次运行按题配对比较。每题只跑一次，限时 3,600 秒，超时或出错都算错，不联网。

这个仓库不放题目、视频和密钥，也不放任何运行结果。`dataset/`、`results/`、`*.mp4` 都已被 git 忽略。Video-Index 只能用于研究评测，里面有商业影片，别往外传。

下面的命令都在仓库根目录运行，用到这些变量：

- `$ITEMS`：`items_test.jsonl` 的路径
- `$VIDEOS`：放着 `<sha256>.mp4` 的文件夹
- `$MODEL`、`$URL`：模型 ID 和接口地址
- `$WHISPER_ENV`：装了 faster-whisper 的 Python 环境
- `$MODELS`：faster-whisper 模型所在的文件夹
- `$TOOLS_ENV`：给 agent 的 shell 用的 Python 环境，可以和 `$WHISPER_ENV` 是同一个
- `$BASELINE_CSV`：要对比的那次运行的 sessions.csv

## 1. 准备环境

需要 Linux、Python 3.10 以上、bubblewrap、ffmpeg、ffprobe，用 `--cpu-slots` 时还要 `taskset`。

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

确认沙箱能启动，能打印出 ok 就行：

```bash
bwrap --ro-bind / / --dev /dev --proc /proc --unshare-net true && echo ok
```

试跑默认用 bubblewrap，不需要 Docker 服务，也不用构建镜像。沙箱只能看到系统目录、`--env-prefix` 指定的 Python 环境和当前这道题的视频，看不到题目文件。这个环境里有什么包（opencv、PyAV 等），agent 就能用什么。ffmpeg 和 ffprobe 也要在这个环境或系统目录里，否则脚本启动时就会报错。想用论文的 Docker 镜像，把 `configs/config_videoindex.yaml` 里的 `sandbox.runtime` 改成 `docker`，再运行 `docker/build.sh all`。

先跑一遍测试。这一步用假模型，真实走 bubblewrap 和 ffmpeg，大约 1 分钟，不花钱。装了第 11 步的依赖后还会测视频记忆，大约 5 分钟：

```bash
.venv/bin/python -m pytest tests -q
```

## 2. 准备数据

需要 `items_test.jsonl` 和放着 `<sha256>.mp4` 的视频文件夹，已有的直接用，路径用 `--items` 和 `--videos` 传进去。

## 3. 提前转写

VideoLoop 在每道题开始时读现成的转写，论文也是这样做的，所以运行时沙箱里不用 Whisper。这一步要在装了 faster-whisper 的 Python 环境里跑。

先只转写试跑要用的 3 道题：

```bash
$WHISPER_ENV/bin/python scripts/transcribe_videoindex.py --items $ITEMS --videos $VIDEOS --model-dir $MODELS/large-v3 --threads 12 --only TOC-Bench_57,VF-Eval_2358,K9-Bench_2956
```

全量 150 道题里，有音轨的视频一共 51.4 小时。可以用 `--shard` 拆开同时跑，比如 4 份：

```bash
for i in 0 1 2 3; do taskset -c $((i*12))-$((i*12+11)) $WHISPER_ENV/bin/python scripts/transcribe_videoindex.py --items $ITEMS --videos $VIDEOS --model-dir $MODELS/large-v3 --threads 12 --shard $i/4 & done; wait
```

没有音轨的视频直接写空转写，不加载模型。已经转写好的会跳过，所以中途可以停，重新运行就接着做。每个文件都记了耗时，打分时单独报告，不算进每题的用时。

## 4. 接模型

所有模型调用都经过 `video_agent/llm_api.py` 的 `openai_completion`，只需要接这一处。有两种接法，都由你来实现。

**接法一：OpenAI 兼容的 HTTP 接口**，比如你自己的网关。运行时传 `--api-base`，密钥放在环境变量 `MAIN_AGENT_API_KEY` 里，没有密钥就不设。

**接法二：一个 Python 函数**，不用起服务。写一个模块放在 `PYTHONPATH` 上，再设置 `VIDEOLOOP_LLM_BACKEND=my_backend:complete`：

```python
def complete(*, model, messages, max_tokens, temperature, timeout, tools, tool_choice, reasoning_effort, extra_body):
    # messages 和 tools 都是 OpenAI 格式
    # temperature 是 None 时不要发；tool_choice 是 None 或 "none"
    return {
        "content": "回复文字",
        "thinking_content": "",
        "tool_calls": [{"id": "...", "type": "function",
                        "function": {"name": "...", "arguments": "JSON 字符串"}}],
        "finish_reason": "stop",   # 截断时写 "length"
        "usage": {
            "promptTokenCount": 0,       # 所有输入 token，包括缓存的
            "candidatesTokenCount": 0,   # 所有输出 token，包括 thinking
            "thoughtsTokenCount": 0,
            "totalTokenCount": 0,
            "cacheReadTokenCount": 0,
            "cacheWriteTokenCount": 0,
        },
    }
```

函数抛异常时，外层会按 HTTP 那条路一样退避重试。异常文字里有 `invalid_request_error`、`input is too long` 这类字样就不重试，这道题直接记为出错。所以限流时直接抛异常就行。

接 Opus 5.5 时要注意这些：

- **effort 要送到模型。** 配置里主 agent 是 `max`，记忆模块是 `high`，也可以用 `--effort` 和 `--orchestrator-effort` 改。这个值从 `reasoning_effort` 传进来，真正的请求字段是 `output_config.effort`。接口把它丢掉也不会报错，模型会悄悄按默认的 `medium` 跑。第 5 步的检查脚本专门查这一条。走 HTTP 时如果网关能透传字段，也可以用 `llm.extra_body` 直接写。
- **thinking 关不掉。** 不要发 `thinking: {type: "disabled"}` 或 `budget_tokens`，这些都会返回 400。
- **不要回放 thinking 块。** VideoLoop 每一步都重写记忆，消息前缀一直在变，回放会触发前缀检查返回 400。返回给它的 `content` 只放文字。
- **`tool_choice` 要么不传，要么是 `"none"`。** 看完帧之后那次调用要文字回复，但历史里有工具调用，所以 tools 还要声明，再配 `{"type": "none"}`。强制调用工具的写法在 Opus 5.5 上会返回 400。
- **图片是 `data:image/...;base64,...` 形式的 `image_url` 块。** 要转成接口要求的格式。
- **`tool` 消息要转成 tool_result，放在工具调用之后那轮 user 消息的开头。** 后面可能紧跟一条带图片的 user 消息，要合并到同一轮里。
- **`max_tokens` 是 65536，请求可能很长。** 接口支持的话，后端内部用流式调用。超时时间在配置的 `main_agent.timeout` 和 `memory.summarizer_timeout`，都是秒。

`configs/config_videoindex.yaml` 的 `llm` 一节已经处理了请求形状：不发 temperature，不以 assistant 结尾，连续的 assistant 消息会合并，大图会缩到 2000 像素以内。

## 5. 检查接口

这一步只发几个很小的请求，几美分。每一项对应运行时依赖的一件事，这里不过，后面每道题都会出错。

```bash
.venv/bin/python scripts/check_model_endpoint.py --model $MODEL --api-base $URL
```

用接法二时去掉 `--api-base`，并设置 `VIDEOLOOP_LLM_BACKEND`。

8 项全过再往下走。`effort` 这一项是比较 `low` 和配置值的输出 token 数，不过的话可能是接口丢了 effort，也可能只是这道题太简单，自己判断一下。

## 6. 试跑 3 道题

这 3 道题分别是：有音轨的短视频、没有音轨的短视频、只有音频的文件。

```bash
.venv/bin/python scripts/run_videoindex.py --items $ITEMS --videos $VIDEOS --out results/videoindex/smoke --model $MODEL --api-base $URL --env-prefix $TOOLS_ENV --only TOC-Bench_57,VF-Eval_2358,K9-Bench_2956
```

```bash
.venv/bin/python scripts/score_videoindex.py --run results/videoindex/smoke --items $ITEMS
```

跑之前可以加 `--dry-run`，它只列出要跑的题和沙箱命令，不调模型。

跑完主要看这几处：

- `results/videoindex/smoke/*/result.json`：`status` 应该是 `succeeded`，`answer_source` 应该是 `submit`。
- `usage` 里的 `failed_calls` 应该是 0，`orchestrator_calls_failed` 也应该是 0。
- `child.log`：有没有反复重试或报错。
- 打分输出的费用：按这 3 道题估算全量的花费和时间，再决定下一步。费用分主 agent 和记忆模块两部分。记忆模块太贵的话，可以用 `--orchestrator-model` 换一个便宜的模型。

再试一道长视频，这种题最费钱，也最容易超时。先转写，再跑：

```bash
$WHISPER_ENV/bin/python scripts/transcribe_videoindex.py --items $ITEMS --videos $VIDEOS --model-dir $MODELS/large-v3 --threads 12 --only ALLVB_10848
```

```bash
.venv/bin/python scripts/run_videoindex.py --items $ITEMS --videos $VIDEOS --out results/videoindex/smoke --model $MODEL --api-base $URL --env-prefix $TOOLS_ENV --only ALLVB_10848
```

## 7. 全量 150 道题

`--cpu-slots` 给每个同时运行的题分一组 CPU，用分号隔开，几组就同时跑几题。转写要先全部做完，缺转写时脚本会直接拒绝启动。

```bash
.venv/bin/python scripts/run_videoindex.py --items $ITEMS --videos $VIDEOS --out results/videoindex/run1 --model $MODEL --api-base $URL --env-prefix $TOOLS_ENV --cpu-slots '0-5,48-53;6-11,54-59;12-17,60-65'
```

- **中途停了：** 用同一条命令重新运行，已有 `result.json` 的题不会再跑。换了模型、effort 或配置，脚本会拒绝接着跑，请换一个 `--out`。
- **连续 5 题出错：** 脚本不再开始新题，比如凭证过期时，后面的题不会全被记成错。修好后加 `--retry-errors` 重新运行，它只重跑 `error` 和 `crashed` 的题，请求太大的题不重跑，这类题就算错。只有服务故障时才用这个参数。
- **每题的目录：** `result.json`、`trajectory.jsonl`（每步记录，已去掉 base64 图片）、`usage.jsonl`（每次模型调用）、`child.log`。运行中抽出的帧在每题结束时就删除了。

## 8. 打分和对比

```bash
.venv/bin/python scripts/score_videoindex.py --run results/videoindex/run1 --items $ITEMS --transcripts dataset/videoindex_v1_1/transcripts_whisperx --baseline $BASELINE_CSV
```

结果会打印出来，同时写到 `results/videoindex/run1/summary.md` 和 `sessions.csv`。`sessions.csv` 用的列名和基线的 sessions.csv 一样，可以作为下一次比较的 `--baseline`。

计分规则：

- 只有 agent 自己提交的答案才算：正常的 `submit_answer`、预算用完被要求后提交的、或者多轮不调工具后明确写出的“the answer is C”。
- VideoLoop 自带的兜底不算答案，也就是从对话文字里用正则抓的字母和默认的 A，都算没答。
- 超时、出错、不在选项里的字母也算没答。没答算错，并在结果里按原因分别列出。
- 准确率用 bootstrap 给 95% 区间，并做精确 McNemar 检验。费用、用时只在两边都答完的题上比较。
- 费用按 `--prices`，默认是 Opus 5.5 的公开价：输入 4、缓存写入 5、缓存读取 0.2、输出 20 美元每百万 token。

## 9. 读结果时要记住

- **缓存：** VideoLoop 不设置 `cache_control`，每步又重写记忆，所以默认所有输入都按全价算。它的费用和大量使用缓存的 harness 比，会显得偏高。这是当前的实际设计，也是后面可以改进的地方。
- **转写：** 这里是提前做好的，不算进每题用时。如果对比的那次运行把转写算在每题用时里，比用时的时候要记得这一点。
- **工具环境：** 沙箱用的是 `--env-prefix` 的环境，不是论文镜像，通常没有 scenedetect、tesseract 这些工具。论文的分数不能直接当成这里的预期。
- **模型：** 论文用 Gemini，这里换成 Opus 5.5，提示词没有为它调过。
- **没验证过的：** 仓库里的测试只用了假模型。真实的 Opus 5.5、真实的接口和你的机器都没跑过，所以第 5、6 步不能省。

## 10. 对上游的改动

基于 [philipxjm/videoloop](https://github.com/philipxjm/videoloop) 的 Apache-2.0 代码修改。

新增的文件：

- `configs/config_videoindex.yaml`
- `scripts/run_videoindex.py`、`score_videoindex.py`、`transcribe_videoindex.py`、`check_model_endpoint.py`
- `video_agent/bwrap_runtime.py`
- `tests/test_videoindex_pilot.py`、`tests/fake_model_server.py`
- 这份文档
- 第 11 步用到的 `scripts/build_video_memory.py`、`video_agent/video_memory/`、`tests/test_video_memory.py`

只在 `config_videoindex.yaml` 这样打开时才生效的改动：

- `llm` 一节：Python 后端、不发 temperature、Messages API 兼容的请求形状、`extra_body`
- `main_agent` 的 `reasoning_effort`、`max_image_long_side`、`max_batch_mb`
- `sandbox` 的 `runtime`、`network_disabled`、`mount_video`、`cpuset_cpus`、`container_name` 等
- `video_memory` 一节和 `--video-memory`：记忆工具、提示词里的 `<VIDEO_MEMORY>`，以及告诉记忆模块这些结果要核实的那句话

在默认配置下也生效的改动，都是修问题：

- `sandbox` 里的 `gpu`、`memory_limit`、`shm_size` 现在真的会生效。原来写了也不起作用，默认值相同。
- 模型名含 `claude` 时，工具改用 OpenAI 格式。原来的格式在调用时会被直接丢掉，模型收不到任何工具。
- 提示词和兜底逻辑里的选项字母跟着选项数走。4 个选项时提示词和原来一样，只有预算用完时那段催促的措辞略有变化。
- 预算用完而没有答案时，一定先要求提交，再用兜底。原来有些路径会跳过这一步。
- 从文字里找 “select X” 这类答案时，字母区分大小写。原来 “select a frame” 会被读成答案 A。
- 没有视频流的文件不再要求先看帧，没有音轨的文件不再要求先转写。原来这些要求满足不了，没有视频流的题会一直被打回。
- 记忆模块的工具循环用完预算时不再崩溃。原来这一步的记忆更新会丢失。
- `invalid_request_error`、`input is too long` 这类错误不再重试。
- 没有密钥时不发 `Authorization` 头。原来空密钥会让请求一直失败。
- tiktoken 改成第一次用到时才加载。原来每次 import 都可能联网下载，只有无记忆模式才用到。

## 11. 视频记忆（可选）

给长视频提前建一份分层记忆，agent 答题时用 9 个工具查它，定位到时间段后再抽帧核实。代码来自 [Qwen-MM-Plugins](https://github.com/QwenLM/Qwen-MM-Plugins) 的 video-memory（Apache-2.0），放在 `video_agent/video_memory/`，改了什么见那里的 [README](../video_agent/video_memory/README.md)。

不传 `--video-memory` 时，这部分不起任何作用，前面的步骤和结果都不变。所有模型调用走第 4 步接好的同一个接口，不用再接一次。

先把第 5、6 步跑通，再做这一步，否则出错时分不清问题在哪一层。

**准备：**

```bash
.venv/bin/pip install -e ".[video-memory]"
```

建库脚本直接调用 ffmpeg 和 ffprobe，它们要在 `PATH` 上。如果只装在 `$TOOLS_ENV` 里，先 `export PATH=$TOOLS_ENV/bin:$PATH`。

**先试 3 个视频。** 这是 30 分钟以上里最短的 3 个，一共约 2 小时。转写要先做好：

```bash
$WHISPER_ENV/bin/python scripts/transcribe_videoindex.py --items $ITEMS --videos $VIDEOS --model-dir $MODELS/large-v3 --threads 12 --only ViMUL-Bench_4920,Video-MME_2411,Video-MME_2180
```

```bash
.venv/bin/python scripts/build_video_memory.py --items $ITEMS --videos $VIDEOS --model $MODEL --api-base $URL --only ViMUL-Bench_4920,Video-MME_2411,Video-MME_2180
```

每个视频的结果在 `dataset/videoindex_v1_1/video_memory/<视频文件名>/`。先看这些：

- `build_stats.json`：`status` 应该是 `done`，`n_empty_subgraphs` 应该接近 0。`usage` 是每类调用的 token，`wall_s` 是用时，`phase_s` 是每个阶段的用时。
- `graph_memory.json`：挑一段的 `subgraph` 看看，事件描述对不对，时间对不对得上视频。这一步不能省，假模型测不出建出来的东西好不好。

超过 10% 的段建不出来时，脚本会停在聚合之前并报错。已经建好的段都在磁盘上，同一条命令再跑一次就接着做，不重复花钱。换了模型、帧率这些设置，脚本会拒绝接着做，请换一个 `--out`。

**用记忆答题：**

```bash
.venv/bin/python scripts/run_videoindex.py --items $ITEMS --videos $VIDEOS --out results/videoindex/memory_smoke --model $MODEL --api-base $URL --env-prefix $TOOLS_ENV --video-memory dataset/videoindex_v1_1/video_memory --only ViMUL-Bench_4920,Video-MME_2411,Video-MME_2180
```

```bash
.venv/bin/python scripts/score_videoindex.py --run results/videoindex/memory_smoke --items $ITEMS --video-memory dataset/videoindex_v1_1/video_memory --baseline $BASELINE_CSV
```

`result.json` 里 `video_memory` 应该是 `true`，`tool_calls` 里应该有 `search_nodes`、`get_subgraph` 这些。如果都是 0，说明 agent 没用记忆，要先看提示词。

**全部 29 个 30 分钟以上的视频，一共 49.6 小时。** 不带 `--only` 时默认只建 30 分钟以上的。可以用 `--shard` 拆开同时跑：

```bash
for i in 0 1 2 3; do .venv/bin/python scripts/build_video_memory.py --items $ITEMS --videos $VIDEOS --model $MODEL --api-base $URL --shard $i/4 > build_$i.log 2>&1 & done; wait
```

```bash
.venv/bin/python scripts/run_videoindex.py --items $ITEMS --videos $VIDEOS --out results/videoindex/memory_long --model $MODEL --api-base $URL --env-prefix $TOOLS_ENV --video-memory dataset/videoindex_v1_1/video_memory --min-duration 1800 --cpu-slots '0-5,48-53;6-11,54-59;12-17,60-65'
```

带 `--video-memory` 时，30 分钟以上的视频必须有建好的记忆，否则脚本拒绝启动。打分时 `--baseline` 只比两边都有的题，所以这里就是这 29 道题的配对比较。

**费用估算，没有实测过。** 这 29 个视频大约切成 1,100 段。每段约 80 帧，宽 720 像素的 16:9 帧约 390 token，所以输入约 3,500 万 token，按公开价约 150 美元。输出取决于模型写多少，每段 3 千到 1 万 token 的话是 65 到 220 美元。合计大约 200 到 370 美元。先看 3 个视频的 `usage` 再估全量。想省钱可以调低 `--fps` 和 `--max-dim`，也可以用 `--effort` 改 effort，默认 `medium`。用时上游的说法是不用 OSS 时每小时视频约 40 分钟，那是另一个模型，只能参考。

**读结果时要记住：**

- **建库费用要单独看。** `cost_usd`、`wall_s` 只算答题。建库在 `build_cost_usd`、`build_wall_s`，两者相加在 `cost_with_build_usd`、`wall_with_build_s`。这里每个视频只有一道题，建库费用全落在这一道题上，没有摊薄。LVBench 这种每个视频多道题的数据集不一样。
- **限时不含建库。** 3,600 秒只算答题。
- **默认只有 BM25。** 也就是按词匹配，换一种说法就可能搜不到。上游默认是向量加 BM25。要接向量，写一个函数放在 `PYTHONPATH` 上，建库和答题时都设置 `VIDEOLOOP_EMBED_BACKEND=my_embed:embed`。这部分没有用真实的向量模型测过。

  ```python
  def embed(texts):
      # texts 是 list[str]，按同样的顺序返回 list[list[float]]
      ...
  ```

- **段的标题和摘要是空的。** 上游就是这样：段是按帧差切的，没有哪一步给它们写标题。所以 `get_macro_events` 和 `search_by_time` 除了时间和主要实体，信息很少，agent 主要靠搜索和 `get_subgraph`。这是一个明显可以改进的地方。
- **建库只看帧。** 上游可以上传视频片段，Opus 5.5 不收视频，所以这里每段最多 100 帧，看不到连续动作，也听不到非语音的声音。千问公布的带记忆分数不是这样建的，不能当成这里的预期。
- **转写在记忆里也有一份。** 提示词让 agent 优先用 `search_asr_text`，只在不够时才调 `transcribe_audio`，因为后者会把整份转写放进之后的每一轮。其他提示词没有改，所以它可能还是会调用。
- **答案不会进记忆。** 建库脚本只读题目条目里视频的位置和长度，不读问题、选项和答案。agent 的沙箱看不到记忆文件，工具在沙箱外的 agent 进程里读。
- **没验证过的：** 这部分和前面一样只用假模型测过。真实模型建出来的质量、费用和用时，以及它到底能不能提高准确率，都还不知道。
