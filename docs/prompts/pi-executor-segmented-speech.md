# pi executor — 实施 prompt

你是这个 PR 的 executor。负责完成实现、必要测试与证据，不扩展产品范围。
不要只返回方案；在授权范围内推进到可审查的 PR。不要 merge。

仓库：`Yang-Zhaowei/media-pipeline`。
唯一功能契约：`docs/contracts/segmented-speech-v0.md` 的 C1–C9，完整阅读全文并遵守。
这份文件与本 prompt 必须一起提供；若文件不可用，先取得文件，不依据摘要自行补契约。
用户的新明确指令优先；其余情况下不可自行弱化契约或扩大范围。

先从仓库恢复状态：检查工作区、当前分支、最新 main 和已有进行中的 PR；读取
AGENTS.md、README.md、docs/CURRENT.md、docs/ROADMAP.md、docs/dev/core-runtime.md，
四个现有生产组件及其测试、PR #5 的验证驱动。规划时基线为 795c4b6，不能假定现在
仍是最新提交。不要覆盖已有用户修改，不要把 PR #5 的 GPU 证据当作本次验证。

## 已确定的用户意图

- Agent 负责分段、修稿和选择声音。仓库只校验，超长拒绝，不自动断句。
- 提供一个同步 Python `render_speech` 入口，从带分段 JSON 生成完整 WAV/SRT。
- TTS 仅加载一次并连续生成所有段；完成后退出，再加载一次 Aligner 连续处理。
  不能逐段起停模型，不能依赖两模型同时驻留显存。
- 各段局部对齐互不依赖。只对明确的段结果校验错误记录并继续；模型/CUDA/子进程/
  I/O/未知故障不能吞掉后继续。
- 不需要中途人工放行。任何段失败都不发布完整节目；保留各段结果和一次运行报告。
- 不将“对齐失败”解释成“音频读错”。无自动修复、重试、断点续跑、缓存或成功总段导出。
- 所有全局偏移由 cleaned WAV 整数帧累计；字幕最后才舍入到毫秒。
- 不加 HTML、ASR、NLE、公开 CLI、MCP、HTTP、通用编排或环境整合。

## 实施方式

1. 建立 C1–C9 对应的最小改动计划后开始实现。模块/私有类命名可自行决定，不为
   未来功能新增 provider、registry、抽象任务引擎等设计。
2. 复用四个现有生产组件，保持 CPU 纯处理与 GPU 运行时隔离。生产代码不得导入
   tests/experiments；验证脚本的经验可以参考，不能变成运行依赖。
3. 内部子进程协议必须能逐段返回成功或已知校验错误，保留阶段、段 ID、原因；
   不可每段启动一个模型进程。未知错误按 C5 停止。
4. 先实现并测试预检、时间轴、拼接和报告等确定性逻辑，再接入实际两个运行环境。
   以 C8 为测试清单；验证真实错误路径，避免只测自己的 mock 返回了预期值。
5. 输出文件按 C6 发布，最终 WAV/SRT/timeline 回读一致；既有输出目录与 fixtures
   不可覆盖。保留失败证据，不增加 resume。
6. 运行相关测试和完整 CPU 回归。一次通过后不无理由反复重跑。记录实际命令、
   passed/skipped/failed；不得将 skip、fake 或 mock 称作真实 GPU 验证。
7. 若可访问已有 ai-core，在实际实现提交上执行 C8 的真实多段验收，不做环境整合。
   若访问/环境条件缺失，明确记录阻塞、准备准确运行说明和需回传的证据，不擅自
   安装替代 GPU 环境或声称验收完成。人类试听必须由用户明确确认。
8. 更新 README/CURRENT 和必要的使用/验证文档，说明新入口、错误语义和确切支持
   范围。真实 GPU/人工验收未完成时必须标为 pending。
9. 在功能分支提交并创建可审查 PR；验收未齐时保持 draft。附真实测试结果、限制、
   逐条 C1–C9 验收映射。不要自动 merge，不发送消息给其他 chat，除非用户另行授权。

## Implement it narrowly.

Requirements:

- reuse existing repository patterns;
- do not redesign unless repository evidence proves the plan invalid;
- do not broaden scope;
- keep portable logic separate from runtime/model integration;
- verify uncertain external API behavior instead of guessing;
- add the required tests;
- run the relevant test suite;
- report exactly what was executed;
- distinguish CPU/unit validation from real runtime/GPU validation;
- do not claim integration validation unless it actually ran.

If implementation must deviate materially from the plan, stop and explain
the repository evidence that requires the deviation.
遇到契约的实质冲突，先完成不受影响的工作并指出冲突，不能静默变更范围。
特别禁止为了让测试通过而放宽失败语义、删除原稿内容、伪造字幕或修改回归 fixtures。
