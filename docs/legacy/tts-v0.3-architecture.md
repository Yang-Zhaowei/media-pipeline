# 0.3 服务端/工作机架构

状态：本轮未运行测试或部署。旧架构保存为 ARCHITECTURE_LOCAL.md，部署步骤以 README 为准。

## 边界

工作机默认安装HTTPX、NumPy、SoundFile、SciPy；服务器再安装[server,qwen]。服务通过EngineClient在独立Python子进程中加载Qwen，模型导入不占用HTTP事件循环。

| 代码 | 职责 |
|---|---|
| pipeline.py、audio.py、parser.py | 编排、断点恢复、音频、字幕 |
| config.py、contracts.py | 配置路由和普通数据契约 |
| remote.py、service_cli.py | HTTP客户端、任务凭据、查询、校验下载 |
| server/app.py、settings.py | 认证、API输入、服务器配置和声音目录 |
| server/store.py | SQLite队列、幂等键、终态回收 |
| server/runner.py | 串行任务消费者及Qwen子进程生命周期 |
| backends/qwen/ | 模型API、参考音、Qwen prompt |

transport=http选择RemoteEngineClient；subprocess或缺省保留原本地EngineClient。仅服务器目前实现qwen3_tts。远程HTTP协议与后端model/provider区分，未来服务器模型替换不要求客户端安装模型包。

请求不接受ref_audio、output_path、Python路径、任意生成参数或pickle/tensor。声音源和模型参数预先注册，客户端只发文本/角色/voice ID/语言/语气/指令。模型私有对象留在服务器。

## HTTP API v1

所有路由要求Bearer token。当前为单用户/可信小组共享密钥服务，没有多租户账号体系。默认监听回环地址，经SSH隧道或HTTPS访问。

| 方法 | 路径 | 返回/作用 |
|---|---|---|
| GET | /healthz | 任务循环存活和backend_state，不代表模型已完成加载 |
| GET | /v1/info | API版本、部署名/指纹、能力、限额、后端状态 |
| GET | /v1/voices | 声音ID、模式、语气、prepare/指令能力 |
| POST | /v1/validate | 目录/长度/能力检查，不加载Qwen |
| POST | /v1/jobs | 接受synthesize/prepare，返回202与ID |
| GET | /v1/jobs/{id} | queued/running/succeeded/failed/expired |
| GET | /v1/jobs/{id}/audio | WAV及SHA256响应头 |

每次提交需要Idempotency-Key（16–128位安全ASCII），deployment_revision来自info。合成请求例：

~~~json
{
  "api_version": 1,
  "deployment_revision": "服务器返回的完整指纹",
  "operation": "synthesize",
  "request": {
    "text": "这是一段台词。",
    "role": "旁白",
    "voice": "narrator",
    "language": "Chinese",
    "tone": "",
    "instruct": null
  }
}
~~~

prepare使用operation=prepare和voice=<id>，不带request。它预热声音特征/创建固定设计参考，返回JSON收据，不远程下载.pt。

401认证失败；404未知或已清除任务；409部署/幂等键冲突或音频未就绪；410音频过期；413请求太大；422字段/声音/能力错误；429队列或保留音频预算满；503任务循环不可用。

目录验证不是模型加载、全部语言/素材解码的验收。实际模型/转写校验发生在任务线程中，失败通过任务和服务器日志报告。本API不声明兼容OpenAI或任意厂商；不同API需要客户端适配。

## 标识与恢复

- plan_hash：工作机输入、配置、请求、输出路径、工作流版本和源码哈希。
- execution_id：本次生成批次；--force 会更换，防止续跑误用上一批的收据或任务。
- deployment_revision：服务器配置、注册素材内容哈希、源码、管理员设置的deployment_id。环境和本地权重变化要求更新deployment_id/model_asset_version。
- compatibility_fingerprint：Qwen包/源码、模型版本、dtype/device等，约束派生prompt复用。

服务器每次执行核对已注册声音源SHA256，运行中修改源文件会拒绝任务。固定设计参考单独持久保存并备份，不随模型包更新重新设计；管理员不能运行中覆盖它们。

设计参考的配方仍包含设计模型标识；改模型目录/配方会创建新源。需要长期固定身份时，应将验收后的设计WAV和文本显式登记为voice_sources并切换voice_clone。固定音源不等于不同模型版本必然保持相同听感，仍需人工回归。

SQLite事务中处理幂等：相同key/payload返回原任务，不同payload返回409。客户端在POST之前保存ticket，丢失提交响应也能重用key找回已接受任务。下载校验长度/SHA256并通过临时文件替换。原始浮点段与结果收据保存在_work。

--resume复用校验通过的原始段；缺段时利用ticket恢复任务。明确failed/expired时新key重试一次；普通断线不换key。损坏的原始段停止恢复，避免误判完成。服务重启将queued/running标为failed，不自动重放。客户端退出/超时不取消服务器任务。

服务器会过期清理终态WAV，再清理记录；幂等保证受保留期限制。TTL边界上的下载可能失败；已下载原始段仍能本地使用，缺失结果可能需要新任务。部署变化拒绝混合续跑。

## 效率和限制

单GPU消费者串行推理，复用常驻模型和特征；HTTP连接池复用连接，传二进制WAV，避免音频base64 JSON。工作机按段处理，用SciPy resample_poly替代librosa；重采样与旧算法可能有边界差异，需试听回归。成片分块复制PCM16分段和静音，不保留整篇数组。

多种Qwen模型一旦使用会保留在模型池中，开启Base/VoiceDesign/CustomVoice组合须评估显存。当前无实时音频流、自动长句切分、vLLM批推理、多GPU或多租户隔离。

同data_dir用OS文件锁限定一个服务进程；同segments_dir用运行锁防止并发写。模型操作超时由runtime.timeout_seconds控制，HTTP/轮询超时由客户端配置控制。

max_storage_bytes只约束已成功音频，不包括生成中临时文件、模型缓存、数据库和可能的崩溃残留；它不是操作系统磁盘配额。日志按job ID定位错误。只有通过新测试计划，才可标记稳定部署版本。
