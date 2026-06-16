# Vast.aiでDiffusionGemma INT8を検証する手順

この手順は、Vast.ai上で `aidendle94/diffusiongemma-26B-A4B-it-INT8-dynamic` をvLLMのOpenAI互換APIとして起動し、Local Fusion Gatewayから疎通確認するためのrunbookです。

このモデルは `google/diffusiongemma-26B-A4B-it` のINT8量子化派生です。元のbf16 checkpointそのものではありません。2x RTX 3090での低コスト検証を第一候補にし、Gateway smokeまでを目的にします。

## 前提

- Vast.ai CLIが利用できること。
- Vast.ai API keyが設定済みであること。
- ここではHugging Face tokenは必須にしない。
- インスタンス作成前に必ず最新offerを提示し、利用者が明示的に選んだoffer IDだけを借りる。
- 検証後はインスタンスをdestroyし、課金を止める。

## 1. Offer検索

2x RTX 3090を第一候補として検索する。

```bash
vastai search offers \
  'rentable=true verified=true gpu_name=RTX_3090 num_gpus=2 cuda_vers>=12.1 direct_port_count>=1 reliability>=0.99' \
  --limit 8 \
  --storage 250 \
  -o 'dph'
```

提示時に見る項目:

- `ID`
- `Model`
- `$/hr`
- `R`
- `Disk`
- `ports`
- `country`

候補が高すぎる、信頼性が低い、またはportが不足している場合は、次の順に再検索する。

```bash
vastai search offers \
  'rentable=true verified=true gpu_name=RTX_3090 num_gpus=4 cuda_vers>=12.1 direct_port_count>=1 reliability>=0.99' \
  --limit 8 \
  --storage 250 \
  -o 'dph'
```

```bash
vastai search offers \
  'rentable=true verified=true gpu_name=RTX_4090 num_gpus=2 cuda_vers>=12.1 direct_port_count>=1 reliability>=0.99' \
  --limit 8 \
  --storage 250 \
  -o 'dph'
```

```bash
vastai search offers \
  'rentable=true verified=true gpu_ram>=80 num_gpus=1 cuda_vers>=12.1 direct_port_count>=1 reliability>=0.99' \
  --limit 8 \
  --storage 250 \
  -o 'dph'
```

## 2. Instance作成

利用者が選んだoffer IDを `<OFFER_ID>` に入れる。ここで初めて課金が発生する。

```bash
vastai create instance <OFFER_ID> \
  --image vllm/vllm-openai:gemma \
  --disk 250 \
  --ssh \
  --direct \
  --env '-p 8000:8000 -e VLLM_USE_V2_MODEL_RUNNER=1' \
  --onstart-cmd 'mkdir -p /workspace && nohup vllm serve aidendle94/diffusiongemma-26B-A4B-it-INT8-dynamic --served-model-name diffusiongemma-int8-vastai --trust-remote-code --tensor-parallel-size 2 --max-num-seqs 4 --gpu-memory-utilization 0.75 --hf-overrides '"'"'{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'"'"' --diffusion-config '"'"'{"canvas_length":256}'"'"' --override-generation-config '"'"'{"max_new_tokens":8192}'"'"' --enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4 --default-chat-template-kwargs '"'"'{"enable_thinking":true}'"'"' --host 0.0.0.0 --port 8000 > /workspace/vllm.log 2>&1 &'
```

作成後、返ってきた `new_contract` を `<INSTANCE_ID>` として控える。

```bash
vastai show instance <INSTANCE_ID>
vastai logs <INSTANCE_ID> --tail 200
```

## 3. vLLMの疎通確認

Vast.aiのinstance detailから、container port `8000` に対応するpublic host/portを確認する。以降はそれを `<VLLM_BASE_URL>` とする。

例:

```text
http://<PUBLIC_HOST>:<PUBLIC_PORT>/v1
```

まず `/v1/models` を確認する。

```bash
curl <VLLM_BASE_URL>/models
```

短いchat completionを確認する。

```bash
curl <VLLM_BASE_URL>/chat/completions \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer EMPTY' \
  -d '{
    "model": "diffusiongemma-int8-vastai",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

失敗時はまずlogを見る。

```bash
vastai logs <INSTANCE_ID> --tail 300
```

短時間で解決できない場合は、課金を止めるためdestroyしてから原因を整理する。

## 4. Gateway smoke

一時configをrepo外に作る。

```bash
cat >/tmp/local-fusion-vastai.yaml <<'YAML'
server:
  host: "127.0.0.1"
  port: 8080
  request_timeout_seconds: 300

fusion:
  default_analysis_models:
    - "diffusiongemma-int8-vastai"
  default_judge_model: "diffusiongemma-int8-vastai"

models:
  diffusiongemma-int8-vastai:
    base_url: "<VLLM_BASE_URL>"
    api_key: "EMPTY"
    model: "diffusiongemma-int8-vastai"
YAML
```

Gatewayを起動する。

```bash
LOCAL_FUSION_CONFIG=/tmp/local-fusion-vastai.yaml uv run local-fusion-gateway
```

別terminalで通常proxyを確認する。

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "diffusiongemma-int8-vastai",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Fusion smokeを確認する。

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'X-Local-Fusion-Debug: true' \
  -d '{
    "model": "openrouter/fusion",
    "messages": [{"role": "user", "content": "Compare JSON and YAML in one paragraph."}],
    "max_tokens": 160,
    "temperature": 0
  }'
```

確認項目:

- HTTP 200が返る。
- 通常proxyで本文が返る。
- Fusionで本文が返る。
- `local_fusion` metadataが返る。
- `local_fusion` にprompt本文やpanel回答本文が入らない。

## 5. Cleanup

検証が終わったらdestroyする。

```bash
vastai destroy instance <INSTANCE_ID>
```

destroy後に表示されなくなったことを確認する。

```bash
vastai show instances
```

## 補足

- 2x RTX 3090で失敗した場合は、同じinstanceで長く試行錯誤せず、destroyしてから次の候補を選び直す。
- 量子化派生モデルなので、品質・速度の結果は元のbf16モデルと同一ではない。
- この手順ではGateway smokeを目的にする。性能比較を行う場合は、別途benchmark datasetとlatency記録を追加する。
