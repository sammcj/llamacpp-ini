# Querying decision models

Decision models (GLiDE-27B, `vllm-d3-flash`, other "System One" models) score the options you give them and return a probability for each. They don't generate text. `/v1/chat/completions` returns `This model is a decision model without text generation`.

`sync-models.py` finds them by the `<arch>.decision.type` GGUF key and tags them `decision` in the router.

## Endpoint

```
POST http://localhost:8080/v1/systemone
```

`model` is the router name, e.g. `Glide-27B-Q4_K_M`. The `-no-mmproj` twin works for text-only requests.

## Example

```bash
curl -s localhost:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "Glide-27B-Q4_K_M",
  "state": {"ticket": "I was charged twice for order #4821 and the second charge is still pending.", "customer_tier": "gold"},
  "questions": {
    "intent": {"type": "choice", "instructions": "What does the customer want?",
               "criteria": {"refund": "Get money back", "cancel": "Cancel the order", "track": "Track a delivery", "other": "Something else"}},
    "urgent": {"type": "noul", "instructions": "Should this ticket be escalated today?"},
    "priority": {"type": "score", "instructions": "What support priority should this ticket get?",
                 "criteria": ["low: no impact on the customer", "medium: the customer is inconvenienced", "high: the customer has lost money or cannot use the service"]}
  }
}' | jq .answers
```

Response (GLiDE-27B Q4_K_M, rounded):

```json
{
  "intent":   {"type": "choice", "choice": "refund", "confidence": 0.88,
               "probabilities": {"refund": 0.964, "cancel": 0.002, "track": 0.001, "other": 0.033}},
  "urgent":   {"type": "noul", "noul": 0.831},
  "priority": {"type": "score", "score": 1.948, "confidence": 0.81,
               "probabilities": {"0": 0.003, "1": 0.047, "2": 0.951},
               "legend": {"0": "low: ...", "1": "medium: ...", "2": "high: ..."}}
}
```

`usage.input_tokens` is the prompt length summed over all questions. `output_tokens` is always 0.

## Request

- `state`: the context to decide on. A string, any JSON value, or a list of parts (see images).
- `questions`: `{id: question}`. Each question is scored independently against the full state.
- `images` (alias `files`): optional array of image data URLs, as an alternative to image parts in `state`.

Question fields:

- `type`: `choice`, `noul` or `score`
- `instructions`: the question text
- `criteria`: depends on `type`:
  - `choice`: required object `{option_key: description}`. GLiDE accepts up to 255 options.
  - `noul`: optional `{"true": "...", "false": "..."}`. Defaults to yes/no.
  - `score`: required array of 2 to 10 level descriptions. Level _i_ is index _i_.

## Response fields

- `choice`: `choice` is the most probable key. `probabilities` covers every option and sums to 1.
- `noul`: `noul` is P(true). llama.cpp doesn't return `confidence` for this type.
- `score`: `score` is the **expected level** (sum of index x probability), so it is a float. Take the highest entry in `probabilities` if you need the most likely level. `legend` echoes the level descriptions.
- `confidence` (choice, score): higher means the top option leads the rest by more. Set thresholds on your own data.

llama.cpp's `score` and `noul` fields differ from the model card's Python server. That server returns an integer `score` plus `expected_level`, and includes `confidence` for `noul`.

## Images

Use a vision model, not the `-no-mmproj` twin. Images must be inline data URLs (no remote URLs), with at most 8 per request.

```json
"state": [
  "Inspection photo from line 3.",
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
],
"questions": {"damaged": {"type": "noul", "instructions": "Is there visible damage on the part?"}}
```

GLiDE allows up to 1,024 tokens per image. At the default 512 `ubatch-size`, llama-server caps that at 256. The 16384 set by `sync-models.py` removes the cap.

## Limits and cost

- **Prompt length:** each question's prompt (state + question + options + image tokens) must fit in one `ubatch-size`, set to 16384 by `sync-models.py`. Longer prompts fail with `input (N tokens) is too large to process`. Raise `DECISION_UBATCH` in `sync-models.py` if you need more, at the cost of a bigger compute buffer.
- **Cost per question:** each question is a separate forward pass over the full state, so ten questions over a 10K-token state process about 100K tokens. Ask fewer, sharper questions.
- **Prompt cache:** off for decision models (`cache-ram = 0`), so repeated states aren't cached.
- **Determinism:** no sampling. The same request gives the same probabilities. `temp` and other sampler settings have no effect.
- **One slot:** decision models run with `parallel = 1`. Questions and concurrent requests queue. In testing, 1 slot was about 25% faster than 4 or 8 for a 4-question request.
- **Measured:** GLiDE-27B Q4_K_M on the M5 Max takes about 4.4 s per question on a 2.5K-token state, warm. A 4-question request takes 17.5 to 19.5 s.
