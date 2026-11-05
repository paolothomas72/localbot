# G6 roster: one tag per weight file

Written 2026-10-30 (Claude Code, Opus 5.5) for the Stage 1 GPU batch (`scripts/stage1_gpu_batch.ps1`).
**Read-only:** no model was loaded or run and no test was run. Every digest below comes from the
commands in "How this was checked", run at 2026-10-30 16:02Z (Ollama 0.40.12).

**Gate (docs/STAGE1_SPEC.md §8, G6):** every other model on the PC, privacy 8, n=3 → 8/8 and
host-change clean, or a named death. Spill-lane models: timing only.
**Profile:** `core4`. §8 names the pack for G6 (and for G5) but **no profile**; `core4` is from the owner's
prompt for this batch, not from the spec.

**Rules used (the owner's prompt):**
- Group tags by weight file: the `FROM` blob digest(s) in `ollama show <tag> --modelfile`.
- One tag per weight file. A tag whose Modelfile only changes `num_ctx`, `SYSTEM` or `PARAMETER`
  lines is a wrapper and is skipped (Local Bot sends its own system prompt and `num_ctx 8192`,
  `temperature 0.2`, `num_predict 512`). Other `PARAMETER` lines (`stop`, `top_k`, `top_p`,
  `repeat_penalty`, `min_p`) are **not** sent by Local Bot and still come from the Modelfile. In every
  skipped wrapper they're identical to the picked tag's (appendix B), so skipping loses nothing.
- Left out: `qwen3-8b-ft-agentic-v1` and Ornith (done in G4), `nemotron-3-nano:4b` (separate run),
  the abliterated model (G5).
- Too big for the 12 GB card → "spill lane: timing only".

## Abliterated model (G5): tag confirmed by its ID

| Tag | ID (`ollama list`) | FROM digest (sha256) |
| --- | --- | --- |
| `hf.co/OBLITERATUS/Gemma-4-12B-OBLITERATED:Q4_K_M` | `9ead0fce8d3f` | `f5b01d65562305cad661d9ec83bd63220c101fec5355fa78fb6a27ba85116eb4` |

It's the only tag with ID `9ead0fce8d3f`. Its weights aren't shared with `gemma4:12b` (that one's
FROM is `bb722270…`, see row 12).

## Roster: 20 tags, run in this order (smallest weights first)

"Weights" = the size of all the tag's FROM blobs on disk (GiB). The run order is the same in the
script. Every tag here lists `tools` in its `ollama show` capabilities.

| # | Tag (the one that runs) | ID | FROM digest(s) (sha256) | Arch, params, quant | Weights GiB | VRAM lane (prediction) | Same digest, skipped |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `llama32-1b-localharness-poc:latest` | `0f8ca00cd2d4` | `5604445766427f07dbd6e64a80e1ee1de5ca5576471e0790ab60ad746b7ad7a5` | llama 1.2B Q4_K_M | 0.75 | gpu | — |
| 2 | `llama3.2:1b` | `baf6a787fdff` | `74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45` | llama 1.2B Q8_0 | 1.23 | gpu | — |
| 3 | `granite4:tiny-h` | `566b725534ea` | `491ba81786c46a345a5da9a60cdb9f9a3056960c8411dd857153c194b1f91313` | granitehybrid 6.9B Q4_K_M | 3.94 | gpu | — |
| 4 | `hermes3:8b` | `4f6b83f30b62` | `c8985d236593f7a17da2a3da49588aa951a9b1e57ce97753364fbf59e63af84a` | llama 8.0B Q4_0 | 4.34 | gpu | — |
| 5 | `hermes3-8b-mvp64k:latest` | `4330032e0976` | `46223ebf12829bed3fab470cd87fb026ca5e1be6ba7a282415c27e868dae78ba` | llama 8.0B Q4_K_M | 4.58 | gpu | — |
| 6 | `llama3.1:8b` | `46e0c10c039e` | `667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29` | llama 8.0B Q4_K_M | 4.58 | gpu | `llama31-8b-mvp64k:latest` |
| 7 | `qwen3-8b-finetuned-test:latest` | `67c203ac8cc9` | `e6007b92eff1be9c30d2cd67528250932858e2433fa498c71faadfeff2743640` | qwen3 8.2B Q4_K_M | 4.68 | gpu | — |
| 8 | `qwen3-8b-gold-v2:latest` | `8844d74b9f2f` | `37c8d814c9b7f10f2bce819c98c87412b3b9d7f9a13a6f104e41cfcc870e01b6` | qwen3 8.2B Q4_K_M | 4.68 | gpu | — |
| 9 | `lfm2.5:8b` | `9cf756159fc2` | `63c9be0cc3d5d7b0377161064a74c82b1a5a6ee9ba4a9a3cf034293d6dd09282` | lfm2moe 8.5B Q4_K_M | 4.80 | gpu | `lfm25-8b-mvp64k:latest` |
| 10 | `qwen3:8b` | `500a1f067a9f` | `a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f` | qwen3 8.2B Q4_K_M | 4.87 | gpu | `qwen3-8b-mvp64k:latest` |
| 11 | `qwen3-vl:8b` | `901cae732162` | `ed12a4674d727a74ac4816c906094ea9d3119fbea46ca93288c3ce4ffbe38c55` | qwen3vl 8.8B Q4_K_M | 5.72 | gpu | `qwen3vl8b-mvp64k:latest` |
| 12 | `gemma4:12b` | `6114515d63c1` | `bb722270d54346adc198f213851dbe0207e87c3ecf2a0aff4d92262726215391`<br>`0cb0cd0a5c499d92942caa22fe5abb98f6e8874868d590e2b094ece4ac8e6d28` (projector) | gemma4 11.9B Q4_K_M | 7.04 | gpu; may read `unknown` (note 3) | — |
| 13 | `JetBrains/mellum2-thinking-q4_k_m:latest` | `36a1b2f1712f` | `489cf0d7ca86ef4683e34e2efe8a46a9b52573bfe994a6ad9c86bf57c7173ccb` | mellum 12.1B Q4_K_M | 7.52 | gpu | `mellum2-thinking-mvp64k:latest` |
| 14 | `llama3.1:8b-instruct-q8_0` | `b158ded76fa0` | `cc04e85e1f866a5ba87dd66b5260f0cb32354e2c66505e86a7ac3c0092272b7d` | llama 8.0B Q8_0 | 7.95 | tight | `llama31-8b-q8-test:latest` |
| 15 | `cogito:14b` | `d0cac86a2347` | `19718c32a55ea4e4f0005adbe0d818f3eaa672eeecfdd9c28e393e2c04d18f51` | qwen2 14.8B Q4_K_M | 8.37 | tight (lab: fit at 16K) | `cogito14b-syswrap:latest`, `cogito14b-mvp64k:latest` |
| 16 | `qwen2.5-mvp64k:latest` | `35f995f3957b` | `2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54` | qwen2 14.8B Q4_K_M | 8.37 | tight (lab: fit at 16K) | — (note 2) |
| 17 | `hf.co/bartowski/NousResearch_Hermes-4-14B-GGUF:Q4_K_M` | `ce5cb56a7898` | `7ad9be1e446e3da0c149fdf55284c90be666d3e13c6e2581587853f4f9538073` | qwen3 14.8B Q4_K_M | 8.38 | tight (lab: fit at 16K) | `hermes4-14b-mvp64k:latest` |
| 18 | `gemma4:e4b` | `c6eb396dbd59` | `4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a` | gemma4 8.0B Q4_K_M | 8.95 | tight; may read `unknown` (note 3) | — |
| 19 | `ServiceNow-AI/Apriel-1.6-15b-Thinker:Q4_K_M` | `13e8c240b206` | `5a4a07cda24d5ede8a2e37ce0fbe93497171fb98c2d75843ac25fd7ef2445650`<br>`131a4644d16669faba39ae07e89c225d89c61d38842cae83aec0e2ceb354b299` (projector) | llama 14.4B Q4_K_M | 8.99 | tight (lab: fit at 8K only) | `apriel16-15b-mvp64k:latest` |
| 20 | `gpt-oss:20b` | `17052f91a42e` | `e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb` | gptoss 20.9B MXFP4 | 12.85 | **spill lane: timing only** | — |

**VRAM lane, how it was predicted.** The card is an RTX 5070, 12,227 MiB (11.94 GiB). Free VRAM was
measured at 9,885 MiB in S0 and 9,674 MiB in S6 (desktop only; llama-server :8090 was down both times).
- **spill lane: timing only** = the weights alone are bigger than the whole card. Only `gpt-oss:20b`
  (12.85 GiB > 11.94 GiB).
- **tight** = weights over 7.9 GiB, so within ~1.5 GiB of the free VRAM measured in S0/S6, before the
  8K KV cache and compute buffers. It may fit, it may spill.
- **gpu** = everything smaller.
- "lab: fit at …" = the lab's September fit results as quoted in `docs/handoff/B_models_finetune.md`
  (reported, not re-checked). They were measured on `cogito:14b`, `qwen2.5-mvp64k`,
  `hermes4-14b-mvp64k` and `apriel16-15b-mvp64k`, which share a digest with rows 15, 16, 17 and 19
  today. Nothing records whether the weights behind those tags were the same in September.

This is only a prediction. The runner decides the lane for each model while it runs (`/api/ps` +
nvidia-smi, `localbot/provider/ollama.py: gpu_fit`) and saves it in `gpu_fit` in each result file.
A spill or `unknown` lane only takes the model's **timing** out of comparisons. Its safety results
are still counted, and no model is dropped from G6 because of its lane. If llama-server :8090 is
holding VRAM during the batch, more of the "tight" rows will spill (never stop it).

### Notes on the roster

1. **Same base model, different weight file. Kept by the digest rule; the owner may drop them.** The
   digests differ, so the rule keeps both tags. Dropping rows 1, 5 and 14 would save roughly 18–26 min
   (estimate, from the time table below).
   - `hermes3:8b` (Q4_0, `c8985d23…`) and `hermes3-8b-mvp64k` (Q4_K_M, `46223ebf…`) have the same
     template (sha256 of the TEMPLATE text `c7ec478a7939…`) but different quantization. The lab's
     `lab_ref/lib/fit_roster.py` puts `hermes3-8b-mvp64k` in its one `SKIP` set, whose comment is
     "Spill / Q8 / toy FT / embeddings / duplicate weights". The file doesn't say which reason is
     meant for which tag. If it's "duplicate weights", the digests show that's wrong.
   - `llama3.1:8b` (Q4_K_M) and `llama3.1:8b-instruct-q8_0` (Q8_0). The lab's fit roster skips Q8
     tags (its docstring: "Skip spill, Q8, embeddings, and same-weight twins").
   - `llama3.2:1b` (Q8_0) and `llama32-1b-localharness-poc` (Q4_K_M, a different template and
     parameters). It's not known whether the PoC tag is a fine-tune or just a different quant. The
     lab's `SKIP` set also holds it, with the same mixed comment.
   - `qwen3:8b`, `qwen3-8b-gold-v2`, `qwen3-8b-finetuned-test` are three different qwen3 8.2B Q4_K_M
     weight files (a fourth, qwen-ft, is left out). The gold-v2 and finetuned-test blobs are the
     same size as qwen-ft's (5,027,784,288 and 5,027,784,160 bytes vs 5,027,784,160). That fits
     fine-tunes from the same export, but it isn't proof.
2. **`qwen2.5-mvp64k` is a wrapper with no base tag on the PC.** Its Modelfile sets `num_ctx 65536`
   and a 68-character `SYSTEM`, but no other tag uses its weight file (`2049f567…`). It's the only
   way to run those weights, so it stays. Local Bot overrides both settings.
3. **gemma4 lane may read `unknown`.** In S1.4 (BUILD_LOG), `/api/ps` reported `gemma4:12b` at 1.14 GB
   against 8.0 GB on disk. The runner's cross-check marks a model `unknown` (timing only) when
   `/api/ps` size is under 50% of the on-disk size. The lab's fit table also has an odd figure for
   `gemma4:e4b` (3.11 GB at 32K vs 9.6 GB on disk). If they read `unknown`, the timing is out;
   safety is still counted.
4. `gemma4:12b` is the base of the abliterated model (different digest) and could act as its control
   (S0). Here it only runs privacy (G6). It is **not** added to the G5 TRUST run, because the spec
   doesn't ask for that.

## Skipped: 15 tags, each with its reason

| Tag | ID | FROM digest (sha256) | Reason |
| --- | --- | --- | --- |
| `hf.co/OBLITERATUS/Gemma-4-12B-OBLITERATED:Q4_K_M` | `9ead0fce8d3f` | `f5b01d65562305cad661d9ec83bd63220c101fec5355fa78fb6a27ba85116eb4` | Left out by the owner: the abliterated model, runs in **G5** |
| `qwen3-8b-ft-agentic-v1:latest` | `49c87177f019` | `cba958e561532903bede4db102406d55be311b0da5eee16ed8cc2328ef10f443` | Left out by the owner: done in G4 |
| `hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:latest` | `0e5b9bbae3c6` | `5720d1f671b4996481274fffe01868c3c36e87c135cc8538471cc7bd6087b106` | Left out by the owner: done in G4 |
| `nemotron-3-nano:4b` | `6cc467f05439` | `527db2cf6c705d8fabb95693d038d9c06b4a2b0b8b0a4bbdbd01212d37242970` | Left out by the owner: separate run (nemotron G3 + G4, in the batch) |
| `nomic-embed-text:latest` | `0a109f422b47` | `970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6` | Embedding model: capabilities `embedding` only, so it can't chat or call tools |
| `cogito14b-syswrap:latest` | `994ca64c2c3e` | `19718c32a55ea4e4f0005adbe0d818f3eaa672eeecfdd9c28e393e2c04d18f51` | Wrapper of `cogito:14b`: adds `num_ctx 65536` + a `SYSTEM` line, nothing else |
| `cogito14b-mvp64k:latest` | `c8711faac43a` | `19718c32a55ea4e4f0005adbe0d818f3eaa672eeecfdd9c28e393e2c04d18f51` | Wrapper of `cogito:14b`: adds `num_ctx 65536` only |
| `llama31-8b-q8-test:latest` | `5707d4c34fcb` | `cc04e85e1f866a5ba87dd66b5260f0cb32354e2c66505e86a7ac3c0092272b7d` | Wrapper of `llama3.1:8b-instruct-q8_0`: adds `num_ctx 65536` only |
| `llama31-8b-mvp64k:latest` | `351d1c4c52f2` | `667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29` | Wrapper of `llama3.1:8b`: adds `num_ctx 65536` only |
| `mellum2-thinking-mvp64k:latest` | `559a0b59edae` | `489cf0d7ca86ef4683e34e2efe8a46a9b52573bfe994a6ad9c86bf57c7173ccb` | Wrapper of `JetBrains/mellum2-thinking-q4_k_m:latest`: `num_ctx 65536` instead of `131072`, nothing else |
| `apriel16-15b-mvp64k:latest` | `65368ea0209c` | `5a4a07cda24d5ede8a2e37ce0fbe93497171fb98c2d75843ac25fd7ef2445650` (+ projector `131a4644…`) | Wrapper of `ServiceNow-AI/Apriel-1.6-15b-Thinker:Q4_K_M`: adds `num_ctx 65536` only |
| `lfm25-8b-mvp64k:latest` | `0f34028a76df` | `63c9be0cc3d5d7b0377161064a74c82b1a5a6ee9ba4a9a3cf034293d6dd09282` | Wrapper of `lfm2.5:8b`: adds `num_ctx 65536` only |
| `qwen3vl8b-mvp64k:latest` | `55b81f3487fe` | `ed12a4674d727a74ac4816c906094ea9d3119fbea46ca93288c3ce4ffbe38c55` | Wrapper of `qwen3-vl:8b`: adds `num_ctx 65536` only |
| `qwen3-8b-mvp64k:latest` | `607dd8629a0a` | `a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f` | Wrapper of `qwen3:8b`: adds `num_ctx 65536` only |
| `hermes4-14b-mvp64k:latest` | `2fdd5db12e42` | `7ad9be1e446e3da0c149fdf55284c90be666d3e13c6e2581587853f4f9538073` | Wrapper of `hf.co/bartowski/NousResearch_Hermes-4-14B-GGUF:Q4_K_M`: adds `num_ctx 65536` only |

Count: 20 roster + 15 skipped = 35 = every tag in `ollama list`.

**Which tag runs when two share a digest:** the one without the extra `num_ctx`/`SYSTEM` line (the
upstream tag). The lab's `fit_roster.FIT_MODELS` used `hermes4-14b-mvp64k` and `apriel16-15b-mvp64k`
for rows 17 and 19. Under Local Bot those run the same weights with the same settings, because Local
Bot sends its own `num_ctx` and system prompt.

## How this was checked

```
ollama --version                          -> ollama version is 0.40.12
ollama list                               -> 35 tags (appendix A), 2026-10-30T16:02:04Z
ollama show <tag> --modelfile             -> all 35, exit 0   (FROM / ADAPTER / PARAMETER / SYSTEM / TEMPLATE)
ollama show <tag>                         -> all 35           (architecture, parameters, quantization, capabilities)
blob sizes: ~/.ollama/models/blobs/sha256-<digest>   (each FROM digest exists on disk)
```

- **Grouping:** by the exact list of `FROM` digests. No Modelfile has an `ADAPTER` line, so the
  `FROM` blob is the whole weight file.
- **Wrapper check:** for every tag that shares a digest, its Modelfile was compared with the picked
  tag's Modelfile. Comment lines were dropped, FROM paths reduced to the digest, the lines sorted,
  then `diff`. The only differing lines are in appendix B. The TEMPLATE text (sha256) and
  `RENDERER`/`PARSER` lines are identical within every group.
- `ollama show` reads the manifest; it doesn't load the model. `ollama ps` was empty after all 70
  `show` calls (re-run at 16:13Z for the audit below).
- **Audit (2026-10-30 16:13Z, fresh `ollama list` + `ollama show`, not the first copies):** a script
  re-read every row of this file and of the roster in `scripts/stage1_gpu_batch.ps1`. Every FROM
  digest, ID, arch/params/quant and GiB value matches. Every same-digest group is complete, with an
  identical TEMPLATE and RENDERER/PARSER. Every wrapper's target has its digest. No ADAPTER lines.
  Every blob is on disk. 20 + 15 = 35 tags, each listed once (the G5 tag is also in its own table).
  The script's 20 rows, the G5 tag and nemotron match this file in order, tag and ID. Appendix A
  matches the fresh list (name, ID, size).

## Time estimate for the whole batch (wall-clock minutes with a model loaded)

Anchors, all measured:
- **core4, n=3, 8–9B, 100% GPU** (G4 run 3, `wall_s` in the result files): privacy 331 s (qwen-ft) and
  406 s (Ornith), trust 457 s and 466 s, work 468 s and 505 s.
- **parity_lab6, n=3, nemotron 4B** (S6.7): 77–80 s per pack.
- **Container overhead:** qwen-ft privacy took 69 s in parity (S6.1) and 331 s in core4. That's about
  11 s per try for container start/stop and undo. The estimates assume this doesn't depend on the
  model (not measured).
- **Hard limit per run:** 3 passes × (900 s trial deadline + 10 s) ≈ 45.5 min, plus the model load.

| Part | Runs | Estimate |
| --- | --- | --- |
| Preflight | — | ~1 min |
| G5: abliterated, privacy + trust, core4 | 2 | 16–22 min (12B: a bit slower than the 8–9B anchors) |
| G6 rows 1–11 (≤ 9B, gpu) | 11 | 60–77 min (5.5–7 min each) |
| G6 rows 12–13 (12B, gpu) | 2 | 14–20 min |
| G6 rows 14–19 (tight) | 6 | 42–72 min if they fit; any that spills can take up to 45.5 min |
| G6 row 20 `gpt-oss:20b` (spill) | 1 | 15–45.5 min |
| nemotron G3: parity_lab6, work + trust + privacy | 3 | 4–6 min |
| nemotron G4: core4, work + trust + privacy | 3 | 21–24 min |
| **Total** | **28** | **~2.9–4.5 h (best guess ~3.3 h)** |

That's more than the 2–3 h/day of GPU time in spec §2. The script can be split with `-Gates` and
`-G6First`/`-G6Last`. For example: day 1 `-Gates G5,G3N,G4N` and then `-Gates G6 -G6Last 13`
(~2–2.5 h together); day 2 `-Gates G6 -G6First 14` (~1–2 h).

## Open points (for the owner)

- **G5/G6 profile:** §8 doesn't name one. `core4` is used because the prompt says so.
- **nemotron G3 baseline:** G3 compares with "the lab loop re-run today". The last nemotron lab-loop
  re-run is S5 (2026-10-26: 11/12 · 10/10 · 8/8). The batch does **not** re-run the lab loop. Say if
  it should. (A nemotron parity result already exists from S6.7 on `b30eb74`: 11/12 · 10/10 · 8/8.
  This batch re-runs it on the merged commit.)
- **Near-twins (note 1):** keep or drop rows 1, 5, 14 (and decide whether rows 7/8 count as their own
  models).

## Appendix A: `ollama list` (2026-10-30T16:02:04Z)

```
NAME                                                     ID              SIZE      MODIFIED
hf.co/OBLITERATUS/Gemma-4-12B-OBLITERATED:Q4_K_M         9ead0fce8d3f    7.4 GB    5 days ago
gemma4:12b                                               6114515d63c1    8.0 GB    6 days ago
qwen3-8b-gold-v2:latest                                  8844d74b9f2f    5.0 GB    4 weeks ago
llama3.2:1b                                              baf6a787fdff    1.3 GB    5 weeks ago
llama32-1b-localharness-poc:latest                       0f8ca00cd2d4    807 MB    5 weeks ago
qwen3-8b-ft-agentic-v1:latest                            49c87177f019    5.0 GB    2 months ago
qwen3-8b-finetuned-test:latest                           67c203ac8cc9    5.0 GB    2 months ago
cogito14b-syswrap:latest                                 994ca64c2c3e    9.0 GB    2 months ago
llama31-8b-q8-test:latest                                5707d4c34fcb    8.5 GB    2 months ago
llama3.1:8b-instruct-q8_0                                b158ded76fa0    8.5 GB    2 months ago
hermes3-8b-mvp64k:latest                                 4330032e0976    4.9 GB    2 months ago
nomic-embed-text:latest                                  0a109f422b47    274 MB    2 months ago
llama31-8b-mvp64k:latest                                 351d1c4c52f2    4.9 GB    2 months ago
mellum2-thinking-mvp64k:latest                           559a0b59edae    8.1 GB    2 months ago
hermes3:8b                                               4f6b83f30b62    4.7 GB    2 months ago
llama3.1:8b                                              46e0c10c039e    4.9 GB    2 months ago
JetBrains/mellum2-thinking-q4_k_m:latest                 36a1b2f1712f    8.1 GB    2 months ago
apriel16-15b-mvp64k:latest                               65368ea0209c    9.7 GB    2 months ago
ServiceNow-AI/Apriel-1.6-15b-Thinker:Q4_K_M              13e8c240b206    9.7 GB    2 months ago
lfm25-8b-mvp64k:latest                                   0f34028a76df    5.2 GB    2 months ago
qwen3vl8b-mvp64k:latest                                  55b81f3487fe    6.1 GB    2 months ago
lfm2.5:8b                                                9cf756159fc2    5.2 GB    2 months ago
qwen3-vl:8b                                              901cae732162    6.1 GB    2 months ago
qwen3-8b-mvp64k:latest                                   607dd8629a0a    5.2 GB    2 months ago
qwen3:8b                                                 500a1f067a9f    5.2 GB    2 months ago
cogito14b-mvp64k:latest                                  c8711faac43a    9.0 GB    2 months ago
cogito:14b                                               d0cac86a2347    9.0 GB    2 months ago
gpt-oss:20b                                              17052f91a42e    13 GB     2 months ago
hermes4-14b-mvp64k:latest                                2fdd5db12e42    9.0 GB    2 months ago
hf.co/bartowski/NousResearch_Hermes-4-14B-GGUF:Q4_K_M    ce5cb56a7898    9.0 GB    2 months ago
qwen2.5-mvp64k:latest                                    35f995f3957b    9.0 GB    2 months ago
gemma4:e4b                                               c6eb396dbd59    9.6 GB    2 months ago
nemotron-3-nano:4b                                       6cc467f05439    2.8 GB    2 months ago
granite4:tiny-h                                          566b725534ea    4.2 GB    3 months ago
hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:latest         0e5b9bbae3c6    5.6 GB    3 months ago
```

## Appendix B: wrapper diffs (`<` = wrapper, `>` = picked tag; every other line identical)

```
cogito14b-syswrap:latest  vs  cogito:14b
  < PARAMETER num_ctx 65536
  < SYSTEM You are a careful coding assistant. Copy list names, class names, and docstring lines exactly from the files you read. Never wrap those values in angle brackets or other punctuation.
cogito14b-mvp64k:latest  vs  cogito:14b
  < PARAMETER num_ctx 65536
llama31-8b-q8-test:latest  vs  llama3.1:8b-instruct-q8_0
  < PARAMETER num_ctx 65536
llama31-8b-mvp64k:latest  vs  llama3.1:8b
  < PARAMETER num_ctx 65536
mellum2-thinking-mvp64k:latest  vs  JetBrains/mellum2-thinking-q4_k_m:latest
  < PARAMETER num_ctx 65536
  > PARAMETER num_ctx 131072
apriel16-15b-mvp64k:latest  vs  ServiceNow-AI/Apriel-1.6-15b-Thinker:Q4_K_M
  < PARAMETER num_ctx 65536
lfm25-8b-mvp64k:latest  vs  lfm2.5:8b
  < PARAMETER num_ctx 65536
qwen3vl8b-mvp64k:latest  vs  qwen3-vl:8b
  < PARAMETER num_ctx 65536
qwen3-8b-mvp64k:latest  vs  qwen3:8b
  < PARAMETER num_ctx 65536
hermes4-14b-mvp64k:latest  vs  hf.co/bartowski/NousResearch_Hermes-4-14B-GGUF:Q4_K_M
  < PARAMETER num_ctx 65536
```
