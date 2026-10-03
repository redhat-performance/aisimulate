# GLM-5.3-Flash config provenance

- Upstream repository: https://huggingface.co/zai-org/GLM-5.3-Flash
- Immutable revision: `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`
- Original path: [`config.json`](https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json)
- License verified from the exact revision's [`LICENSE`](https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/LICENSE): MIT, copyright (c) 2026 Z.AI Co., Ltd.
- Derived file: `zai-org--GLM-5.3-Flash_config.json` (modified/reduced and reformatted).

This is an offline **geometry and quantization-default fixture**, not a runnable
Transformers checkpoint config. Repeated per-module `modules_to_not_convert`
entries are stored as lossless layer/suffix groups; the loader expands them to
the standard list before consumption. Retained values are unchanged. FP8 block128 describes the
checkpoint default, not every tensor: upstream excludes KDA projections, mHC,
norms, selected sparse-attention projections, and vision components. Execution
modeling must account for those exclusions separately; do not assume uniform FP8.
When loading a complete local/HF config, the loader retains the complete root
quantization object, including exclusions, in `raw_config`.

The 45 text layers comprise 34 KDA layers and 11 sparse MLA layers. KDA and full
attention layer IDs are **zero-based**, unlike Kimi-K3. The first three FFNs are
dense; the remaining 42 route top-8 of 288 experts plus one shared expert. Sparse
attention uses NoPE: `head_dim=0` and `qk_rope_head_dim=0` are valid, and normalized
`d` is `v_head_dim=256`, not the GQA fallback 64.

Only `zai-org/GLM-5.3-Flash` is registered here. It is natively FP8 despite its
unquantized dtype being BF16. No BF16 variant fixture or alias is added. Vision
metadata is retained for provenance, not multimodal inference support. No silicon
support-matrix PASS or performance-accuracy claim follows from registration.

The source and license are also recorded in both packaged
`THIRD_PARTY_NOTICES.md` copies. Runtime BF16 upcasting of attention weights is
documented in `docs/glm5-next.md` at the repository root.

## Upstream license (verbatim)

```text
MIT License

Copyright (c) 2026 Z.AI Co., Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
