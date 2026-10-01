// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Modified: independent CPU-only analytical operation inventories, not GPU kernels.
// Architecture/serving source (Apache-2.0):
// https://github.com/vllm-project/vllm/tree/0a30bc3f9ac3cc1a9339e115377a99d32252aed0/vllm/models/glm5next
// Original paths: common/{model,kda,attention,sparse_indexer}.py,
// nvidia/sparse_indexer.py; vllm/model_executor/layers/mamba/mamba_utils.py.

//! GLM5NEXT analytical, UNVALIDATED operation inventories. No measured table is
//! consulted, including old GLM DSA, DeepSeek MLA/mHC, or Kimi KDA tables.
//!
//! Sparse prefill requires serving with `sparse_mla_force_mqa=True`. BF16 MLA
//! cache/activations, FP8 index keys, NoPE, one decode token, no CP/PP/MTP.
//! Projections and resident attention/indexer weights belong to the model's
//! GEMMs/inventory. Absorbed kv_b weights are READ here but not OWNED here.
//!
//! SOL sums serial stage rooflines. HYBRID/EMPIRICAL use the standard tableless
//! memory-op assumption: configured mem_bw_empirical_scaling_factor and
//! mem_empirical_constant_latency per modeled stage, with peak compute rates.
//! This is NOT measured utilization. Top-k uses a comparison-heap work proxy;
//! nonlinear scalar operations count as one operation, cache reads assume no
//! inter-query reuse, and kernel fusion/tile padding are not calibrated.
//! Every successful query records `analytic_unvalidated` provenance.

use serde::{Deserialize, Serialize};

use crate::common::enums::{DatabaseMode, GemmQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::{SystemSpec, quant_tc_flops};
use crate::operators::base::{PerformanceResult, SolComponents, Source};
use crate::operators::op::RuntimeContext;
use crate::operators::util_empirical::ProvenanceTier;
use crate::perf_database::PerfDatabase;

fn invalid(message: &str) -> AicError {
    AicError::InvalidEngineConfig(format!("GLM5NEXT: {message}"))
}

fn validate_scale(scale: f64) -> Result<(), AicError> {
    if !scale.is_finite() || scale < 0.0 {
        return Err(invalid("num/scale_factor must be finite and nonnegative"));
    }
    Ok(())
}

pub(crate) fn mode_ready(mode: DatabaseMode) -> Result<(), AicError> {
    if mode == DatabaseMode::Silicon {
        return Err(AicError::PerfDatabase(
            "GLM5NEXT has no measured SILICON data; use SOL or explicitly accept the analytical, unvalidated HYBRID/EMPIRICAL fallback".into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
struct Work {
    bf16: f64,
    fp8: f64,
    fp32: f64,
    bytes: f64,
}

fn positive(value: f64, name: &str) -> Result<f64, AicError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(invalid(&format!("{name} must be finite and positive")))
    }
}

fn evaluate(
    spec: &SystemSpec,
    mode: DatabaseMode,
    stages: &[Work],
    scale: f64,
) -> Result<PerformanceResult, AicError> {
    mode_ready(mode)?;
    validate_scale(scale)?;
    let sol = matches!(mode, DatabaseMode::Sol | DatabaseMode::SolFull);
    let mut result = PerformanceResult::sol(SolComponents::default());
    if scale == 0.0 || stages.is_empty() {
        result.source = if sol { Source::Sol } else { Source::Empirical };
        return Ok(result);
    }
    let bandwidth = positive(spec.gpu.mem_bw, "mem_bw")?;
    let (efficiency, overhead) = if sol {
        (1.0, 0.0)
    } else {
        let efficiency = positive(
            spec.gpu.mem_bw_empirical_scaling_factor,
            "memory efficiency",
        )?;
        let overhead = spec.gpu.mem_empirical_constant_latency;
        if !overhead.is_finite() || overhead < 0.0 {
            return Err(invalid(
                "memory latency constant must be finite and nonnegative",
            ));
        }
        (efficiency, overhead * 1000.0)
    };
    for stage in stages {
        let mut seconds = 0.0;
        for (ops, quant) in [
            (stage.bf16, GemmQuantMode::Bfloat16),
            (stage.fp8, GemmQuantMode::Fp8),
        ] {
            if ops > 0.0 {
                seconds += ops / quant_tc_flops(spec, quant.mapping())?;
            }
        }
        if stage.fp32 > 0.0 {
            let rate = spec.gpu.fp32_flops.ok_or_else(|| {
                AicError::MissingSystemFlops(
                    "GLM5NEXT needs scalar fp32_flops, not BF16 tensor-core throughput".into(),
                )
            })?;
            seconds += stage.fp32 / positive(rate, "fp32_flops")?;
        }
        let components = SolComponents::new(seconds * 1000.0, stage.bytes / bandwidth * 1000.0);
        let leaf = if sol {
            PerformanceResult::sol(components)
        } else {
            PerformanceResult::new(
                components.math_ms.max(components.mem_ms / efficiency) + overhead,
                Source::Empirical,
            )
        };
        result = result.plus(leaf);
    }
    Ok(result.scaled(scale))
}

fn query_result(
    db: &PerfDatabase,
    stages: &[Work],
    scale: f64,
) -> Result<PerformanceResult, AicError> {
    let result = evaluate(&db.system_spec, db.database_mode, stages, scale)?;
    db.note_provenance(ProvenanceTier::AnalyticUnvalidated);
    Ok(result)
}

// Closed forms avoid O(context length) work and keep boundary arithmetic exact.
fn triangle(n: u64) -> u128 {
    u128::from(n) * u128::from(n + 1) / 2
}

fn remainder_sum(n: u64, pool: u32) -> u128 {
    let r = u128::from(pool);
    let q = u128::from(n / u64::from(pool));
    let tail = u128::from(n % u64::from(pool));
    q * r * (r - 1) / 2 + tail * (tail + 1) / 2
}

fn pool_sum(n: u64, pool: u32) -> u128 {
    (triangle(n) - remainder_sum(n, pool)) / u128::from(pool)
}

fn selected_sum(n: u64, topk: u32, pool: u32) -> u128 {
    let knee = u64::from(topk);
    if n <= knee {
        triangle(n)
    } else {
        triangle(knee) + u128::from(n - knee) * u128::from(topk) + remainder_sum(n, pool)
            - remainder_sum(knee, pool)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm5NextSparseAttentionOp {
    pub name: String,
    pub scale_factor: f64,
    pub is_context: bool,
    /// LOCAL attention heads; indexer heads below are replicated, not TP-sharded.
    pub num_heads: u32,
    pub kv_lora_rank: u32,
    /// NoPE query width before absorption, not the latent attention width.
    pub qk_head_dim: u32,
    pub v_head_dim: u32,
    pub index_n_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    pub index_kpool: u32,
}

impl Glm5NextSparseAttentionOp {
    pub fn validate(&self) -> Result<(), AicError> {
        validate_scale(self.scale_factor)?;
        if [
            self.num_heads,
            self.kv_lora_rank,
            self.qk_head_dim,
            self.v_head_dim,
            self.index_n_heads,
            self.index_head_dim,
            self.index_topk,
            self.index_kpool,
        ]
        .contains(&0)
        {
            return Err(invalid("sparse geometry dimensions must be positive"));
        }
        if self.index_head_dim != 128
            || self.index_kpool < 2
            || self.index_topk % self.index_kpool != 0
        {
            return Err(invalid(
                "sparse indexer requires head_dim=128, kpool>=2, and topk divisible by kpool",
            ));
        }
        Ok(())
    }

    fn work(&self, batch: u32, s: u32, prefix: u32) -> Result<Vec<Work>, AicError> {
        self.validate()?;
        if !self.is_context && prefix != 0 {
            return Err(invalid(
                "decode s is total sequence length; prefix must be zero",
            ));
        }
        if batch == 0 || s == 0 || self.scale_factor == 0.0 {
            return Ok(vec![]);
        }
        let (start, end, q) = if self.is_context {
            (
                u64::from(prefix),
                u64::from(prefix) + u64::from(s),
                f64::from(s),
            )
        } else {
            (u64::from(s - 1), u64::from(s), 1.0)
        };
        let b = f64::from(batch);
        let t = b * q;
        let n = f64::from(self.num_heads);
        let l = f64::from(self.kv_lora_rank);
        let d = f64::from(self.index_head_dim);
        let ih = f64::from(self.index_n_heads);
        let r = f64::from(self.index_kpool);
        let pairs = b
            * (selected_sum(end, self.index_topk, self.index_kpool)
                - selected_sum(start, self.index_topk, self.index_kpool)) as f64;
        let completed = b
            * ((end / u64::from(self.index_kpool)) - (start / u64::from(self.index_kpool))) as f64;
        let index_entry = d + 4.0; // one FP32 scale for the 128 FP8 values
        let mut stages = Vec::with_capacity(7);
        // BF16 head-batched absorption. These weight reads do NOT add residency.
        for width in [self.qk_head_dim, self.v_head_dim] {
            let w = f64::from(width);
            stages.push(Work {
                bf16: 2.0 * t * n * w * l,
                bytes: 2.0 * n * w * l + 2.0 * t * n * (w + l),
                ..Work::default()
            });
        }
        // Per-token key norm, FWHT/FP8 conversion, softmax-weighted complete
        // pools and BF16 raw-key/gate tail stash. These still run below top-k.
        stages.push(Work {
            fp32: 5.0 * t * d
                + 7.0 * (t * ih * d + completed * d)
                + 8.0 * completed * r * d
                + 2.0 * t * ih,
            bytes: 8.0 * t * d
                + 3.0 * t * ih * d
                + 4.0 * t * ih
                + 4.0 * completed * r * d
                + completed * index_entry
                + 4.0 * r * d,
            ..Work::default()
        });
        // The short-sequence bypass skips only scoring/selection, not the
        // projections or cache writes. Causal candidate counts include past.
        if end > u64::from(self.index_topk) {
            let candidates =
                b * (pool_sum(end, self.index_kpool) - pool_sum(start, self.index_kpool)) as f64;
            stages.push(Work {
                fp8: 2.0 * candidates * ih * d,
                fp32: 3.0 * candidates * ih,
                bytes: candidates * index_entry + t * ih * (d + 4.0) + 4.0 * candidates,
                ..Work::default()
            });
            let selected_pools = f64::from(self.index_topk / self.index_kpool);
            stages.push(Work {
                // Explicit unvalidated comparison-heap proxy, NOT a measured
                // radix-topk timing. At least one comparison per candidate.
                fp32: candidates * selected_pools.log2().ceil().max(1.0),
                bytes: 4.0 * candidates + 4.0 * t * selected_pools,
                ..Work::default()
            });
        }
        // Expand selected pools, append the incomplete tail, and fill masked
        // padding in the 128-column-aligned int32 sparse-index workspace.
        let width =
            (u64::from(self.index_topk) + u64::from(self.index_kpool) - 1).div_ceil(128) * 128;
        stages.push(Work {
            fp32: pairs,
            bytes: 4.0 * t * width as f64 + 4.0 * pairs,
            ..Work::default()
        });
        // Latent NoPE MQA (QK and AV each width L). Model two cache-read
        // passes, sharing each latent across local heads, not across queries.
        stages.push(Work {
            bf16: 4.0 * n * l * pairs,
            fp32: 5.0 * n * pairs,
            bytes: 4.0 * l * pairs + 4.0 * t * n * l + 2.0 * t * l,
            ..Work::default()
        });
        Ok(stages)
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        if ctx.beam_width != 1 {
            return Err(invalid("sparse attention supports beam_width=1 only"));
        }
        let stages = self.work(ctx.batch_size, ctx.s, ctx.prefix)?;
        query_result(db, &stages, self.scale_factor)
    }
}

/// ONE mHC site including pre/post. A decoder layer has TWO sites. Projection
/// weights are FP32 and counted once, not once per pre/post half.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm5NextMhcOp {
    pub name: String,
    pub scale_factor: f64,
    pub hidden_size: u32,
    pub hc_mult: u32,
    pub hc_sinkhorn_iters: u32,
}

impl Glm5NextMhcOp {
    pub fn validate(&self) -> Result<(), AicError> {
        validate_scale(self.scale_factor)?;
        if self.hidden_size == 0 || self.hc_mult == 0 || self.hc_sinkhorn_iters == 0 {
            return Err(invalid(
                "mHC hidden size, stream count and Sinkhorn iterations must be positive",
            ));
        }
        Ok(())
    }

    pub fn weight_bytes(&self) -> f64 {
        let c = f64::from(self.hc_mult);
        let mix = c * (c + 2.0);
        4.0 * (mix * c * f64::from(self.hidden_size) + mix + 3.0) * self.scale_factor
    }

    fn work(&self, tokens: u32) -> Result<Vec<Work>, AicError> {
        self.validate()?;
        if tokens == 0 || self.scale_factor == 0.0 {
            return Ok(vec![]);
        }
        let (t, h, c, it) = (
            f64::from(tokens),
            f64::from(self.hidden_size),
            f64::from(self.hc_mult),
            f64::from(self.hc_sinkhorn_iters),
        );
        let dim = c * h;
        let mix = c * (c + 2.0);
        Ok(vec![
            Work {
                fp32: 2.0 * t * dim * mix
                    + 4.0 * t * dim
                    + t * (c * c * (8.0 + 4.0 * it) + 8.0 * c)
                    + 2.0 * t * dim,
                bytes: 4.0 * (mix * dim + mix + 3.0) + 2.0 * t * (dim + h) + 4.0 * t * mix,
                ..Work::default()
            },
            Work {
                fp32: 2.0 * t * c * c * h + 2.0 * t * dim,
                bytes: 2.0 * t * (2.0 * dim + h) + 4.0 * t * (c * c + c),
                ..Work::default()
            },
        ])
    }

    pub fn query(&self, db: &PerfDatabase, tokens: u32) -> Result<PerformanceResult, AicError> {
        query_result(db, &self.work(tokens)?, self.scale_factor)
    }
}

/// GLM-specific packed convolution + bounded-gate recurrence + gated RMSNorm.
/// No Kimi table lookup, projection GEMMs, or resident parameter ownership.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm5NextKdaOp {
    pub name: String,
    pub scale_factor: f64,
    pub is_context: bool,
    pub num_heads: u32,
    pub head_dim: u32,
    pub conv_kernel: u32,
}

impl Glm5NextKdaOp {
    pub fn validate(&self) -> Result<(), AicError> {
        validate_scale(self.scale_factor)?;
        if self.num_heads == 0 || self.head_dim != 128 || self.conv_kernel != 4 {
            return Err(invalid(
                "KDA requires positive LOCAL heads, head_dim=128 and conv_kernel=4",
            ));
        }
        Ok(())
    }

    fn work(&self, ctx: &RuntimeContext) -> Result<Vec<Work>, AicError> {
        self.validate()?;
        if ctx.beam_width != 1 || (!self.is_context && ctx.prefix != 0) {
            return Err(invalid(
                "KDA supports non-speculative beam_width=1; decode prefix must be zero",
            ));
        }
        if ctx.batch_size == 0 || ctx.s == 0 || self.scale_factor == 0.0 {
            return Ok(vec![]);
        }
        let b = f64::from(ctx.batch_size);
        let s = if self.is_context {
            f64::from(ctx.s)
        } else {
            1.0
        };
        let t = b * s;
        let p = f64::from(self.num_heads) * f64::from(self.head_dim);
        let state = p * f64::from(self.head_dim) * 4.0;
        let conv = f64::from(self.conv_kernel);
        let initial = if !self.is_context || ctx.prefix > 0 {
            b * state
        } else {
            0.0
        };
        // Logical scan lower bound; 64-token intermediate-state traffic is an
        // explicit unvalidated proxy, not an assertion about FlashKDA fusion.
        let intermediate = if self.is_context {
            b * (s / 64.0).ceil() * state * 2.0
        } else {
            0.0
        };
        Ok(vec![
            Work {
                fp32: 2.0 * t * 3.0 * p * conv,
                bytes: 12.0 * p * conv + 12.0 * t * p + 12.0 * b * p * (conv - 1.0),
                ..Work::default()
            },
            Work {
                fp32: 6.0 * t * p * f64::from(self.head_dim) + 8.0 * t * p,
                bytes: 10.0 * t * p + initial + b * state + intermediate,
                ..Work::default()
            },
            Work {
                fp32: 8.0 * t * p,
                bytes: 6.0 * t * p,
                ..Work::default()
            },
        ])
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        ctx: &RuntimeContext,
    ) -> Result<PerformanceResult, AicError> {
        query_result(db, &self.work(ctx)?, self.scale_factor)
    }
}

/// FP32 router/head-weight projection. Explicit scalar FLOPS, never BF16 TC.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Glm5NextFp32LinearOp {
    pub name: String,
    pub scale_factor: f64,
    pub n: u32,
    pub k: u32,
}

impl Glm5NextFp32LinearOp {
    pub fn validate(&self) -> Result<(), AicError> {
        validate_scale(self.scale_factor)?;
        if self.n == 0 || self.k == 0 {
            return Err(invalid("FP32 projection dimensions must be positive"));
        }
        Ok(())
    }

    pub fn weight_bytes(&self) -> f64 {
        4.0 * f64::from(self.n) * f64::from(self.k) * self.scale_factor
    }

    fn work(&self, tokens: u32) -> Result<Vec<Work>, AicError> {
        self.validate()?;
        let (t, n, k) = (f64::from(tokens), f64::from(self.n), f64::from(self.k));
        Ok(if tokens == 0 {
            vec![]
        } else {
            vec![Work {
                fp32: 2.0 * t * n * k,
                bytes: 4.0 * (n * k + t * (n + k)),
                ..Work::default()
            }]
        })
    }

    pub fn query(&self, db: &PerfDatabase, tokens: u32) -> Result<PerformanceResult, AicError> {
        query_result(db, &self.work(tokens)?, self.scale_factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp32_projection_has_independent_numerical_roofline() {
        use crate::common::enums::TransferPolicy;
        let root = tempfile::tempdir().unwrap();
        crate::perf_database::energy_test_fixtures::write_energy_systems_root(root.path());
        let db = PerfDatabase::load(root.path(), "testsys", "vllm", "1.0")
            .unwrap()
            .with_mode(DatabaseMode::Sol, TransferPolicy::ALL);
        let mut spec = db.system_spec.clone();
        spec.gpu.mem_bw = 1024.0;
        spec.gpu.fp32_flops = Some(512.0);
        let op = Glm5NextFp32LinearOp {
            name: "fp32".into(),
            scale_factor: 1.0,
            n: 2,
            k: 4,
        };
        // Eight [1,4] rows times [4,2]: 128 FLOPs = 250ms at 512/s.
        // Weights+input+output = (8+32+16)*4 = 224 B = 218.75ms.
        let actual = evaluate(&spec, DatabaseMode::Sol, &op.work(8).unwrap(), 1.0).unwrap();
        assert_eq!(actual.latency_ms, 250.0);
        assert_eq!(op.weight_bytes(), 32.0);
        assert_eq!(op.query(&db, 0).unwrap().latency_ms, 0.0);
        spec.gpu.fp32_flops = None;
        assert!(evaluate(&spec, DatabaseMode::Sol, &op.work(8).unwrap(), 1.0).is_err());
    }

    #[test]
    fn pooled_causal_counts_match_explicit_small_sequences() {
        // Independent enumeration: select complete pools up to topk, plus tail.
        for pool in [2u32, 4, 16] {
            let topk = 2 * pool;
            for n in 0..100u64 {
                let selected: u128 = (1..=n)
                    .map(|s| {
                        u128::from(
                            (s / u64::from(pool)).min(2) * u64::from(pool) + s % u64::from(pool),
                        )
                    })
                    .sum();
                let candidates: u128 = (1..=n).map(|s| u128::from(s / u64::from(pool))).sum();
                assert_eq!(selected_sum(n, topk, pool), selected);
                assert_eq!(pool_sum(n, pool), candidates);
            }
        }
    }

    #[test]
    fn mhc_residency_counts_one_pre_post_site() {
        // c=2,h=4 -> fn[8,8] + base[8] + scale[3] = 75 FP32 values.
        let op = Glm5NextMhcOp {
            name: "mhc".into(),
            scale_factor: 3.0,
            hidden_size: 4,
            hc_mult: 2,
            hc_sinkhorn_iters: 2,
        };
        assert_eq!(op.weight_bytes(), 900.0);
        assert_eq!(op.work(0).unwrap().len(), 0);
        assert_eq!(op.work(1).unwrap().len(), 2);
    }

    #[test]
    fn unvalidated_ops_never_claim_silicon() {
        assert!(mode_ready(DatabaseMode::Silicon).is_err());
        for mode in [
            DatabaseMode::Sol,
            DatabaseMode::SolFull,
            DatabaseMode::Hybrid,
            DatabaseMode::Empirical,
        ] {
            assert!(mode_ready(mode).is_ok());
        }
        for scale in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(validate_scale(scale).is_err());
        }
    }

    #[test]
    fn sparse_empty_prefix_and_large_context_boundaries() {
        let mut op = Glm5NextSparseAttentionOp {
            name: "sparse".into(),
            scale_factor: 1.0,
            is_context: true,
            num_heads: 16,
            kv_lora_rank: 512,
            qk_head_dim: 256,
            v_head_dim: 256,
            index_n_heads: 32,
            index_head_dim: 128,
            index_topk: 2048,
            index_kpool: 4,
        };
        assert!(op.work(0, 4096, 0).unwrap().is_empty());
        assert!(op.work(4, 0, 4096).unwrap().is_empty());
        for (s, prefix) in [
            (1, 0),
            (2048, 0),
            (2049, 0),
            (4096, 512),
            (u32::MAX, u32::MAX),
        ] {
            let stages = op.work(2, s, prefix).unwrap();
            assert!(stages.iter().all(|w| {
                [w.bf16, w.fp8, w.fp32, w.bytes]
                    .iter()
                    .all(|v| v.is_finite() && *v >= 0.0)
            }));
        }
        op.is_context = false;
        assert!(op.work(1, 4096, 1).is_err());
        op.index_kpool = 0;
        assert!(op.work(1, 1, 0).is_err());
    }
}
