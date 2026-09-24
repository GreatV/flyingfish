use crate::{
    core::{AttentionChunking, AttentionKeyChunkPolicy},
    model::TransformerChunking,
};

use ff_core::weights::{CacheGranularity, CachePolicy, DeviceCachePolicy, WeightSource};

use anyhow::{Context, Result, bail};

use candle_core::{DType, Device, Tensor};

use serde::{Deserialize, Serialize};

use std::{collections::HashMap, fs, num::NonZeroUsize, path::Path};

mod cuda_artifacts;
mod qwen;
mod verified;

pub use cuda_artifacts::*;
pub use qwen::*;
use verified::*;

pub const EXECUTION_POLICY_SCHEMA_VERSION: u32 = 4;

pub const H3_NUMERICAL_CONTRACT_SCHEMA_VERSION: u32 = 6;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionBackendPolicy {
    Cpu,
    /// CUDA. There is one CUDA backend; what a given device afforded it is
    /// recorded in [`CudaCapabilities`] rather than encoded in this name.
    Cuda,
    Metal,
}

impl ExecutionBackendPolicy {
    pub fn from_device(device: &Device) -> Self {
        if device.is_cuda() {
            Self::Cuda
        } else if device.is_metal() {
            Self::Metal
        } else {
            Self::Cpu
        }
    }

    /// Whether this backend runs on a CUDA device at all.
    pub fn is_cuda(self) -> bool {
        matches!(self, Self::Cuda)
    }
}

/// What a CUDA device afforded the operators, observed rather than assumed.
///
/// These two axes are independent and neither is a property of "which machine
/// this is" as a whole, which is why one backend value plus this record says
/// what a tier enum used to only approximate.
///
/// `tuned_kernels` covers the kernels compiled from this repository's own
/// `src/cuda/*.cu`. Their numerics are fixed by that source and the single
/// `compute_80` PTX it compiles to, so the only question is whether the
/// hardware can execute those instructions.
///
/// `reference_libraries` covers the operators that call cuBLASLt or cuDNN.
/// Those libraries select different kernels per architecture and per build, so
/// reproducing a recorded result through them is a claim about one host.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CudaCapabilities {
    pub tuned_kernels: bool,
    pub reference_libraries: bool,
}

impl CudaCapabilities {
    /// Neither axis available: a CUDA device runs Candle's kernels throughout.
    pub const NONE: Self = Self {
        tuned_kernels: false,
        reference_libraries: false,
    };

    /// Both axes available, which is what the recorded evidence was produced
    /// under and what a plan models unless told otherwise.
    pub const REFERENCE: Self = Self {
        tuned_kernels: true,
        reference_libraries: true,
    };

    pub fn from_device(device: &Device) -> Self {
        if !device.is_cuda() {
            return Self::NONE;
        }
        Self {
            tuned_kernels: crate::cuda::tuned_kernels_available(device),
            reference_libraries: crate::cuda::profile::reference_libraries_available(device),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionBackendPolicy {
    #[serde(rename = "full")]
    FullSoftmax,
    #[serde(rename = "online")]
    OnlineSoftmax,
    #[serde(rename = "flash")]
    FlashAttention,
}

/// Closed identifiers for the tensor runtime that evaluates the H3 graph.
///
/// These are semantic contract versions, not user-defined labels. A change to
/// the operations selected by one of these runtimes must introduce a new enum
/// variant and therefore changes the enclosing execution-policy hash.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3TensorBackendContract {
    CandleCpu011V1,
    /// Candle's CUDA kernels. Which host ran them, and whether that host's
    /// libraries were the reference ones, is recorded in `cuda_capabilities`
    /// and `cuda_artifacts` rather than spelled into this identifier.
    CandleCuda011V1,
    CandleMetal011V1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3TransformerRmsNormContract {
    CandleF32FusedWeightSingleCastV1,
    Pytorch7269437VectorizedBf16FusedWeightSingleCastCudaV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3TransformerRotaryContract {
    HostF32PowReciprocalDeviceF32PositionMathTrigInputDtypeRotationV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3BiasedLinearContract {
    CandleLinearV1,
    CublasLt130401Sm89Sm128Driver59584Bf16Compute32BiasEpilogueV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3OutputHeadOrderContract {
    BothHeadsOnContiguousFullPackedChunksThenModalitySelectV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3AttentionContract {
    CandleFullSoftmaxV1,
    CandleOnlineSoftmaxV1,
    /// Upstream's own width dispatch: the persistent kernel through 2048 rows
    /// and the regular register kernel from 2049 through 9216. Naming only one
    /// of them would let a recorded contract claim a kernel the run did not
    /// use.
    Pytorch7269437NativeMathPersistentAndRegularSoftmaxCudaV1,
    OnlineSoftmaxWithPersistentTokenRefinerCudaV1,
    FlashAttention011MainAndTokenRefinerCudaV1,
}

/// How one evaluation's arithmetic is divided across ranks.
///
/// A row-parallel linear computes partial products on each rank and sums them
/// across ranks. Floating-point addition is not associative, so that sum is a
/// different number from the one a single-device GEMM produces over the same
/// inputs — not approximately, but in the low bits, deterministically, every
/// time. Same weights, same operations, different order, different last bits: a
/// rounding change, which this project admits, and which therefore has to be
/// named here beside every other arithmetic decision the contract already
/// records.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3CollectivePartitionContract {
    /// One rank holds and computes the whole evaluation: no partition, no
    /// cross-rank sum, and the arithmetic every run recorded so far did.
    #[default]
    SingleRankNoCollectiveV1,
    /// Megatron-shaped: QKV and the feed-forward gate/up projections split by
    /// column, the attention output and feed-forward down projections split by
    /// row, with one all-reduce after each row-parallel projection.
    MegatronColumnRowV1,
}

/// In what order the cross-rank partial sums are put back together. This is
/// half of the arithmetic identity: the same partition reduced in a different
/// order is a different number.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum H3CollectiveReductionOrderContract {
    /// Nothing is summed across ranks, so there is no order to name.
    #[default]
    NoCrossRankSumV1,
    /// The ring's own order: each rank's partial reaches its successor first,
    /// so the sum is resolved in ascending ring order. Reproducible for a fixed
    /// rank count and a fixed ring, and not equal to the single-device result.
    RingRankAscendingV1,
}

/// The partition and reduction order one run used, and the rank count they were
/// resolved over.
///
/// The rank count is part of the identity because the same partition over a
/// different number of ranks resolves a different sum. Recording the rank count
/// in the contract is the cheap route, and the one that permanently splits the
/// evidence base; the alternative is a rank-invariant reduction that would keep
/// one contract for every configuration at a throughput cost. This field names
/// what a run actually did; it does not choose between those futures, and a
/// rank-invariant reduction would appear here as its own order rather than by
/// removing the field.
///
/// A single-rank contract is the default and is omitted from the serialized
/// form, so every policy recorded before this field existed keeps its bytes and
/// therefore its hash: the evidence base splits only for the rank counts that
/// genuinely differ. A build that predates the field rejects a multi-rank
/// policy outright, because the contract denies unknown fields.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct H3CollectiveContract {
    pub partition: H3CollectivePartitionContract,
    pub reduction_order: H3CollectiveReductionOrderContract,
    pub rank_count: u32,
}

/// A missing collective is a single-rank one, not a zero-rank one: `u32`'s own
/// default would name a contract no run can have had.
impl Default for H3CollectiveContract {
    fn default() -> Self {
        Self::SINGLE_RANK
    }
}

impl H3CollectiveContract {
    /// What every H3 run has done so far, and what this build still does.
    pub const SINGLE_RANK: Self = Self {
        partition: H3CollectivePartitionContract::SingleRankNoCollectiveV1,
        reduction_order: H3CollectiveReductionOrderContract::NoCrossRankSumV1,
        rank_count: 1,
    };

    pub const fn is_single_rank(&self) -> bool {
        self.rank_count == 1
            && matches!(
                self.partition,
                H3CollectivePartitionContract::SingleRankNoCollectiveV1
            )
            && matches!(
                self.reduction_order,
                H3CollectiveReductionOrderContract::NoCrossRankSumV1
            )
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.rank_count > 0,
            "an H3 collective contract must name at least one rank"
        );
        let partitioned = self.partition != H3CollectivePartitionContract::SingleRankNoCollectiveV1;
        let reduced = self.reduction_order != H3CollectiveReductionOrderContract::NoCrossRankSumV1;
        anyhow::ensure!(
            partitioned == reduced,
            "an H3 collective partition and its reduction order must both be present or both absent"
        );
        anyhow::ensure!(
            partitioned == (self.rank_count > 1),
            "an H3 collective contract at {} ranks disagrees with its partition {:?}",
            self.rank_count,
            self.partition
        );
        Ok(())
    }
}

fn collective_is_single_rank(collective: &H3CollectiveContract) -> bool {
    collective.is_single_rank()
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct H3NumericalContract {
    pub schema_version: u32,
    pub tensor_backend: H3TensorBackendContract,
    /// Present exactly when the backend is CUDA.
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_capabilities: Option<CudaCapabilities>,
    pub transformer_rms_norm: H3TransformerRmsNormContract,
    pub transformer_rotary: H3TransformerRotaryContract,
    pub biased_linear: H3BiasedLinearContract,
    pub output_head_order: H3OutputHeadOrderContract,
    pub attention: H3AttentionContract,
    /// The partition and reduction order this run's arithmetic used. Absent
    /// from the serialized form when it is single-rank, which is what every
    /// path in this build produces.
    #[serde(default, skip_serializing_if = "collective_is_single_rank")]
    pub collective: H3CollectiveContract,
    #[serde(deserialize_with = "crate::required_option")]
    pub cuda_artifacts: Option<H3CudaNumericalArtifacts>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionExecutionPolicy {
    pub backend: AttentionBackendPolicy,
    pub configured_projection_rows: u64,
    pub configured_query_rows: u64,
    #[serde(deserialize_with = "crate::required_option")]
    pub configured_key_rows: Option<u64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightSourcePolicy {
    Mmap,
    Memory,
}

impl From<WeightSource> for WeightSourcePolicy {
    fn from(value: WeightSource) -> Self {
        match value {
            WeightSource::Mmap => Self::Mmap,
            WeightSource::Memory => Self::Memory,
        }
    }
}

impl From<WeightSourcePolicy> for WeightSource {
    fn from(value: WeightSourcePolicy) -> Self {
        match value {
            WeightSourcePolicy::Mmap => Self::Mmap,
            WeightSourcePolicy::Memory => Self::Memory,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WeightExecutionPolicy {
    #[serde(default, skip_serializing_if = "host_phase_priority_is_off")]
    pub host_phase_priority: bool,
    /// Optional retained arithmetic weights, charged separately from live stages.
    #[serde(default, skip_serializing_if = "device_cache_is_default")]
    pub device_cache: DeviceCachePolicy,
    pub source: WeightSourcePolicy,
    pub cache_shards: u64,
    #[serde(deserialize_with = "crate::required_option")]
    pub cache_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "CacheGranularity::is_shard")]
    pub granularity: CacheGranularity,
}

fn host_phase_priority_is_off(value: &bool) -> bool {
    !*value
}

fn device_cache_is_default(policy: &DeviceCachePolicy) -> bool {
    *policy == DeviceCachePolicy::DISABLED
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionPolicy {
    pub schema_version: u32,
    pub execution_backend: ExecutionBackendPolicy,
    /// Not optional: a policy without a numerical contract described a run
    /// whose arithmetic was unbound, which this build cannot execute and no
    /// longer accepts.
    pub numerics: Box<H3NumericalContract>,
    pub attention: AttentionExecutionPolicy,
    pub configured_ffn_rows: u64,
    pub configured_output_rows: u64,
    pub weights: WeightExecutionPolicy,
    pub precompute_adaln: bool,
}

#[derive(Default)]
enum PresentOption<T> {
    #[default]
    Missing,
    Present(Option<T>),
}

fn deserialize_present_option<'de, D, T>(
    deserializer: D,
) -> std::result::Result<PresentOption<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(PresentOption::Present)
}

impl<'de> Deserialize<'de> for ExecutionPolicy {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            schema_version: u32,
            execution_backend: ExecutionBackendPolicy,
            #[serde(default, deserialize_with = "deserialize_present_option")]
            numerics: PresentOption<H3NumericalContract>,
            attention: AttentionExecutionPolicy,
            configured_ffn_rows: u64,
            configured_output_rows: u64,
            weights: WeightExecutionPolicy,
            precompute_adaln: bool,
        }

        let wire = Wire::deserialize(deserializer)?;
        let numerics = match (wire.schema_version, wire.numerics) {
            (EXECUTION_POLICY_SCHEMA_VERSION, PresentOption::Present(Some(numerics))) => {
                Box::new(numerics)
            }
            (EXECUTION_POLICY_SCHEMA_VERSION, PresentOption::Missing) => {
                return Err(serde::de::Error::custom(
                    "execution-policy schema 4 requires numerics",
                ));
            }
            (EXECUTION_POLICY_SCHEMA_VERSION, PresentOption::Present(None)) => {
                return Err(serde::de::Error::custom(
                    "execution-policy schema 4 requires non-null numerics",
                ));
            }
            (schema, _) => {
                return Err(serde::de::Error::custom(format!(
                    "unsupported execution-policy schema {schema}; this build supports schema {EXECUTION_POLICY_SCHEMA_VERSION}"
                )));
            }
        };
        Ok(Self {
            schema_version: wire.schema_version,
            execution_backend: wire.execution_backend,
            numerics,
            attention: wire.attention,
            configured_ffn_rows: wire.configured_ffn_rows,
            configured_output_rows: wire.configured_output_rows,
            weights: wire.weights,
            precompute_adaln: wire.precompute_adaln,
        })
    }
}

/// One operator's contract per capability axis, so construction and validation
/// cannot drift apart.
///
/// Which axis each operator reads is the same split the dispatch code makes:
/// an operator compiled from this repository's sources reads `tuned_kernels`,
/// one that calls the reference cuBLASLt or cuDNN path reads
/// `reference_libraries`. FlashAttention has its own build artifacts and does
/// not require those vendor-library identities.
struct ContractShape {
    tensor_backend: H3TensorBackendContract,
    transformer_rms_norm: H3TransformerRmsNormContract,
    biased_linear: H3BiasedLinearContract,
    attention: H3AttentionContract,
    cuda_artifacts: Option<H3CudaNumericalArtifacts>,
}

fn contract_shape(
    execution_backend: ExecutionBackendPolicy,
    capabilities: Option<CudaCapabilities>,
    attention_backend: AttentionBackendPolicy,
) -> Result<ContractShape> {
    let (tensor_backend, label) = match execution_backend {
        ExecutionBackendPolicy::Cpu => (H3TensorBackendContract::CandleCpu011V1, "CPU"),
        ExecutionBackendPolicy::Metal => (H3TensorBackendContract::CandleMetal011V1, "Metal"),
        ExecutionBackendPolicy::Cuda => (H3TensorBackendContract::CandleCuda011V1, "CUDA"),
    };
    let Some(capabilities) = capabilities else {
        anyhow::ensure!(
            !execution_backend.is_cuda(),
            "a CUDA H3 numerical contract must record the device capabilities"
        );
        return Ok(ContractShape {
            tensor_backend,
            transformer_rms_norm: H3TransformerRmsNormContract::CandleF32FusedWeightSingleCastV1,
            biased_linear: H3BiasedLinearContract::CandleLinearV1,
            attention: non_cuda_attention_contract(attention_backend, label)?,
            cuda_artifacts: None,
        });
    };
    anyhow::ensure!(
        execution_backend.is_cuda(),
        "a non-CUDA H3 numerical contract must not record device capabilities"
    );
    Ok(ContractShape {
        tensor_backend,
        transformer_rms_norm: if capabilities.tuned_kernels {
            H3TransformerRmsNormContract::Pytorch7269437VectorizedBf16FusedWeightSingleCastCudaV1
        } else {
            H3TransformerRmsNormContract::CandleF32FusedWeightSingleCastV1
        },
        biased_linear: if capabilities.reference_libraries {
            H3BiasedLinearContract::CublasLt130401Sm89Sm128Driver59584Bf16Compute32BiasEpilogueV1
        } else {
            H3BiasedLinearContract::CandleLinearV1
        },
        attention: if capabilities.reference_libraries
            || attention_backend == AttentionBackendPolicy::FlashAttention
        {
            attention_contract_for(execution_backend, attention_backend)?
        } else {
            non_cuda_attention_contract(attention_backend, "CUDA without the reference libraries")?
        },
        cuda_artifacts: Some(cuda_artifacts_for(capabilities, attention_backend)),
    })
}

impl H3NumericalContract {
    /// The contract for a target whose CUDA device affords the reference
    /// capabilities, which is what a plan models unless told otherwise.
    ///
    /// This is deliberately independent of compiled features so a CPU
    /// coordinator can plan CUDA work without inventing artifact identities.
    /// The CUDA worker verifies its embedded artifacts against these constants
    /// immediately before execution.
    pub fn for_verified_target(
        execution_backend: ExecutionBackendPolicy,
        attention_backend: AttentionBackendPolicy,
    ) -> Result<Self> {
        Self::for_target(
            execution_backend,
            execution_backend
                .is_cuda()
                .then_some(CudaCapabilities::REFERENCE),
            attention_backend,
        )
    }

    /// The contract for a target with the capabilities its device actually
    /// affords. Execution paths pass what they observed; planning paths pass
    /// what they are modelling.
    pub fn for_target(
        execution_backend: ExecutionBackendPolicy,
        capabilities: Option<CudaCapabilities>,
        attention_backend: AttentionBackendPolicy,
    ) -> Result<Self> {
        let ContractShape {
            tensor_backend,
            transformer_rms_norm,
            biased_linear,
            attention,
            cuda_artifacts,
        } = contract_shape(execution_backend, capabilities, attention_backend)?;
        let contract = Self {
            schema_version: H3_NUMERICAL_CONTRACT_SCHEMA_VERSION,
            tensor_backend,
            cuda_capabilities: capabilities,
            transformer_rms_norm,
            transformer_rotary: H3TransformerRotaryContract::HostF32PowReciprocalDeviceF32PositionMathTrigInputDtypeRotationV1,
            biased_linear,
            output_head_order:
                H3OutputHeadOrderContract::BothHeadsOnContiguousFullPackedChunksThenModalitySelectV1,
            attention,
            collective: H3CollectiveContract::SINGLE_RANK,
            cuda_artifacts,
        };
        contract.validate_for(execution_backend, capabilities, attention_backend)?;
        Ok(contract)
    }

    fn validate_for(
        &self,
        execution_backend: ExecutionBackendPolicy,
        capabilities: Option<CudaCapabilities>,
        attention_backend: AttentionBackendPolicy,
    ) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == H3_NUMERICAL_CONTRACT_SCHEMA_VERSION,
            "unsupported H3 numerical-contract schema {}; this build supports schema {}",
            self.schema_version,
            H3_NUMERICAL_CONTRACT_SCHEMA_VERSION
        );
        anyhow::ensure!(
            self.cuda_capabilities == capabilities,
            "H3 numerical contract capabilities {:?} disagree with {:?}",
            self.cuda_capabilities,
            capabilities
        );
        let expected = contract_shape(execution_backend, capabilities, attention_backend)?;
        anyhow::ensure!(
            self.tensor_backend == expected.tensor_backend,
            "H3 numerical contract tensor backend {:?} is incompatible with {:?}",
            self.tensor_backend,
            execution_backend
        );
        anyhow::ensure!(
            self.transformer_rms_norm == expected.transformer_rms_norm,
            "H3 numerical contract RMSNorm {:?} is incompatible with {:?}",
            self.transformer_rms_norm,
            execution_backend
        );
        anyhow::ensure!(
            self.biased_linear == expected.biased_linear,
            "H3 numerical contract biased linear {:?} is incompatible with {:?}",
            self.biased_linear,
            execution_backend
        );
        anyhow::ensure!(
            self.attention == expected.attention,
            "H3 numerical contract attention {:?} is incompatible with {:?}/{:?}",
            self.attention,
            execution_backend,
            attention_backend
        );
        self.collective.validate()?;
        match (execution_backend, &self.cuda_artifacts) {
            (ExecutionBackendPolicy::Cuda, Some(artifacts)) => {
                artifacts.validate()?;
                anyhow::ensure!(
                    artifacts.tuned_kernels.is_some() == capabilities.unwrap().tuned_kernels
                        && artifacts.reference_libraries.is_some()
                            == capabilities.unwrap().reference_libraries
                        && artifacts.attention.is_some()
                            == expected
                                .cuda_artifacts
                                .as_ref()
                                .unwrap()
                                .attention
                                .is_some(),
                    "H3 CUDA numerical artifact presence disagrees with its capabilities and attention backend"
                );
                if let Some(attention) = &artifacts.attention {
                    let attention_matches = matches!(
                        (attention_backend, attention),
                        (
                            AttentionBackendPolicy::FullSoftmax,
                            H3CudaAttentionArtifacts::PytorchNativeMathPersistentSoftmax
                        ) | (
                            AttentionBackendPolicy::OnlineSoftmax,
                            H3CudaAttentionArtifacts::OnlineSoftmaxWithPersistentTokenRefiner
                        ) | (
                            AttentionBackendPolicy::FlashAttention,
                            H3CudaAttentionArtifacts::CandleFlashAttention011
                        )
                    );
                    anyhow::ensure!(
                        attention_matches,
                        "H3 CUDA numerical attention artifacts are incompatible with {:?}",
                        attention_backend
                    );
                }
            }
            (ExecutionBackendPolicy::Cuda, None) => {
                bail!("H3 CUDA numerical contract is missing cuda_artifacts")
            }
            (ExecutionBackendPolicy::Cpu | ExecutionBackendPolicy::Metal, None) => {}
            (ExecutionBackendPolicy::Cpu | ExecutionBackendPolicy::Metal, Some(_)) => {
                bail!("non-CUDA H3 numerical contract must not contain cuda_artifacts")
            }
        }
        Ok(())
    }

    fn ensure_current_for_build(
        &self,
        execution_backend: ExecutionBackendPolicy,
        attention_backend: AttentionBackendPolicy,
    ) -> Result<()> {
        let current =
            Self::for_target(execution_backend, self.cuda_capabilities, attention_backend)?;
        if let Some(field) = self.first_difference(&current) {
            bail!("execution policy numerical contract differs from this binary at {field}");
        }
        anyhow::ensure!(
            crate::model::H3_OUTPUT_HEAD_ORDER_BACKEND == VERIFIED_H3_OUTPUT_HEAD_ORDER_BACKEND,
            "compiled H3 numerical contract mismatch at numerics.output_head_order"
        );
        if execution_backend == ExecutionBackendPolicy::Cuda {
            validate_compiled_cuda_contract(
                self.cuda_capabilities
                    .context("CUDA numerical contract is missing capabilities")?,
                attention_backend,
            )?;
        }
        Ok(())
    }

    fn first_difference(&self, other: &Self) -> Option<&'static str> {
        if self.schema_version != other.schema_version {
            Some("numerics.schema_version")
        } else if self.cuda_capabilities != other.cuda_capabilities {
            Some("numerics.cuda_capabilities")
        } else if self.tensor_backend != other.tensor_backend {
            Some("numerics.tensor_backend")
        } else if self.transformer_rms_norm != other.transformer_rms_norm {
            Some("numerics.transformer_rms_norm")
        } else if self.transformer_rotary != other.transformer_rotary {
            Some("numerics.transformer_rotary")
        } else if self.biased_linear != other.biased_linear {
            Some("numerics.biased_linear")
        } else if self.output_head_order != other.output_head_order {
            Some("numerics.output_head_order")
        } else if self.attention != other.attention {
            Some("numerics.attention")
        } else if self.collective != other.collective {
            Some("numerics.collective")
        } else {
            match (&self.cuda_artifacts, &other.cuda_artifacts) {
                (Some(left), Some(right)) => left.first_difference(right),
                (None, None) => None,
                _ => Some("numerics.cuda_artifacts"),
            }
        }
    }
}

fn non_cuda_attention_contract(
    attention_backend: AttentionBackendPolicy,
    device_label: &str,
) -> Result<H3AttentionContract> {
    match attention_backend {
        AttentionBackendPolicy::FullSoftmax => Ok(H3AttentionContract::CandleFullSoftmaxV1),
        AttentionBackendPolicy::OnlineSoftmax => Ok(H3AttentionContract::CandleOnlineSoftmaxV1),
        AttentionBackendPolicy::FlashAttention => {
            bail!("FlashAttention requires CUDA, not the {device_label} numerical backend")
        }
    }
}

fn attention_contract_for(
    execution_backend: ExecutionBackendPolicy,
    attention_backend: AttentionBackendPolicy,
) -> Result<H3AttentionContract> {
    match execution_backend {
        ExecutionBackendPolicy::Cpu => non_cuda_attention_contract(attention_backend, "CPU"),
        ExecutionBackendPolicy::Metal => non_cuda_attention_contract(attention_backend, "Metal"),
        ExecutionBackendPolicy::Cuda => Ok(match attention_backend {
            AttentionBackendPolicy::FullSoftmax => {
                H3AttentionContract::Pytorch7269437NativeMathPersistentAndRegularSoftmaxCudaV1
            }
            AttentionBackendPolicy::OnlineSoftmax => {
                H3AttentionContract::OnlineSoftmaxWithPersistentTokenRefinerCudaV1
            }
            AttentionBackendPolicy::FlashAttention => {
                H3AttentionContract::FlashAttention011MainAndTokenRefinerCudaV1
            }
        }),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EffectiveAttentionChunks {
    pub projection_rows: NonZeroUsize,
    pub query_rows: NonZeroUsize,
    pub key_rows: NonZeroUsize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EffectiveExecutionChunks {
    pub attention: EffectiveAttentionChunks,
    pub feed_forward_rows: NonZeroUsize,
    pub output_rows: NonZeroUsize,
}

impl ExecutionPolicy {
    /// Rebind the closed attention semantic identifier after a solver changes
    /// `attention.backend`. Shared CUDA RMSNorm and biased-linear identities
    /// stay fixed while the mode-specific full, online, or flash artifact
    /// identity is replaced as one closed unit.
    pub fn rebind_attention_numerics(&mut self) -> Result<()> {
        let numerics = &mut self.numerics;
        let capabilities = numerics.cuda_capabilities;
        let shape = contract_shape(self.execution_backend, capabilities, self.attention.backend)?;
        numerics.attention = shape.attention;
        numerics.cuda_artifacts = shape.cuda_artifacts;
        numerics.validate_for(self.execution_backend, capabilities, self.attention.backend)
    }

    /// The partition and reduction order this policy's arithmetic used.
    pub fn collective(&self) -> H3CollectiveContract {
        self.numerics.collective
    }

    pub fn first_numerical_difference(&self, other: &Self) -> Option<&'static str> {
        if self.schema_version != other.schema_version {
            Some("schema_version")
        } else if self.execution_backend != other.execution_backend {
            Some("execution_backend")
        } else if self.numerics != other.numerics {
            self.numerics
                .first_difference(&other.numerics)
                .or(Some("numerics"))
        } else {
            None
        }
    }

    pub fn first_difference(&self, other: &Self) -> Option<&'static str> {
        if let Some(field) = self.first_numerical_difference(other) {
            Some(field)
        } else if self.attention.backend != other.attention.backend {
            Some("attention.backend")
        } else if self.attention.configured_projection_rows
            != other.attention.configured_projection_rows
        {
            Some("attention.configured_projection_rows")
        } else if self.attention.configured_query_rows != other.attention.configured_query_rows {
            Some("attention.configured_query_rows")
        } else if self.attention.configured_key_rows != other.attention.configured_key_rows {
            Some("attention.configured_key_rows")
        } else if self.configured_ffn_rows != other.configured_ffn_rows {
            Some("configured_ffn_rows")
        } else if self.configured_output_rows != other.configured_output_rows {
            Some("configured_output_rows")
        } else if self.weights.source != other.weights.source {
            Some("weights.source")
        } else if self.weights.granularity != other.weights.granularity {
            Some("weights.granularity")
        } else if self.weights.cache_shards != other.weights.cache_shards {
            Some("weights.cache_shards")
        } else if self.weights.cache_bytes != other.weights.cache_bytes {
            Some("weights.cache_bytes")
        } else if self.weights.host_phase_priority != other.weights.host_phase_priority {
            Some("weights.host_phase_priority")
        } else if self.weights.device_cache != other.weights.device_cache {
            Some("weights.device_cache")
        } else if self.precompute_adaln != other.precompute_adaln {
            Some("precompute_adaln")
        } else {
            None
        }
    }

    pub fn from_runtime(
        device: &Device,
        weight_source: WeightSource,
        cache_policy: CachePolicy,
        chunking: TransformerChunking,
        flash_attention: bool,
        precompute_adaln: bool,
    ) -> Result<Self> {
        anyhow::ensure!(
            !(flash_attention
                && matches!(chunking.attention.key, AttentionKeyChunkPolicy::Chunked(_))),
            "FlashAttention and online key-chunked attention are mutually exclusive"
        );
        let (backend, configured_key_rows) = if flash_attention {
            (AttentionBackendPolicy::FlashAttention, None)
        } else {
            match chunking.attention.key {
                AttentionKeyChunkPolicy::Full => (AttentionBackendPolicy::FullSoftmax, None),
                AttentionKeyChunkPolicy::Chunked(rows) => (
                    AttentionBackendPolicy::OnlineSoftmax,
                    Some(rows.get() as u64),
                ),
            }
        };
        let policy = Self {
            schema_version: EXECUTION_POLICY_SCHEMA_VERSION,
            execution_backend: ExecutionBackendPolicy::from_device(device),
            numerics: Box::new(H3NumericalContract::for_target(
                ExecutionBackendPolicy::from_device(device),
                ExecutionBackendPolicy::from_device(device)
                    .is_cuda()
                    .then(|| CudaCapabilities::from_device(device)),
                backend,
            )?),
            attention: AttentionExecutionPolicy {
                backend,
                configured_projection_rows: chunking.attention.projection_chunk_size.get() as u64,
                configured_query_rows: match backend {
                    AttentionBackendPolicy::FlashAttention => {
                        chunking.attention.projection_chunk_size.get() as u64
                    }
                    AttentionBackendPolicy::FullSoftmax | AttentionBackendPolicy::OnlineSoftmax => {
                        chunking.attention.query_chunk_size.get() as u64
                    }
                },
                configured_key_rows,
            },
            configured_ffn_rows: chunking.feed_forward_chunk_size.get() as u64,
            configured_output_rows: chunking.output_chunk_size.get() as u64,
            weights: WeightExecutionPolicy {
                host_phase_priority: false,
                device_cache: Default::default(),
                source: weight_source.into(),
                cache_shards: recorded_cache_shards(cache_policy),
                cache_bytes: cache_policy.max_bytes,
                granularity: cache_policy.granularity,
            },
            precompute_adaln,
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.schema_version == EXECUTION_POLICY_SCHEMA_VERSION,
            "unsupported execution-policy schema {}; this build supports schema {EXECUTION_POLICY_SCHEMA_VERSION}",
            self.schema_version
        );
        self.numerics.validate_for(
            self.execution_backend,
            self.numerics.cuda_capabilities,
            self.attention.backend,
        )?;
        for (name, value) in [
            (
                "configured_projection_rows",
                self.attention.configured_projection_rows,
            ),
            (
                "configured_query_rows",
                self.attention.configured_query_rows,
            ),
            ("configured_ffn_rows", self.configured_ffn_rows),
            ("configured_output_rows", self.configured_output_rows),
            ("cache_shards", self.weights.cache_shards),
        ] {
            anyhow::ensure!(value > 0, "execution policy {name} must be non-zero");
            usize::try_from(value)
                .with_context(|| format!("execution policy {name} exceeds usize"))?;
        }
        anyhow::ensure!(
            !self.weights.host_phase_priority
                || (self.weights.granularity == CacheGranularity::Tensor
                    && self.weights.cache_bytes.is_some()),
            "host phase priority requires a byte-bounded tensor cache"
        );
        if let Some(bytes) = self.weights.cache_bytes {
            anyhow::ensure!(
                bytes > 0,
                "execution policy cache_bytes must be non-zero when present"
            );
        }
        match (self.attention.backend, self.attention.configured_key_rows) {
            (AttentionBackendPolicy::OnlineSoftmax, Some(rows)) => {
                anyhow::ensure!(rows > 0, "online attention key rows must be non-zero");
                usize::try_from(rows).context("online attention key rows exceed usize")?;
            }
            (AttentionBackendPolicy::OnlineSoftmax, None) => {
                bail!("online attention requires configured_key_rows")
            }
            (
                AttentionBackendPolicy::FullSoftmax | AttentionBackendPolicy::FlashAttention,
                None,
            ) => {}
            (
                AttentionBackendPolicy::FullSoftmax | AttentionBackendPolicy::FlashAttention,
                Some(_),
            ) => {
                bail!("full and flash attention require configured_key_rows to be null")
            }
        }
        if self.attention.backend == AttentionBackendPolicy::FlashAttention {
            anyhow::ensure!(
                self.execution_backend == ExecutionBackendPolicy::Cuda,
                "flash attention requires the CUDA execution backend"
            );
        }
        Ok(())
    }

    pub fn validate_device(&self, device: &Device) -> Result<()> {
        self.validate()?;
        let actual = ExecutionBackendPolicy::from_device(device);
        anyhow::ensure!(
            self.execution_backend == actual,
            "execution policy requires {:?}, but the selected device uses {:?}",
            self.execution_backend,
            actual
        );
        let numerics = &self.numerics;
        anyhow::ensure!(
            numerics.cuda_capabilities
                == device
                    .is_cuda()
                    .then(|| CudaCapabilities::from_device(device)),
            "selected device capabilities differ from the execution-policy numerical contract"
        );
        if self.flash_attention() {
            crate::cuda::validate_flash_attention_device(device)?;
        }
        numerics.ensure_current_for_build(self.execution_backend, self.attention.backend)?;
        if self.execution_backend == ExecutionBackendPolicy::Cuda {
            crate::cuda::profile::validate_selected_device(device)?;
            if numerics.cuda_capabilities.unwrap().reference_libraries {
                crate::cuda::profile::validate_exact_profile(device).context(
                    "selected CUDA device does not satisfy the execution-policy numerical profile",
                )?;
            }
        }
        Ok(())
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        #[derive(Deserialize)]
        struct SchemaEnvelope {
            schema_version: u32,
        }

        let envelope: SchemaEnvelope =
            serde_json::from_slice(bytes).context("invalid execution-policy JSON")?;
        anyhow::ensure!(
            envelope.schema_version == EXECUTION_POLICY_SCHEMA_VERSION,
            "unsupported execution-policy schema {}; this build supports schema {}",
            envelope.schema_version,
            EXECUTION_POLICY_SCHEMA_VERSION
        );
        let policy: Self =
            serde_json::from_slice(bytes).context("invalid execution-policy JSON")?;
        policy.validate()?;
        Ok(policy)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read execution policy {}", path.display()))?;
        Self::from_json(&bytes)
            .with_context(|| format!("failed to load execution policy {}", path.display()))
    }

    /// Resolve the policy for a run: load a recorded one from `path`, or
    /// build one from the runtime inputs, then check it can execute on
    /// `device`.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        path: Option<&Path>,
        device: &Device,
        weight_source: WeightSource,
        cache_policy: CachePolicy,
        chunks: TransformerChunking,
        flash_attention: bool,
        precompute_adaln: bool,
    ) -> Result<Self> {
        let policy = match path {
            Some(path) => Self::load(path)?,
            None => Self::from_runtime(
                device,
                weight_source,
                cache_policy,
                chunks,
                flash_attention,
                precompute_adaln,
            )?,
        };
        policy.validate_executable(device)?;
        Ok(policy)
    }

    /// Check the policy can actually run on `device` with this build.
    pub fn validate_executable(&self, device: &Device) -> Result<()> {
        self.validate_device(device)?;
        anyhow::ensure!(
            !self.flash_attention() || cfg!(feature = "flash-attn"),
            "execution policy selects FlashAttention, but this binary was not compiled with \
             --features flash-attn"
        );
        Ok(())
    }

    /// Choose an attention backend the request can actually run on.
    ///
    /// The exact full-softmax path covers a bounded number of packed rows.
    /// Past that bound it is not that the operator prefers another backend --
    /// full softmax does not run the request at all -- and the row count is
    /// known here. Online softmax processes keys in blocks instead, so the
    /// bound applies per block rather than to the sequence.
    ///
    /// The block is the largest the verified exact-softmax kernels cover,
    /// which is the same bound the full path ran out of: larger blocks mean
    /// fewer passes, and the modelled peak grows slowly enough with block
    /// size that memory does not decide this. A pinned or replayed policy is
    /// left alone, and so is one that already names a backend without the
    /// bound.
    pub fn promote_attention_backend_for_rows(
        &mut self,
        on_cuda: bool,
        packed_rows: u64,
        operator_named_the_backend: bool,
    ) -> Result<()> {
        let bound = u64::try_from(crate::core::CUDA_EXACT_SOFTMAX_MAX_KEY_ROWS)
            .context("exact softmax row bound exceeds u64")?;
        if operator_named_the_backend
            || !on_cuda
            || self.attention.backend != AttentionBackendPolicy::FullSoftmax
            || packed_rows <= bound
        {
            return Ok(());
        }
        self.attention.backend = AttentionBackendPolicy::OnlineSoftmax;
        self.attention.configured_key_rows = Some(bound);
        self.rebind_attention_numerics()?;
        eprintln!(
            "attention: {packed_rows} packed rows exceed the {bound} full softmax covers; \
             using online softmax over {bound}-row key blocks"
        );
        Ok(())
    }

    /// Select the policy for a resume: check the requested policy against
    /// the recorded one, fall back to the recorded one, or require one is
    /// present.
    pub fn select_for_resume(
        requested: Self,
        recorded: Option<&Self>,
        explicit_policy: bool,
        explicit_policy_settings: bool,
        requires_recorded_policy: bool,
    ) -> Result<Self> {
        let mut requested = requested;
        match recorded {
            Some(recorded) => {
                if explicit_policy || explicit_policy_settings {
                    // Device residency is planned against whatever the card
                    // has free at that moment, so the recorded ceiling
                    // describes the first run's machine rather than anything
                    // the operator asked for. Two runs minutes apart can plan
                    // differently and agree on every choice a person made.
                    // Carry the recorded ceiling forward -- keeping the run's
                    // placement stable across a resume -- and let admission
                    // decide whether it still fits; refusing the resume for
                    // it would name a field no H3 command even exposes as a
                    // flag.
                    if !explicit_policy {
                        requested.weights.device_cache = recorded.weights.device_cache;
                    }
                    if let Some(field) = recorded.first_difference(&requested) {
                        bail!(
                            "the checkpoint's execution policy disagrees with the requested one \
                             at {field}"
                        );
                    }
                    Ok(requested)
                } else {
                    Ok(recorded.clone())
                }
            }
            None if requires_recorded_policy => bail!("checkpoint has no execution policy"),
            None => Ok(requested),
        }
    }

    /// A conservative baseline policy (minimal chunk sizes) for a solver to
    /// search upward from.
    pub fn conservative(cpu: bool) -> Result<Self> {
        let execution_backend = if cpu {
            ExecutionBackendPolicy::Cpu
        } else {
            ExecutionBackendPolicy::Cuda
        };
        let policy = Self {
            schema_version: EXECUTION_POLICY_SCHEMA_VERSION,
            execution_backend,
            numerics: Box::new(H3NumericalContract::for_verified_target(
                execution_backend,
                AttentionBackendPolicy::FullSoftmax,
            )?),
            attention: AttentionExecutionPolicy {
                backend: AttentionBackendPolicy::FullSoftmax,
                configured_projection_rows: 1,
                configured_query_rows: 1,
                configured_key_rows: None,
            },
            configured_ffn_rows: 1,
            configured_output_rows: 1,
            weights: WeightExecutionPolicy {
                host_phase_priority: false,
                device_cache: Default::default(),
                source: WeightSourcePolicy::Mmap,
                cache_shards: 1,
                cache_bytes: None,
                granularity: CacheGranularity::Shard,
            },
            precompute_adaln: true,
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        serde_json::to_vec(self).context("failed to serialize execution policy")
    }

    pub fn cache_policy(&self) -> Result<CachePolicy> {
        self.validate()?;
        let shards = usize::try_from(self.weights.cache_shards)
            .context("execution policy cache_shards exceeds usize")?;
        Ok((match self.weights.cache_bytes {
            Some(bytes)
                if self.weights.cache_shards == 1
                    && self.weights.granularity == CacheGranularity::Shard =>
            {
                CachePolicy::unbounded_units().with_max_bytes(bytes)
            }
            Some(bytes) => CachePolicy::new(shards).with_max_bytes(bytes),
            None => CachePolicy::new(shards),
        })
        .with_granularity(self.weights.granularity))
    }

    pub fn weight_source(&self) -> WeightSource {
        self.weights.source.into()
    }

    pub fn transformer_chunking(&self) -> Result<TransformerChunking> {
        self.validate()?;
        let nonzero = |name: &str, value: u64| -> Result<NonZeroUsize> {
            let value = usize::try_from(value)
                .with_context(|| format!("execution policy {name} exceeds usize"))?;
            NonZeroUsize::new(value)
                .with_context(|| format!("execution policy {name} must be non-zero"))
        };
        let key = match self.attention.backend {
            AttentionBackendPolicy::OnlineSoftmax => {
                let rows = usize::try_from(
                    self.attention
                        .configured_key_rows
                        .context("online attention requires configured_key_rows")?,
                )
                .context("online attention key rows exceed usize")?;
                AttentionKeyChunkPolicy::chunked(rows)
                    .context("invalid online attention key rows")?
            }
            AttentionBackendPolicy::FullSoftmax | AttentionBackendPolicy::FlashAttention => {
                AttentionKeyChunkPolicy::Full
            }
        };
        Ok(TransformerChunking {
            attention: AttentionChunking {
                projection_chunk_size: nonzero(
                    "configured_projection_rows",
                    self.attention.configured_projection_rows,
                )?,
                query_chunk_size: nonzero(
                    "configured_query_rows",
                    self.attention.configured_query_rows,
                )?,
                key,
            },
            feed_forward_chunk_size: nonzero("configured_ffn_rows", self.configured_ffn_rows)?,
            output_chunk_size: nonzero("configured_output_rows", self.configured_output_rows)?,
        })
    }

    pub fn effective_chunks(
        &self,
        sequence_rows: NonZeroUsize,
        output_rows: NonZeroUsize,
    ) -> Result<EffectiveExecutionChunks> {
        let configured = self.transformer_chunking()?;
        let projection_rows = configured
            .attention
            .projection_chunk_size
            .min(sequence_rows);
        Ok(EffectiveExecutionChunks {
            attention: EffectiveAttentionChunks {
                projection_rows,
                query_rows: configured.attention.query_chunk_size.min(projection_rows),
                key_rows: configured.attention.key.effective_chunk_size(sequence_rows),
            },
            feed_forward_rows: configured.feed_forward_chunk_size.min(sequence_rows),
            output_rows: configured.output_chunk_size.min(output_rows),
        })
    }

    /// Require the exact official output-head GEMM shape for a parity gate.
    /// Production may deliberately use multiple contiguous packed chunks, but
    /// that is a different numerical contract from one full-packed GEMM even
    /// though both heads still run before modality selection.
    pub fn validate_official_full_packed_output_heads(
        &self,
        packed_rows: NonZeroUsize,
    ) -> Result<()> {
        self.validate()?;
        let numerics = &self.numerics;
        anyhow::ensure!(
            numerics.output_head_order
                == H3OutputHeadOrderContract::BothHeadsOnContiguousFullPackedChunksThenModalitySelectV1,
            "execution policy does not select the official both-heads-before-modality output order"
        );
        let packed_rows = u64::try_from(packed_rows.get()).context("packed rows exceed u64")?;
        anyhow::ensure!(
            self.configured_output_rows >= packed_rows,
            "official full-packed output-head parity requires configured_output_rows >= packed rows ({packed_rows}), got {}",
            self.configured_output_rows
        );
        Ok(())
    }

    pub fn flash_attention(&self) -> bool {
        self.attention.backend == AttentionBackendPolicy::FlashAttention
    }
}

fn recorded_cache_shards(policy: CachePolicy) -> u64 {
    if policy.granularity == CacheGranularity::Shard && policy.max_shards == usize::MAX {
        1
    } else {
        policy.max_shards as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TransformerChunking;

    fn sample_policy() -> ExecutionPolicy {
        ExecutionPolicy::from_runtime(
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::new(2).with_max_bytes(64 * 1024 * 1024),
            TransformerChunking::default(),
            false,
            true,
        )
        .unwrap()
    }

    fn cuda_policy(attention: AttentionBackendPolicy) -> ExecutionPolicy {
        let mut policy = sample_policy();
        policy.execution_backend = ExecutionBackendPolicy::Cuda;
        policy.attention.backend = attention;
        policy.attention.configured_key_rows =
            (attention == AttentionBackendPolicy::OnlineSoftmax).then_some(16);
        policy.numerics = Box::new(
            H3NumericalContract::for_verified_target(ExecutionBackendPolicy::Cuda, attention)
                .unwrap(),
        );
        policy.validate().unwrap();
        policy
    }

    fn conditioning_contract(
        backend: ExecutionBackendPolicy,
        configured_query_rows: usize,
        language_rows: usize,
        max_vision_segment_rows: usize,
    ) -> H3QwenNumericalContract {
        H3QwenNumericalContract::for_verified_target(
            backend,
            NonZeroUsize::new(configured_query_rows).unwrap(),
            NonZeroUsize::new(language_rows).unwrap(),
            max_vision_segment_rows,
            H3QwenVisionLinearGeometry::from_patch_rows(0, 0).unwrap(),
        )
        .unwrap()
    }

    fn vision_conditioning_contract(
        backend: ExecutionBackendPolicy,
        configured_query_rows: usize,
        language_rows: usize,
        grids: &[(H3QwenVisionGridModality, usize, usize, usize)],
    ) -> H3QwenNumericalContract {
        let mut image_rows = 0usize;
        let mut video_rows = 0usize;
        let mut max_segment = 0usize;
        for &(modality, temporal, height, width) in grids {
            let segment = height * width;
            let rows = temporal * segment;
            match modality {
                H3QwenVisionGridModality::Image => image_rows += rows,
                H3QwenVisionGridModality::Video => video_rows += rows,
            }
            max_segment = max_segment.max(segment);
        }
        H3QwenNumericalContract::for_verified_target_with_grids(
            backend,
            NonZeroUsize::new(configured_query_rows).unwrap(),
            NonZeroUsize::new(language_rows).unwrap(),
            max_segment,
            H3QwenVisionLinearGeometry::from_patch_rows(image_rows, video_rows).unwrap(),
            grids,
        )
        .unwrap()
    }

    #[test]
    fn device_retention_roundtrips_without_changing_numerics_or_legacy_defaults() {
        let baseline = sample_policy();
        let bytes = baseline.canonical_json().unwrap();
        assert!(!String::from_utf8_lossy(&bytes).contains("device_cache"));
        assert_eq!(
            ExecutionPolicy::from_json(&bytes)
                .unwrap()
                .weights
                .device_cache,
            DeviceCachePolicy::DISABLED
        );
        let mut retained = baseline.clone();
        retained.weights.device_cache = DeviceCachePolicy::with_max_bytes(4096)
            .with_cuda_allocator(ff_core::weights::CudaWeightAllocator::Direct);
        assert_eq!(retained.first_numerical_difference(&baseline), None);
        assert_eq!(
            retained.first_difference(&baseline),
            Some("weights.device_cache")
        );
        assert_eq!(
            ExecutionPolicy::from_json(&retained.canonical_json().unwrap()).unwrap(),
            retained
        );
    }

    #[test]
    fn host_priority_is_explicit_bounded_and_numerically_compatible() {
        let baseline = sample_policy();
        assert!(
            !String::from_utf8(baseline.canonical_json().unwrap())
                .unwrap()
                .contains("host_phase_priority")
        );
        let mut prioritized = baseline.clone();
        prioritized.weights.host_phase_priority = true;
        assert!(prioritized.validate().is_err());
        prioritized.weights.granularity = CacheGranularity::Tensor;
        prioritized.validate().unwrap();
        assert_eq!(prioritized.first_numerical_difference(&baseline), None);
        assert_eq!(
            ExecutionPolicy::from_json(&prioritized.canonical_json().unwrap()).unwrap(),
            prioritized
        );
        let mut unprioritized = prioritized.clone();
        unprioritized.weights.host_phase_priority = false;
        assert_eq!(
            unprioritized.first_difference(&prioritized),
            Some("weights.host_phase_priority")
        );
        prioritized.weights.cache_bytes = None;
        assert!(prioritized.validate().is_err());
    }

    #[test]
    fn canonical_json_and_hash_are_stable() {
        let policy = sample_policy();
        let json = String::from_utf8(policy.canonical_json().unwrap()).unwrap();
        assert_eq!(
            json,
            "{\"schema_version\":4,\"execution_backend\":\"cpu\",\"numerics\":{\"schema_version\":6,\"tensor_backend\":\"candle_cpu011_v1\",\"cuda_capabilities\":null,\"transformer_rms_norm\":\"candle_f32_fused_weight_single_cast_v1\",\"transformer_rotary\":\"host_f32_pow_reciprocal_device_f32_position_math_trig_input_dtype_rotation_v1\",\"biased_linear\":\"candle_linear_v1\",\"output_head_order\":\"both_heads_on_contiguous_full_packed_chunks_then_modality_select_v1\",\"attention\":\"candle_full_softmax_v1\",\"cuda_artifacts\":null},\"attention\":{\"backend\":\"full\",\"configured_projection_rows\":32,\"configured_query_rows\":32,\"configured_key_rows\":null},\"configured_ffn_rows\":256,\"configured_output_rows\":256,\"weights\":{\"source\":\"mmap\",\"cache_shards\":2,\"cache_bytes\":67108864},\"precompute_adaln\":true}"
        );
        let pretty = serde_json::to_vec_pretty(&policy).unwrap();
        assert_eq!(ExecutionPolicy::from_json(&pretty).unwrap(), policy);
    }

    #[test]
    fn tensor_granularity_changes_policy_identity_and_survives_replay() {
        let shard = sample_policy();
        let mut tensor = shard.clone();
        tensor.weights.granularity = CacheGranularity::Tensor;
        assert_eq!(shard.first_difference(&tensor), Some("weights.granularity"));
        assert_ne!(
            shard.canonical_json().unwrap(),
            tensor.canonical_json().unwrap()
        );
        let restored = ExecutionPolicy::from_json(&tensor.canonical_json().unwrap()).unwrap();
        assert_eq!(
            restored.cache_policy().unwrap().granularity,
            CacheGranularity::Tensor
        );
        assert_eq!(restored, tensor);
        for sources in [1, usize::MAX] {
            let actual = CachePolicy::new(sources)
                .with_max_bytes(32 * 1024 * 1024)
                .with_granularity(CacheGranularity::Tensor);
            let mut recorded = tensor.clone();
            recorded.weights.cache_shards = recorded_cache_shards(actual);
            recorded.weights.cache_bytes = actual.max_bytes;
            assert_eq!(recorded.cache_policy().unwrap(), actual);
        }
        assert!(
            !String::from_utf8(shard.canonical_json().unwrap())
                .unwrap()
                .contains("granularity")
        );
    }

    #[test]
    fn a_policy_without_a_numerical_contract_is_refused() {
        // Schema 1 recorded a run whose arithmetic was unbound. It was readable
        // for audit and never executable; it is now not a form this build
        // accepts at all, and a schema-4 record must carry its contract.
        const LEGACY: &str = "{\"schema_version\":1,\"execution_backend\":\"cpu\",\"attention\":{\"backend\":\"full\",\"configured_projection_rows\":32,\"configured_query_rows\":32,\"configured_key_rows\":null},\"configured_ffn_rows\":256,\"configured_output_rows\":256,\"weights\":{\"source\":\"mmap\",\"cache_shards\":2,\"cache_bytes\":67108864},\"precompute_adaln\":true}";
        let error = ExecutionPolicy::from_json(LEGACY.as_bytes()).unwrap_err();
        assert!(
            format!("{error:#}").contains("unsupported execution-policy schema 1"),
            "unexpected error: {error:#}"
        );

        let mut missing = serde_json::to_value(sample_policy()).unwrap();
        missing.as_object_mut().unwrap().remove("numerics");
        let error = ExecutionPolicy::from_json(&serde_json::to_vec(&missing).unwrap()).unwrap_err();
        assert!(
            format!("{error:#}").contains("requires numerics"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn runtime_policy_rejects_conflicting_attention_algorithms() {
        let mut chunking = TransformerChunking::default();
        chunking.attention.key = AttentionKeyChunkPolicy::chunked(16).unwrap();
        let error = ExecutionPolicy::from_runtime(
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::new(1),
            chunking,
            true,
            true,
        )
        .unwrap_err();
        assert!(error.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn rejects_unknown_schema_partial_metadata_and_tampering() {
        let mut old_rotary = serde_json::to_value(sample_policy()).unwrap();
        old_rotary["numerics"]["schema_version"] = serde_json::json!(2);
        let error =
            ExecutionPolicy::from_json(&serde_json::to_vec(&old_rotary).unwrap()).unwrap_err();
        assert!(format!("{error:#}").contains("unsupported H3 numerical-contract schema 2"));
        let mut missing_rotary = serde_json::to_value(sample_policy()).unwrap();
        missing_rotary["numerics"]
            .as_object_mut()
            .unwrap()
            .remove("transformer_rotary");
        let error =
            ExecutionPolicy::from_json(&serde_json::to_vec(&missing_rotary).unwrap()).unwrap_err();
        assert!(format!("{error:#}").contains("transformer_rotary"));
        let unknown = br#"{"schema_version":5}"#;
        assert!(
            ExecutionPolicy::from_json(unknown)
                .unwrap_err()
                .to_string()
                .contains("unsupported execution-policy schema 5")
        );
        let mut missing_numerics = serde_json::to_value(sample_policy()).unwrap();
        missing_numerics.as_object_mut().unwrap().remove("numerics");
        let error = ExecutionPolicy::from_json(&serde_json::to_vec(&missing_numerics).unwrap())
            .unwrap_err();
        assert!(format!("{error:#}").contains("schema 4 requires numerics"));

        // Schema 1 is not a form this build reads, with or without numerics.
        let mut schema_one = serde_json::to_value(sample_policy()).unwrap();
        schema_one["schema_version"] = serde_json::json!(1);
        for numerics in [None, Some(serde_json::Value::Null)] {
            if let Some(value) = numerics {
                schema_one["numerics"] = value;
            }
            let error =
                ExecutionPolicy::from_json(&serde_json::to_vec(&schema_one).unwrap()).unwrap_err();
            assert!(
                format!("{error:#}").contains("unsupported execution-policy schema 1"),
                "unexpected error: {error:#}"
            );
        }

        let mut null_numerics = serde_json::to_value(sample_policy()).unwrap();
        null_numerics["numerics"] = serde_json::Value::Null;
        let error =
            ExecutionPolicy::from_json(&serde_json::to_vec(&null_numerics).unwrap()).unwrap_err();
        assert!(format!("{error:#}").contains("requires non-null numerics"));
        let duplicate =
            ExecutionPolicy::from_json(br#"{"schema_version":2,"schema_version":2}"#).unwrap_err();
        assert!(format!("{duplicate:#}").contains("duplicate field"));
        let unknown_numerics = sample_policy()
            .canonical_json()
            .and_then(|bytes| {
                let mut value: serde_json::Value = serde_json::from_slice(&bytes)?;
                value["numerics"]["transformer_rms_norm"] =
                    serde_json::json!("unregistered_backend");
                ExecutionPolicy::from_json(&serde_json::to_vec(&value)?)?;
                Ok(())
            })
            .unwrap_err();
        assert!(format!("{unknown_numerics:#}").contains("unknown variant"));

        let mut unknown_output_order = serde_json::to_value(sample_policy()).unwrap();
        unknown_output_order["numerics"]["output_head_order"] =
            serde_json::json!("select_modality_before_heads");
        let error = ExecutionPolicy::from_json(&serde_json::to_vec(&unknown_output_order).unwrap())
            .unwrap_err();
        assert!(format!("{error:#}").contains("unknown variant"));
    }

    #[test]
    fn validates_backend_and_key_policy_as_one_choice() {
        let mut policy = sample_policy();
        policy.attention.backend = AttentionBackendPolicy::OnlineSoftmax;
        assert!(policy.validate().is_err());
        policy.attention.configured_key_rows = Some(1024);
        policy.rebind_attention_numerics().unwrap();
        policy.validate().unwrap();
        assert_eq!(
            policy.transformer_chunking().unwrap().attention.key,
            AttentionKeyChunkPolicy::chunked(1024).unwrap()
        );
        policy.attention.backend = AttentionBackendPolicy::FlashAttention;
        policy.attention.configured_key_rows = None;
        assert!(
            policy
                .rebind_attention_numerics()
                .unwrap_err()
                .to_string()
                .contains("FlashAttention requires CUDA")
        );
    }

    #[test]
    fn effective_chunks_are_derived_without_changing_the_recorded_policy() {
        let policy = sample_policy();
        let effective = policy
            .effective_chunks(
                NonZeroUsize::new(10).unwrap(),
                NonZeroUsize::new(3).unwrap(),
            )
            .unwrap();
        assert_eq!(effective.attention.projection_rows.get(), 10);
        assert_eq!(effective.attention.query_rows.get(), 10);
        assert_eq!(effective.attention.key_rows.get(), 10);
        assert_eq!(effective.feed_forward_rows.get(), 10);
        assert_eq!(effective.output_rows.get(), 3);
        assert_eq!(policy.attention.configured_query_rows, 32);
    }

    #[test]
    fn official_output_head_gate_requires_one_full_packed_gemm_chunk() {
        let policy = sample_policy();
        policy
            .validate_official_full_packed_output_heads(NonZeroUsize::new(256).unwrap())
            .unwrap();
        let error = policy
            .validate_official_full_packed_output_heads(NonZeroUsize::new(257).unwrap())
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("configured_output_rows >= packed rows")
        );
    }

    #[test]
    fn policy_difference_names_the_first_changed_field() {
        let policy = sample_policy();
        assert_eq!(policy.first_difference(&policy), None);
        let mut changed = policy.clone();
        changed.attention.configured_query_rows += 1;
        assert_eq!(
            policy.first_difference(&changed),
            Some("attention.configured_query_rows")
        );
        changed = policy.clone();
        changed.numerics.schema_version += 1;
        assert_eq!(
            policy.first_difference(&changed),
            Some("numerics.schema_version")
        );
    }

    #[test]
    fn every_recorded_numerical_field_is_part_of_the_record() {
        let policy = sample_policy();
        let original = serde_json::to_vec(&policy).unwrap();
        let mut mutations = Vec::new();

        let mut changed = policy.clone();
        changed.numerics.schema_version += 1;
        mutations.push(changed);

        let mut changed = policy.clone();
        changed.numerics.tensor_backend = H3TensorBackendContract::CandleMetal011V1;
        mutations.push(changed);

        let mut changed = policy.clone();
        changed.numerics.transformer_rms_norm =
            H3TransformerRmsNormContract::Pytorch7269437VectorizedBf16FusedWeightSingleCastCudaV1;
        mutations.push(changed);

        let mut changed = policy.clone();
        changed.numerics.biased_linear =
            H3BiasedLinearContract::CublasLt130401Sm89Sm128Driver59584Bf16Compute32BiasEpilogueV1;
        mutations.push(changed);

        let mut changed = policy.clone();
        changed.numerics.attention = H3AttentionContract::CandleOnlineSoftmaxV1;
        mutations.push(changed);

        for changed in mutations {
            assert_ne!(serde_json::to_vec(&changed).unwrap(), original);
            assert!(changed.first_difference(&policy).is_some());
            assert!(changed.validate().is_err());
        }

        let mut changed = policy.clone();
        changed.numerics.collective = FOUR_RANK_COLLECTIVE;
        assert_ne!(serde_json::to_vec(&changed).unwrap(), original);
        assert_eq!(
            changed.first_difference(&policy),
            Some("numerics.collective")
        );
        changed.validate().unwrap();
    }

    #[test]
    fn cuda_attention_contracts_are_closed_mode_specific_and_hashed() {
        let full = cuda_policy(AttentionBackendPolicy::FullSoftmax);
        let online = cuda_policy(AttentionBackendPolicy::OnlineSoftmax);
        let flash = cuda_policy(AttentionBackendPolicy::FlashAttention);
        assert!(matches!(
            full.numerics
                .cuda_artifacts
                .as_ref()
                .unwrap()
                .attention
                .as_ref()
                .unwrap(),
            H3CudaAttentionArtifacts::PytorchNativeMathPersistentSoftmax
        ));
        assert!(matches!(
            online
                .numerics
                .cuda_artifacts
                .as_ref()
                .unwrap()
                .attention
                .as_ref()
                .unwrap(),
            H3CudaAttentionArtifacts::OnlineSoftmaxWithPersistentTokenRefiner
        ));
        assert!(matches!(
            flash
                .numerics
                .cuda_artifacts
                .as_ref()
                .unwrap()
                .attention
                .as_ref()
                .unwrap(),
            H3CudaAttentionArtifacts::CandleFlashAttention011
        ));
        // Three backends, three distinct contracts: a resume onto the wrong
        // one is refused by the field the contracts disagree on.
        assert_eq!(full.first_difference(&online), Some("numerics.attention"));
        assert_eq!(full.first_difference(&flash), Some("numerics.attention"));
        assert_eq!(online.first_difference(&flash), Some("numerics.attention"));
    }

    #[test]
    fn flash_contract_is_independent_of_reference_libraries_and_tracks_its_artifacts() {
        for tuned_kernels in [false, true] {
            let caps = CudaCapabilities {
                tuned_kernels,
                reference_libraries: false,
            };
            let contract = H3NumericalContract::for_target(
                ExecutionBackendPolicy::Cuda,
                Some(caps),
                AttentionBackendPolicy::FlashAttention,
            )
            .unwrap();
            assert_eq!(
                contract.biased_linear,
                H3BiasedLinearContract::CandleLinearV1
            );
            assert_eq!(
                contract.attention,
                H3AttentionContract::FlashAttention011MainAndTokenRefinerCudaV1
            );
            let artifacts = contract.cuda_artifacts.as_ref().unwrap();
            assert!(artifacts.reference_libraries.is_none());
            assert_eq!(artifacts.tuned_kernels.is_some(), tuned_kernels);
            assert!(matches!(
                artifacts.attention,
                Some(H3CudaAttentionArtifacts::CandleFlashAttention011)
            ));
            let mut changed = contract.clone();
            changed.cuda_artifacts.as_mut().unwrap().attention = None;
            assert!(
                changed
                    .validate_for(
                        ExecutionBackendPolicy::Cuda,
                        Some(caps),
                        AttentionBackendPolicy::FlashAttention
                    )
                    .is_err()
            );
            let mut old = contract.clone();
            old.schema_version = 4;
            assert!(
                old.validate_for(
                    ExecutionBackendPolicy::Cuda,
                    Some(caps),
                    AttentionBackendPolicy::FlashAttention
                )
                .is_err()
            );
            let full = H3NumericalContract::for_target(
                ExecutionBackendPolicy::Cuda,
                Some(caps),
                AttentionBackendPolicy::FullSoftmax,
            )
            .unwrap();
            assert!(full.cuda_artifacts.as_ref().unwrap().attention.is_none());
        }
    }

    #[test]
    fn conditioning_contract_is_canonical_bounded_and_geometry_bound() {
        let contract = conditioning_contract(ExecutionBackendPolicy::Cpu, 256, 357, 0);
        assert_eq!(contract.configured_query_rows, 256);
        assert_eq!(contract.language_rows, 357);
        assert_eq!(contract.effective_language_query_rows, 256);
        assert_eq!(contract.max_vision_segment_rows, 0);
        assert_eq!(contract.effective_vision_query_rows, 0);
        assert_eq!(contract.max_attention_width, 357);
        assert!(contract.cuda_artifacts.is_none());
        let canonical = contract.canonical_json().unwrap();
        assert_eq!(
            String::from_utf8(canonical.clone()).unwrap(),
            "{\"schema_version\":2,\"execution_backend\":\"cpu\",\"cuda_capabilities\":null,\"tensor_backend\":\"candle_cpu011_v1\",\"rms_norm\":\"f32_normalize_then_input_dtype_cast_and_input_dtype_weight_multiply_v1\",\"rotary\":\"pinned_cpu_f32_inv_freq_bits_device_f32_position_math_cos_sin_cast_v1\",\"attention_scores\":\"eager_input_dtype_matmul_then_scale_v1\",\"attention_mask\":\"eager_causal_add_finite_input_dtype_minimum_v1\",\"attention_matmul\":\"candle_chunked_qk_and_pv_v1\",\"softmax\":\"candle_f32_composite_v1\",\"silu\":\"f32_promote_then_input_dtype_cast_v1\",\"vision_block_gelu\":\"not_applicable_no_vision\",\"vision_merger_gelu\":\"not_applicable_no_vision\",\"biased_linear\":\"candle_linear_v1\",\"vision_patch_projection\":\"not_applicable_no_vision\",\"vision_layer_norm\":\"not_applicable_no_vision\",\"vision_position_interpolation\":\"not_applicable_no_vision\",\"vision_grid_encoding\":\"modality_u8_count_u32_thw_u64_little_endian_v1\",\"ordered_vision_grids\":[],\"configured_query_rows\":256,\"language_rows\":357,\"effective_language_query_rows\":256,\"max_vision_segment_rows\":0,\"effective_vision_query_rows\":0,\"max_attention_width\":357,\"vision_linear_geometry\":{\"image_total_patch_rows\":0,\"image_merger_rows\":0,\"video_total_patch_rows\":0,\"video_merger_rows\":0},\"cuda_artifacts\":null}"
        );
        assert_eq!(
            H3QwenNumericalContract::from_json(&canonical).unwrap(),
            contract
        );
        let pretty = serde_json::to_vec_pretty(&contract).unwrap();
        assert_eq!(
            H3QwenNumericalContract::from_json(&pretty).unwrap(),
            contract
        );

        let mut tampered = contract.clone();
        tampered.effective_language_query_rows -= 1;
        assert!(tampered.validate().is_err());
        let mut tampered = contract.clone();
        tampered.effective_vision_query_rows = 1;
        assert!(tampered.validate().is_err());
        for field in [
            "rms_norm",
            "rotary",
            "attention_scores",
            "attention_mask",
            "attention_matmul",
            "softmax",
            "silu",
            "vision_block_gelu",
            "vision_merger_gelu",
            "biased_linear",
            "vision_patch_projection",
            "vision_layer_norm",
            "vision_position_interpolation",
        ] {
            let mut unknown = serde_json::to_value(&contract).unwrap();
            unknown[field] = serde_json::json!("unknown_numerical_backend");
            let encoded = serde_json::to_vec(&unknown).unwrap();
            assert_ne!(encoded, canonical);
            assert!(H3QwenNumericalContract::from_json(&encoded).is_err());
        }
        let mut unknown = serde_json::to_value(&contract).unwrap();
        unknown["unexpected"] = serde_json::json!(true);
        assert!(
            H3QwenNumericalContract::from_json(&serde_json::to_vec(&unknown).unwrap()).is_err()
        );
        assert!(
            H3QwenNumericalContract::from_json(&vec![
                b' ';
                MAX_H3_QWEN_NUMERICAL_CONTRACT_JSON_BYTES + 1
            ])
            .is_err()
        );
    }

    #[test]
    fn qwen_contract_validation_follows_each_capability_axis() {
        for tuned_kernels in [false, true] {
            for reference_libraries in [false, true] {
                for vision in [false, true] {
                    let grids = if vision {
                        vec![(H3QwenVisionGridModality::Image, 1, 48, 84)]
                    } else {
                        vec![]
                    };
                    let rows = if vision { 4032 } else { 0 };
                    let contract = H3QwenNumericalContract::for_target_with_grids(
                        ExecutionBackendPolicy::Cuda,
                        Some(CudaCapabilities {
                            tuned_kernels,
                            reference_libraries,
                        }),
                        NonZeroUsize::new(256).unwrap(),
                        NonZeroUsize::new(if vision { 1935 } else { 357 }).unwrap(),
                        rows,
                        H3QwenVisionLinearGeometry::from_patch_rows(rows, 0).unwrap(),
                        &grids,
                    )
                    .unwrap();
                    contract.validate().unwrap();
                    let restored =
                        H3QwenNumericalContract::from_json(&contract.canonical_json().unwrap())
                            .unwrap();
                    assert_eq!(restored, contract);
                    assert_eq!(contract.cuda_artifacts.is_some(), reference_libraries);
                    let mut changed = contract.clone();
                    changed
                        .cuda_capabilities
                        .as_mut()
                        .unwrap()
                        .reference_libraries = !reference_libraries;
                    assert!(changed.validate().is_err());
                    let mut changed = contract;
                    changed.cuda_capabilities.as_mut().unwrap().tuned_kernels = !tuned_kernels;
                    assert!(changed.validate().is_err());
                }
            }
        }
    }

    #[test]
    fn qwen_contract_artifact_tensors_are_canonical_and_complete() {
        let contract = conditioning_contract(ExecutionBackendPolicy::Cpu, 256, 357, 0);
        let mut encoded = HashMap::new();
        contract
            .insert_artifact_tensors(&mut encoded, &Device::Cpu)
            .unwrap();
        // The contract and its schema; the digest tensor that used to sit
        // beside them described the JSON already there.
        assert_eq!(encoded.len(), 2);
        let mut loaded = encoded
            .iter()
            .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
            .collect::<HashMap<_, _>>();
        assert_eq!(
            H3QwenNumericalContract::take_artifact_tensors(&mut loaded).unwrap(),
            Some(contract.clone())
        );
        assert!(loaded.is_empty());

        let mut incomplete = HashMap::from([(
            H3_QWEN_NUMERICAL_CONTRACT_JSON_TENSOR.to_owned(),
            encoded[H3_QWEN_NUMERICAL_CONTRACT_JSON_TENSOR].clone(),
        )]);
        assert!(
            H3QwenNumericalContract::take_artifact_tensors(&mut incomplete)
                .unwrap_err()
                .to_string()
                .contains("incomplete")
        );

        let mut corrupt = encoded
            .iter()
            .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
            .collect::<HashMap<_, _>>();
        corrupt.insert(
            H3_QWEN_NUMERICAL_CONTRACT_SCHEMA_TENSOR.to_owned(),
            Tensor::zeros(32, DType::U8, &Device::Cpu).unwrap(),
        );
        assert!(
            H3QwenNumericalContract::take_artifact_tensors(&mut corrupt)
                .unwrap_err()
                .to_string()
                .contains("schema tensor must be a U32 scalar")
        );

        let pretty = serde_json::to_vec_pretty(&contract).unwrap();
        let mut noncanonical = encoded
            .iter()
            .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
            .collect::<HashMap<_, _>>();
        noncanonical.insert(
            H3_QWEN_NUMERICAL_CONTRACT_JSON_TENSOR.to_owned(),
            Tensor::from_vec(pretty.clone(), pretty.len(), &Device::Cpu).unwrap(),
        );
        assert!(
            H3QwenNumericalContract::take_artifact_tensors(&mut noncanonical)
                .unwrap_err()
                .to_string()
                .contains("not canonical")
        );
    }

    #[test]
    fn legacy_cuda_code_annotations_are_preserved_but_not_gates() {
        fn annotate(value: &mut serde_json::Value) {
            match value {
                serde_json::Value::Object(fields) => {
                    for (name, value) in fields {
                        if name.ends_with("_sha256") {
                            *value = "legacy annotation, not a digest".into();
                        } else {
                            annotate(value);
                        }
                    }
                }
                serde_json::Value::Array(values) => values.iter_mut().for_each(annotate),
                _ => {}
            }
        }
        let h3 = H3NumericalContract::for_verified_target(
            ExecutionBackendPolicy::Cuda,
            AttentionBackendPolicy::FullSoftmax,
        )
        .unwrap();
        let mut legacy = serde_json::to_value(&h3).unwrap();
        annotate(&mut legacy["cuda_artifacts"]);
        let loaded: H3NumericalContract = serde_json::from_value(legacy.clone()).unwrap();
        loaded
            .validate_for(
                ExecutionBackendPolicy::Cuda,
                Some(CudaCapabilities::REFERENCE),
                AttentionBackendPolicy::FullSoftmax,
            )
            .unwrap();
        assert_eq!(loaded, h3);
        assert_eq!(serde_json::to_value(&loaded).unwrap(), legacy);

        let qwen = vision_conditioning_contract(
            ExecutionBackendPolicy::Cuda,
            256,
            1935,
            &[(H3QwenVisionGridModality::Image, 1, 48, 84)],
        );
        let mut legacy = serde_json::to_value(&qwen).unwrap();
        annotate(&mut legacy["cuda_artifacts"]);
        let loaded: H3QwenNumericalContract = serde_json::from_value(legacy.clone()).unwrap();
        loaded.validate().unwrap();
        assert_eq!(loaded.first_difference(&qwen), None);
        assert_eq!(loaded, qwen);
        assert_eq!(serde_json::to_value(&loaded).unwrap(), legacy);
        let mut tensors = HashMap::new();
        loaded
            .insert_artifact_tensors(&mut tensors, &Device::Cpu)
            .unwrap();
        let mut tensors = tensors
            .into_iter()
            .map(|(name, tensor)| (name.to_owned(), tensor))
            .collect();
        assert_eq!(
            H3QwenNumericalContract::take_artifact_tensors(&mut tensors).unwrap(),
            Some(loaded)
        );
    }

    #[test]
    fn conditioning_first_difference_and_hash_cover_geometry_and_cuda_artifacts() {
        let grids = [(H3QwenVisionGridModality::Image, 1, 48, 84)];
        let baseline =
            vision_conditioning_contract(ExecutionBackendPolicy::Cuda, 256, 1935, &grids);
        assert_eq!(baseline.effective_language_query_rows, 256);
        assert_eq!(baseline.effective_vision_query_rows, 256);
        let changed_geometry =
            vision_conditioning_contract(ExecutionBackendPolicy::Cuda, 128, 1935, &grids);
        assert_eq!(
            baseline.first_difference(&changed_geometry),
            Some("qwen.configured_query_rows")
        );

        let mut changed_precision = baseline.clone();
        changed_precision
            .cuda_artifacts
            .as_mut()
            .unwrap()
            .tensor_runtime
            .candle_gemm_reduced_precision_bf16 = true;
        assert!(changed_precision.validate().is_err());
        assert_eq!(
            baseline.first_difference(&changed_precision),
            Some("qwen.cuda_artifacts.tensor_runtime.candle_gemm_reduced_precision_bf16")
        );

        let oversized = H3QwenNumericalContract::for_verified_target(
            ExecutionBackendPolicy::Cuda,
            NonZeroUsize::new(256).unwrap(),
            NonZeroUsize::new(9217).unwrap(),
            0,
            H3QwenVisionLinearGeometry::from_patch_rows(0, 0).unwrap(),
        )
        .unwrap_err();
        assert!(oversized.to_string().contains("1..=9216"));
    }

    #[test]
    fn qwen_vision_linear_geometry_binds_real_fl_and_ref_gemm_rows() {
        let geometry = H3QwenVisionLinearGeometry::from_patch_rows(4032, 28224).unwrap();
        assert_eq!(geometry.image_merger_rows, 1008);
        assert_eq!(geometry.video_merger_rows, 7056);
        let patch_contract = H3QwenNumericalContract::for_verified_target_with_grids(
            ExecutionBackendPolicy::Cuda,
            NonZeroUsize::new(256).unwrap(),
            NonZeroUsize::new(8620).unwrap(),
            4032,
            geometry,
            &[
                (H3QwenVisionGridModality::Image, 1, 48, 84),
                (H3QwenVisionGridModality::Video, 7, 48, 84),
            ],
        )
        .unwrap();
        assert_eq!(
            patch_contract.vision_patch_projection,
            H3QwenVisionPatchProjectionContract::Cudnn92101LegacyImplicitPrecompGemmTensorOpMathEmpiricalV8ParityV1
        );
        let patch = patch_contract
            .cuda_artifacts
            .as_ref()
            .unwrap()
            .patch
            .as_ref()
            .unwrap();
        assert_eq!(patch.fl_workspace_bytes, 42_467_344);
        assert_eq!(patch.ref_workspace_bytes, 240_648_208);
        assert_eq!(
            patch_contract.vision_block_gelu,
            H3QwenVisionBlockGeluContract::Pytorch7269437Nvrtc130Bf16TanhRealFlRefV1
        );
        assert_eq!(
            patch_contract.vision_merger_gelu,
            H3QwenVisionMergerGeluContract::Pytorch7269437Nvrtc130Bf16ErfRealFlRefV1
        );
        let mut changed_patch = patch_contract.clone();
        changed_patch
            .cuda_artifacts
            .as_mut()
            .unwrap()
            .patch
            .as_mut()
            .unwrap()
            .fl_workspace_bytes += 1;
        changed_patch.validate().unwrap();
        assert_eq!(
            patch_contract.first_difference(&changed_patch),
            Some("qwen.cuda_artifacts.patch.fl_workspace_bytes")
        );
        assert!(H3QwenVisionLinearGeometry::from_patch_rows(4031, 0).is_err());
        let unsupported = H3QwenNumericalContract::for_verified_target_with_grids(
            ExecutionBackendPolicy::Cuda,
            NonZeroUsize::new(256).unwrap(),
            NonZeroUsize::new(1935).unwrap(),
            4032,
            H3QwenVisionLinearGeometry::from_patch_rows(8064, 0).unwrap(),
            &[(H3QwenVisionGridModality::Image, 2, 48, 84)],
        )
        .unwrap_err();
        assert!(
            unsupported
                .to_string()
                .contains("verified FL/Ref shape profile")
        );
    }

    #[test]
    fn ordered_vision_grids_match_oracle_codec_and_bind_every_derived_shape() {
        let fl = vision_conditioning_contract(
            ExecutionBackendPolicy::Cuda,
            256,
            1935,
            &[(H3QwenVisionGridModality::Image, 1, 48, 84)],
        );
        assert_eq!(fl.ordered_vision_grids.len(), 1);
        let grid = &fl.ordered_vision_grids[0];
        assert_eq!(
            (
                grid.ordinal,
                grid.temporal,
                grid.height,
                grid.width,
                grid.patch_rows,
                grid.merger_rows,
                grid.attention_segment_rows,
                grid.attention_segment_count,
                grid.query_full_chunks,
                grid.query_tail_rows,
            ),
            (0, 1, 48, 84, 4032, 1008, 4032, 1, 15, 192)
        );

        let reference = vision_conditioning_contract(
            ExecutionBackendPolicy::Cuda,
            256,
            8620,
            &[(H3QwenVisionGridModality::Video, 7, 48, 84)],
        );
        assert_eq!(
            (
                reference.ordered_vision_grids[0].attention_segment_count,
                reference.ordered_vision_grids[0].query_full_chunks,
                reference.ordered_vision_grids[0].query_tail_rows,
            ),
            (7, 15, 192)
        );

        let same_shape_video = vision_conditioning_contract(
            ExecutionBackendPolicy::Cuda,
            256,
            1935,
            &[(H3QwenVisionGridModality::Video, 1, 48, 84)],
        );
        assert_eq!(
            same_shape_video.vision_position_interpolation,
            H3QwenVisionPositionInterpolationContract::HostF32BilinearTapsThenCandleGatherWeightedSumV1
        );
        let swapped_image = vision_conditioning_contract(
            ExecutionBackendPolicy::Cuda,
            256,
            1935,
            &[(H3QwenVisionGridModality::Image, 7, 48, 84)],
        );
        assert_eq!(
            swapped_image.vision_position_interpolation,
            H3QwenVisionPositionInterpolationContract::HostF32BilinearTapsThenCandleGatherWeightedSumV1
        );
        assert_eq!(
            fl.first_difference(&same_shape_video),
            Some("qwen.ordered_vision_grids.modality")
        );
        let mut bad_tail = fl.clone();
        bad_tail.ordered_vision_grids[0].query_tail_rows += 1;
        assert!(bad_tail.validate().is_err());
        // The count bound the canonical encoder used to carry now lives in
        // `validate`, so it still refuses a grid list no verified run produces.
        let mut too_many = fl.clone();
        let grid = too_many.ordered_vision_grids[0].clone();
        too_many.ordered_vision_grids = (0..=MAX_QWEN_ORDERED_VISION_GRIDS)
            .map(|ordinal| {
                let mut grid = grid.clone();
                grid.ordinal = ordinal as u64;
                grid
            })
            .collect();
        let error = too_many.validate().unwrap_err().to_string();
        assert!(
            error.contains("ordered vision-grid count exceeds"),
            "{error}"
        );
        assert!(
            H3QwenNumericalContract::for_verified_target(
                ExecutionBackendPolicy::Cuda,
                NonZeroUsize::new(256).unwrap(),
                NonZeroUsize::new(1935).unwrap(),
                4032,
                H3QwenVisionLinearGeometry::from_patch_rows(4032, 0).unwrap(),
            )
            .unwrap_err()
            .to_string()
            .contains("totals alone")
        );
    }

    #[test]
    fn qwen_contract_closes_cpu_and_only_the_exact_cuda_text357_profile() {
        let contract = conditioning_contract(ExecutionBackendPolicy::Cpu, 256, 357, 0);
        contract
            .validate_for(
                &Device::Cpu,
                NonZeroUsize::new(256).unwrap(),
                NonZeroUsize::new(357).unwrap(),
                0,
                H3QwenVisionLinearGeometry::from_patch_rows(0, 0).unwrap(),
            )
            .unwrap();
        assert_eq!(
            contract.attention_matmul,
            H3QwenAttentionMatmulContract::CandleChunkedQkAndPvV1
        );
        let error = contract
            .validate_for(
                &Device::Cpu,
                NonZeroUsize::new(128).unwrap(),
                NonZeroUsize::new(357).unwrap(),
                0,
                H3QwenVisionLinearGeometry::from_patch_rows(0, 0).unwrap(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("qwen.configured_query_rows"));

        let exact_cuda = conditioning_contract(ExecutionBackendPolicy::Cuda, 357, 357, 0);
        assert_eq!(
            exact_cuda.attention_matmul,
            H3QwenAttentionMatmulContract::CublasLt130401PytorchScaleText357FullQueryV1
        );
        let wrong_query = conditioning_contract(ExecutionBackendPolicy::Cuda, 256, 357, 0);
        assert_eq!(
            wrong_query.attention_matmul,
            H3QwenAttentionMatmulContract::CandleChunkedQkAndPvCudaV1
        );
        let wrong_rows = conditioning_contract(ExecutionBackendPolicy::Cuda, 357, 356, 0);
        assert_eq!(
            wrong_rows.attention_matmul,
            H3QwenAttentionMatmulContract::CandleChunkedQkAndPvCudaV1
        );
        assert!(
            wrong_rows.first_difference(&exact_cuda).is_some(),
            "the portable path must record differently from the pinned one"
        );
        let fl = vision_conditioning_contract(
            ExecutionBackendPolicy::Cuda,
            256,
            1_935,
            &[(H3QwenVisionGridModality::Image, 1, 48, 84)],
        );
        assert_eq!(
            fl.attention_matmul,
            H3QwenAttentionMatmulContract::CublasLt130401PytorchScaleFl1935Query256Tail143V1
        );
        let reference = vision_conditioning_contract(
            ExecutionBackendPolicy::Cuda,
            256,
            8_620,
            &[(H3QwenVisionGridModality::Video, 7, 48, 84)],
        );
        assert_eq!(
            reference.attention_matmul,
            H3QwenAttentionMatmulContract::CublasLt130401PytorchScaleRef8620Query256Tail172OperatorReferenceV1
        );
        for (rows, query_rows) in [(8_619, 256), (8_620, 128)] {
            let unsupported = vision_conditioning_contract(
                ExecutionBackendPolicy::Cuda,
                query_rows,
                rows,
                &[(H3QwenVisionGridModality::Video, 7, 48, 84)],
            );
            assert_eq!(
                unsupported.attention_matmul,
                H3QwenAttentionMatmulContract::CandleChunkedQkAndPvCudaV1
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn conditioning_cuda_contract_matches_both_softmax_kernel_ranges() {
        let contract = vision_conditioning_contract(
            ExecutionBackendPolicy::Cuda,
            256,
            8620,
            &[(H3QwenVisionGridModality::Video, 7, 48, 84)],
        );
        let artifacts = contract.cuda_artifacts.as_ref().unwrap();
        assert_eq!(
            (
                artifacts.persistent_min_width,
                artifacts.persistent_max_width
            ),
            (1, 2048)
        );
        assert_eq!(
            (artifacts.regular_min_width, artifacts.regular_max_width),
            (2049, 9216)
        );
        validate_compiled_qwen_contract(ExecutionBackendPolicy::Cuda, true).unwrap();
    }

    #[cfg(feature = "cuda")]
    #[cfg(all(feature = "cuda", not(feature = "flash-attn")))]
    #[test]
    fn flash_contract_requires_the_compiled_flash_implementation() {
        for capabilities in [CudaCapabilities::NONE, CudaCapabilities::REFERENCE] {
            let error = H3NumericalContract::for_target(
                ExecutionBackendPolicy::Cuda,
                Some(capabilities),
                AttentionBackendPolicy::FlashAttention,
            )
            .unwrap()
            .ensure_current_for_build(
                ExecutionBackendPolicy::Cuda,
                AttentionBackendPolicy::FlashAttention,
            )
            .unwrap_err();
            assert!(error.to_string().contains("flash-attn feature"));
        }
    }

    #[cfg(feature = "flash-attn")]
    #[test]
    fn flash_contract_accepts_compiled_main_refiner_backend() {
        H3NumericalContract::for_verified_target(
            ExecutionBackendPolicy::Cuda,
            AttentionBackendPolicy::FlashAttention,
        )
        .unwrap()
        .ensure_current_for_build(
            ExecutionBackendPolicy::Cuda,
            AttentionBackendPolicy::FlashAttention,
        )
        .unwrap();
    }

    #[test]
    fn a_one_shard_byte_ceiling_replaces_the_unit_bound_on_replay() {
        let recorded = ExecutionPolicy::from_runtime(
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::unbounded_units().with_max_bytes(64 * 1024 * 1024),
            TransformerChunking::default(),
            false,
            true,
        )
        .unwrap();
        assert_eq!(recorded.weights.cache_shards, 1);
        assert_eq!(recorded.weights.cache_bytes, Some(64 * 1024 * 1024));
        assert_eq!(
            recorded.cache_policy().unwrap(),
            CachePolicy::unbounded_units().with_max_bytes(64 * 1024 * 1024)
        );

        let one_plus_bytes = ExecutionPolicy::from_runtime(
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::new(1).with_max_bytes(8),
            TransformerChunking::default(),
            false,
            true,
        )
        .unwrap();
        assert_eq!(
            one_plus_bytes.cache_policy().unwrap(),
            CachePolicy::unbounded_units().with_max_bytes(8)
        );

        let complementary = sample_policy();
        assert_eq!(complementary.weights.cache_shards, 2);
        assert_eq!(
            complementary.cache_policy().unwrap(),
            CachePolicy::new(2).with_max_bytes(64 * 1024 * 1024)
        );

        let tighter = ExecutionPolicy::from_runtime(
            &Device::Cpu,
            WeightSource::Mmap,
            CachePolicy::unbounded_units().with_max_bytes(32 * 1024 * 1024),
            TransformerChunking::default(),
            false,
            true,
        )
        .unwrap();
        assert_eq!(
            recorded.first_difference(&tighter),
            Some("weights.cache_bytes")
        );
        assert_ne!(
            recorded.cache_policy().unwrap().max_bytes,
            tighter.cache_policy().unwrap().max_bytes
        );
    }

    /// A four-rank run resolves a different sum than
    /// a one-rank run does, so its recorded policy is a different policy: not
    /// equal, differently hashed, and named at `numerics.collective` rather
    /// than merely failing a hash check.
    const FOUR_RANK_COLLECTIVE: H3CollectiveContract = H3CollectiveContract {
        partition: H3CollectivePartitionContract::MegatronColumnRowV1,
        reduction_order: H3CollectiveReductionOrderContract::RingRankAscendingV1,
        rank_count: 4,
    };

    fn with_collective(
        policy: &ExecutionPolicy,
        collective: H3CollectiveContract,
    ) -> ExecutionPolicy {
        let mut changed = policy.clone();
        changed.numerics.collective = collective;
        changed
    }

    #[test]
    fn a_four_rank_policy_is_not_a_one_rank_policy() {
        let single = sample_policy();
        assert_eq!(single.collective(), H3CollectiveContract::SINGLE_RANK);
        let four = with_collective(&single, FOUR_RANK_COLLECTIVE);
        assert_ne!(single, four);

        assert_eq!(
            single.first_difference(&four),
            Some("numerics.collective"),
            "the rank count is part of the arithmetic identity, not a resource knob"
        );
        assert_eq!(
            single.first_numerical_difference(&four),
            Some("numerics.collective")
        );
        let two = with_collective(
            &single,
            H3CollectiveContract {
                rank_count: 2,
                ..FOUR_RANK_COLLECTIVE
            },
        );
        assert_eq!(two.first_difference(&four), Some("numerics.collective"));
    }

    /// The single-rank contract is omitted from the serialized form, so every
    /// policy recorded before this field existed keeps its bytes. This is what
    /// keeps the evidence base from splitting for runs that did not actually
    /// change.
    #[test]
    fn a_single_rank_contract_leaves_the_recorded_bytes_and_hash_alone() {
        let policy = sample_policy();
        let json = policy.canonical_json().unwrap();
        let text = String::from_utf8(json.clone()).unwrap();
        assert!(!text.contains("collective"), "{text}");
        let reloaded = ExecutionPolicy::from_json(&json).unwrap();
        assert_eq!(reloaded, policy);
        assert_eq!(reloaded.collective(), H3CollectiveContract::SINGLE_RANK);
        let four = with_collective(&policy, FOUR_RANK_COLLECTIVE);
        let text = String::from_utf8(four.canonical_json().unwrap()).unwrap();
        assert!(text.contains("\"rank_count\":4"), "{text}");
        assert_eq!(
            ExecutionPolicy::from_json(&four.canonical_json().unwrap()).unwrap(),
            four
        );
    }

    /// A collective contract has to be internally consistent: a partition with
    /// no reduction order, or four ranks with no partition, describes no run
    /// that could have happened.
    #[test]
    fn an_inconsistent_collective_contract_is_refused() {
        let policy = sample_policy();
        for (collective, expected) in [
            (
                H3CollectiveContract {
                    rank_count: 0,
                    ..H3CollectiveContract::SINGLE_RANK
                },
                "at least one rank",
            ),
            (
                H3CollectiveContract {
                    partition: H3CollectivePartitionContract::MegatronColumnRowV1,
                    ..H3CollectiveContract::SINGLE_RANK
                },
                "both be present or both absent",
            ),
            (
                H3CollectiveContract {
                    rank_count: 4,
                    ..H3CollectiveContract::SINGLE_RANK
                },
                "disagrees with its partition",
            ),
            (
                H3CollectiveContract {
                    rank_count: 1,
                    ..FOUR_RANK_COLLECTIVE
                },
                "disagrees with its partition",
            ),
        ] {
            let error = with_collective(&policy, collective)
                .validate()
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "{error} did not name {expected}");
        }
    }

    /// This build splits nothing, so it refuses to execute a policy that says a
    /// run was split. T3 is what would change that, and it is gated on T0.
    #[test]
    fn this_build_refuses_to_execute_a_partitioned_policy() {
        let four = with_collective(&sample_policy(), FOUR_RANK_COLLECTIVE);
        four.validate().unwrap();
        let error = four
            .numerics
            .ensure_current_for_build(
                ExecutionBackendPolicy::Cpu,
                AttentionBackendPolicy::FullSoftmax,
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("numerics.collective"), "{error}");
    }
}
