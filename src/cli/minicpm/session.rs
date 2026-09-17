use super::*;
use flyingfish::minicpm::memory::{RequestGeometry, estimate_request_memory};

pub(super) struct WorkerOptions<'a> {
    pub model: &'a Path,
    pub draft_model: Option<&'a Path>,
    pub config: Config,
    pub device: Device,
    pub weights: WeightCacheArgs,
    pub device_cache: DeviceCacheArgs,
    pub query_chunk: usize,
    pub batch_invariant: bool,
}

pub(super) struct Worker {
    target: Decoder,
    draft: Option<dspark::Draft>,
    pub cache: DeviceCache,
    pub required_tensor_bytes: u64,
}

impl Worker {
    pub fn open(options: WorkerOptions<'_>, requests: &[RequestGeometry]) -> Result<Self> {
        anyhow::ensure!(!requests.is_empty(), "worker needs at least one request");
        let source = options.weights.weight_source;
        let host_policy = options.weights.cache_policy()?;
        let mut weights = ModelWeights::open(options.model, source, host_policy)?;
        let steps = requests
            .iter()
            .map(|r| r.max_new_tokens as u64)
            .max()
            .unwrap()
            .saturating_add(1);
        let phases = options.config.device_residency_phases(steps);
        let mut demands = phases
            .iter()
            .map(|phase| {
                flyingfish::runtime::residency::PhaseResidencyDemand::from_phase(
                    phase,
                    &weights,
                    &options.device,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let draft = options
            .draft_model
            .map(|path| -> Result<_> {
                let config = dspark::Config::read(path)?;
                config.validate_for(&options.config)?;
                let weights = ModelWeights::open(path, source, host_policy)?;
                let phase = flyingfish::runtime::residency::WeightPhase::new(
                    "minicpm.dspark",
                    weights.tensor_names().map(str::to_owned),
                    steps,
                );
                demands.push(
                    flyingfish::runtime::residency::PhaseResidencyDemand::from_phase(
                        &phase,
                        &weights,
                        &options.device,
                    )?,
                );
                Ok((config, weights))
            })
            .transpose()?;
        let memory = requests
            .iter()
            .map(|geometry| {
                estimate_request_memory(
                    &options.config,
                    &weights,
                    draft.as_ref().map(|(c, w)| (c, w)),
                    &options.device,
                    *geometry,
                )
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max_by_key(|m| m.tensor_peak_bytes)
            .unwrap();
        eprintln!(
            "request tensor budget: {} bytes KV peak, {} bytes activation estimate, {} bytes weight-load reserve; library/allocator margin added separately",
            memory.kv.peak_bytes, memory.activation_peak_bytes, memory.streamed_weight_bytes
        );
        let cache = super::super::decide_auto_residency_with_required_memory(
            &demands,
            &options.device,
            options.device_cache,
            if options.device.is_cpu() {
                0
            } else {
                memory.tensor_peak_bytes
            },
            0,
            1,
        )?;
        let priority = phases
            .iter()
            .filter(|phase| cache.phase_selected(&phase.name))
            .flat_map(|phase| phase.tensors.iter().cloned())
            .collect::<Vec<_>>();
        weights.configure_device_cache_with_priority(cache.clone(), priority)?;
        let target = Decoder::new(options.config, weights, options.device, options.query_chunk)?
            .with_batch_invariant_decode(options.batch_invariant)?;
        let draft = draft
            .map(|(config, mut weights)| -> Result<_> {
                let priority = if cache.phase_selected("minicpm.dspark") {
                    weights
                        .tensor_names()
                        .map(str::to_owned)
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                weights.configure_device_cache_with_priority(cache.clone(), priority)?;
                dspark::Draft::new(config, weights, &target)
            })
            .transpose()?;
        Ok(Self {
            target,
            draft,
            cache,
            required_tensor_bytes: memory.tensor_peak_bytes,
        })
    }

    pub fn generate(
        &mut self,
        input: &[u32],
        limit: usize,
        emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Option<dspark::GenerationStats>> {
        self.target.clear_cache();
        match &mut self.draft {
            Some(draft) => {
                dspark::generate_greedy(&mut self.target, draft, input, limit, emit).map(Some)
            }
            None => self
                .target
                .generate_greedy(input, limit, emit)
                .map(|_| None),
        }
    }
}
