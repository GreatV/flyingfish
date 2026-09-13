use super::{TrellisCommand, ensure_new_output, parse_device, publish_staged_bytes};
use anyhow::{Context, Result};
use flyingfish::runtime::artifact::ArtifactStaging;
use flyingfish::trellis::{
    checkpoint::TrellisCheckpoint,
    pipeline::{StructureOutcome, TextToStructure},
    sampler::SamplerParameters,
};
use serde_json::json;
use std::time::Instant;

pub(super) fn run_inspect(command: TrellisCommand) -> Result<()> {
    let TrellisCommand::Inspect {
        model,
        models_root,
        json: emit_json,
    } = command
    else {
        unreachable!("run_inspect received a different TRELLIS command")
    };
    let checkpoint = TrellisCheckpoint::open(&model, models_root.as_deref())?;
    let conditioner = checkpoint.pipeline.conditioner();

    if emit_json {
        let components: Vec<_> = checkpoint
            .components
            .iter()
            .map(|component| {
                json!({
                    "role": component.role,
                    "reference": component.reference,
                    "component": component.config.name(),
                    "weights": component.weights_path().display().to_string(),
                    "cross_checkpoint": component.is_cross_checkpoint,
                    "executable": component.config.is_executable(),
                })
            })
            .collect();
        let report = json!({
            "pipeline": checkpoint.pipeline.name,
            "root": checkpoint.root.display().to_string(),
            "conditioner": conditioner.as_ref().map(|conditioner| json!({
                "kind": match conditioner.kind {
                    flyingfish::trellis::config::ConditionerKind::Text => "text",
                    flyingfish::trellis::config::ConditionerKind::Image => "image",
                },
                "model": conditioner.model,
            })),
            "components": components,
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("pipeline: {}", checkpoint.pipeline.name);
    if let Some(conditioner) = &conditioner {
        println!(
            "conditioner: {} (not in this checkpoint; supply it with --conditioner)",
            conditioner.model
        );
    }
    println!("components:");
    for component in &checkpoint.components {
        println!(
            "  {:<28} {:<26} {}{}",
            component.role,
            component.config.name(),
            component.reference,
            if component.is_cross_checkpoint {
                " [another checkpoint]"
            } else {
                ""
            }
        );
    }
    Ok(())
}

pub(super) fn run_structure(command: TrellisCommand) -> Result<()> {
    let TrellisCommand::Structure {
        model,
        conditioner,
        models_root,
        prompt,
        samples,
        seed,
        steps,
        cfg_strength,
        cfg_interval_start,
        cfg_interval_end,
        rescale_t,
        device,
        json: emit_json,
        output,
    } = command
    else {
        unreachable!("run_structure received a different TRELLIS command")
    };
    if let Some(output) = &output {
        ensure_new_output(output, "voxel output")?;
    }
    let device = parse_device(&device)?;

    let loading = Instant::now();
    let pipeline = TextToStructure::load(&model, &conditioner, &device, models_root.as_deref())?;
    let loading = loading.elapsed();

    let published = pipeline.parameters();
    let parameters = SamplerParameters {
        steps: steps.unwrap_or(published.steps),
        cfg_strength: cfg_strength.unwrap_or(published.cfg_strength),
        cfg_interval: (
            cfg_interval_start.unwrap_or(published.cfg_interval.0),
            cfg_interval_end.unwrap_or(published.cfg_interval.1),
        ),
        rescale_t: rescale_t.unwrap_or(published.rescale_t),
    };

    let sampling = Instant::now();
    let outcome = pipeline
        .generate(&prompt, samples.get(), seed, Some(parameters))
        .context("failed to generate a sparse structure")?;
    let sampling = sampling.elapsed();

    if let Some(output) = &output {
        let bytes = voxel_json(&prompt, seed, &outcome)?;
        publish_staged_bytes(ArtifactStaging::new(output)?, &bytes)?;
    }

    if emit_json {
        let report = json!({
            "prompt": prompt,
            "seed": seed,
            "samples": samples.get(),
            "resolution": outcome.resolution,
            "sampler": sampler_json(&outcome.parameters),
            "occupancy": outcome.occupancy_per_sample,
            "density": (0..samples.get()).map(|sample| outcome.density(sample)).collect::<Vec<_>>(),
            "timings_seconds": {
                "load": loading.as_secs_f64(),
                "sample_and_decode": sampling.as_secs_f64(),
            },
            "voxel_output": output.as_ref().map(|path| path.display().to_string()),
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("prompt: {prompt}");
    println!(
        "sampler: {} steps, cfg {} over [{}, {}], rescale_t {}",
        outcome.parameters.steps,
        outcome.parameters.cfg_strength,
        outcome.parameters.cfg_interval.0,
        outcome.parameters.cfg_interval.1,
        outcome.parameters.rescale_t
    );
    let volume = outcome.resolution.pow(3);
    for (sample, occupied) in outcome.occupancy_per_sample.iter().enumerate() {
        println!(
            "sample {sample}: {occupied} of {volume} voxels occupied ({:.2}% of the {}^3 grid)",
            outcome.density(sample) * 100.0,
            outcome.resolution
        );
    }
    println!(
        "loaded in {:.2} s, sampled and decoded in {:.2} s",
        loading.as_secs_f64(),
        sampling.as_secs_f64()
    );
    if let Some(output) = &output {
        println!("wrote {}", output.display());
    }
    Ok(())
}

fn sampler_json(parameters: &SamplerParameters) -> serde_json::Value {
    json!({
        "steps": parameters.steps,
        "cfg_strength": parameters.cfg_strength,
        "cfg_interval": [parameters.cfg_interval.0, parameters.cfg_interval.1],
        "rescale_t": parameters.rescale_t,
    })
}

/// The voxel set, as a record another tool can read.
fn voxel_json(prompt: &str, seed: u64, outcome: &StructureOutcome) -> Result<Vec<u8>> {
    let samples: Vec<_> = outcome
        .occupancy_per_sample
        .iter()
        .enumerate()
        .map(|(sample, occupied)| {
            let coordinates: Vec<[u32; 3]> = outcome
                .voxels
                .iter()
                .filter(|voxel| voxel.sample as usize == sample)
                .map(|voxel| [voxel.x, voxel.y, voxel.z])
                .collect();
            json!({
                "sample": sample,
                "occupied": occupied,
                "coordinates": coordinates,
            })
        })
        .collect();
    let record = json!({
        "schema_version": 1,
        "prompt": prompt,
        "seed": seed,
        "resolution": outcome.resolution,
        "sampler": sampler_json(&outcome.parameters),
        "samples": samples,
    });
    let mut bytes = serde_json::to_vec_pretty(&record).context("failed to encode voxel output")?;
    bytes.push(b'\n');
    Ok(bytes)
}
