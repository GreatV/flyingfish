use super::*;
use crate::backend::cuda::moe::reference::{MoeDims, MoeFixture, fixture};
use crate::experts::plan::Capacity;
use crate::vision_input::{ImageGrid, MultimodalInput};
use safetensors::{Dtype, SafeTensors};

#[test]
fn resident_routed_wave_preserves_slots_and_shared_rows() -> Result<()> {
    use crate::experts::{catalogue::ExpertLayout, plan::equal_slots};
    use crate::model_spec::ExpertFormat;
    let model = crate::model_spec::ModelSpec::minicpm(
        &serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/minicpm-config.json"
        )))?,
        &serde_json::json!({"eos_token_id": [248044, 248046]}),
    )?;
    let geometry = ExpertGeometry::from_model(&model)?;
    let shared = (
        model
            .moe
            .context("MiniCPM MoE geometry missing")?
            .shared_intermediate,
        model.routed_weight_boundary(),
    );
    assert_eq!(shared.0, geometry.intermediate);
    assert_eq!(
        (geometry.experts, geometry.top_k, geometry.hidden),
        (256, 8, 2048)
    );
    let layout = ExpertLayout::new(ExpertFormat::Fp8W8a16, &geometry)?;
    use crate::backend::cuda::moe::prefill::{
        prefill_scratch_bytes, prefill_scratch_bytes_mixed_with_boundary,
    };
    for rows in [16, 2048] {
        assert_eq!(
            prefill_scratch_bytes(rows, 16640, 8, 256, 2048, 512, layout.fp8_slot())?,
            prefill_scratch_bytes_mixed_with_boundary(
                rows,
                16640,
                8,
                256,
                2048,
                (512, shared.0 as usize),
                (layout.fp8_slot(), shared.1),
            )?,
        );
    }
    let resident_bytes = layout.slot_bytes() * 10240;
    let mut capacity = Capacity {
        layout,
        span_rows: 16640,
        wave_rows: 16,
        device_expert_bytes: resident_bytes + 197_741_824,
        peer_expert_bytes: 0,
        gpu0_layers: 40,
        host_expert_bytes: 1 << 30,
        slots_per_layer: vec![256; 40],
        host_blocks: 218,
        spare_blocks: 0,
        disk_bytes: 0,
        worker_count: 16,
        disk_workers: 0,
        prefill_staging_slots: 2,
        pinned: true,
    };
    for (span_rows, workspace_rows) in [(4352, 4096), (16640, 16384)] {
        capacity.span_rows = span_rows;
        let (rows, extra) = routed_wave_rows(geometry, &capacity, workspace_rows, shared)?;
        assert_eq!((rows, extra), (2048, 197_741_824));
        assert_eq!(capacity.device_expert_bytes - extra, resident_bytes);
        assert_eq!(capacity.wave_rows, 16);
        assert_eq!(capacity.slots_per_layer, vec![256; 40]);
    }
    for boundary in [
        crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
        crate::model_spec::RoutedWeightBoundary::BeforeContributionBf16,
    ] {
        let mixed_scratch = |rows| {
            prefill_scratch_bytes_mixed_with_boundary(
                rows,
                16640,
                8,
                256,
                2048,
                (512, 1024),
                (capacity.layout.fp8_slot(), boundary),
            )
        };
        let extra = mixed_scratch(2048)? - mixed_scratch(16)?;
        assert_ne!(extra, 197_741_824);
        capacity.device_expert_bytes = resident_bytes + extra;
        assert_eq!(
            routed_wave_rows(geometry, &capacity, 16384, (1024, boundary))?,
            (2048, extra),
        );
    }
    capacity.device_expert_bytes = resident_bytes + 197_741_824;
    capacity.device_expert_bytes -= 1;
    let (rows, extra) = routed_wave_rows(geometry, &capacity, 16384, shared)?;
    assert_eq!(rows, 2047);
    assert!(capacity.device_expert_bytes - extra >= resident_bytes);
    capacity.device_expert_bytes = resident_bytes;
    assert_eq!(
        routed_wave_rows(geometry, &capacity, 16384, shared)?,
        (16, 0)
    );
    capacity.device_expert_bytes = resident_bytes - 1;
    assert!(routed_wave_rows(geometry, &capacity, 16384, shared).is_err());
    capacity.device_expert_bytes = resident_bytes + (1 << 30);
    assert!(
        routed_wave_rows(
            ExpertGeometry {
                top_k: 0,
                ..geometry
            },
            &capacity,
            16384,
            shared
        )
        .is_err()
    );
    assert_eq!(
        routed_wave_rows(
            ExpertGeometry {
                top_k: 3,
                ..geometry
            },
            &capacity,
            16384,
            shared
        )?
        .0,
        5456
    );
    assert_eq!(routed_wave_rows(geometry, &capacity, 1024, shared)?.0, 1024);
    capacity.span_rows = 1023;
    assert_eq!(routed_wave_rows(geometry, &capacity, 4096, shared)?.0, 1023);
    capacity.span_rows = 16640;
    capacity.wave_rows = 4096;
    assert_eq!(
        routed_wave_rows(geometry, &capacity, 16384, shared)?,
        (4096, 0)
    );
    assert!(routed_wave_rows(geometry, &capacity, 2048, shared).is_err());
    capacity.wave_rows = 256;
    for (format, span, workspace, slots, host, spare) in [
        (ExpertFormat::Fp8W8a16, 2144, 2048, 4889, 10240, 0),
        (ExpertFormat::Fp8W8a16, 16352, 8192, 4550, 10240, 0),
        (ExpertFormat::Bf16, 2144, 2048, 2411, 7844, 14),
        (ExpertFormat::Bf16, 16352, 8192, 2246, 8101, 14),
    ] {
        capacity.layout = ExpertLayout::new(format, &geometry)?;
        capacity.span_rows = span;
        capacity.slots_per_layer = equal_slots(slots, geometry.layers)?;
        capacity.device_expert_bytes = slots * capacity.slot_bytes();
        capacity.host_blocks = host;
        capacity.spare_blocks = spare;
        assert_eq!(
            routed_wave_rows(geometry, &capacity, workspace, shared)?,
            (256, 0)
        );
        println!(
            "PASS routed_wave_partial_replay format={format:?} span={span} workspace={workspace} slots={slots} shared_rows=256 routed_rows=256 extra_bytes=0"
        );
    }
    capacity.slots_per_layer = vec![256; 40];
    capacity.device_expert_bytes = 10240 * capacity.slot_bytes() + (1 << 30);
    assert_eq!(
        routed_wave_rows(geometry, &capacity, 8192, shared)?,
        (256, 0)
    );
    capacity.layout = ExpertLayout::new(ExpertFormat::Fp8W8a16, &geometry)?;
    capacity.gpu0_layers = 20;
    capacity.device_expert_bytes = 20 * 256 * capacity.slot_bytes();
    capacity.peer_expert_bytes = 20 * 256 * capacity.slot_bytes();
    assert_eq!(
        routed_wave_rows(geometry, &capacity, 8192, shared)?,
        (256, 0)
    );
    capacity.span_rows = 1312;
    capacity.wave_rows = 16;
    capacity.gpu0_layers = 21;
    let mut split_slots = vec![188; 21];
    split_slots.extend(std::iter::repeat_n(256, 19));
    capacity.slots_per_layer = split_slots;
    capacity.device_expert_bytes = 21 * 188 * capacity.slot_bytes();
    let peer_base = 19 * 256 * capacity.slot_bytes();
    capacity.peer_expert_bytes = peer_base;
    assert_eq!(peer_routed_wave_rows(geometry, &capacity, shared)?, (16, 0));
    let fp8 = capacity
        .layout
        .fp8_slot()
        .context("FP8 slot missing")?
        .clone();
    let peer_baseline = u64::try_from(residency::peer_prefill_bytes(
        geometry,
        1312,
        16,
        shared,
        Some(&fp8),
    )?)?;
    let peer_full = u64::try_from(residency::peer_prefill_bytes(
        geometry,
        1312,
        1312,
        shared,
        Some(&fp8),
    )?)?;
    let peer_extra = peer_full - peer_baseline;
    capacity.peer_expert_bytes = peer_base + peer_extra;
    assert_eq!(
        peer_routed_wave_rows(geometry, &capacity, shared)?,
        (1312, peer_extra)
    );
    capacity.peer_expert_bytes = peer_base + peer_extra - 1;
    assert_eq!(peer_routed_wave_rows(geometry, &capacity, shared)?.0, 1311);
    capacity.peer_expert_bytes = peer_base + 200_828_000;
    let (a4000_peer_wave, a4000_peer_extra) = peer_routed_wave_rows(geometry, &capacity, shared)?;
    assert!(a4000_peer_wave > 600 && a4000_peer_extra <= 200_828_000);
    capacity.gpu0_layers = 40;
    capacity.slots_per_layer = vec![256; 40];
    assert_eq!(peer_routed_wave_rows(geometry, &capacity, shared)?, (16, 0));
    println!(
        "PASS routed_wave_full_resident target_rows=2048 shared_rows=16 extra_bytes=197676800 exact_budget=true boundary_minus_one_rows=2047 non_fp8_unchanged=true peer_wave={a4000_peer_wave}"
    );
    Ok(())
}

#[test]
fn peer_send_layout_packs_in_place_aligned_without_overlap() -> Result<()> {
    for (hidden, top_k, exact_gap_free) in
        [(2048usize, 8usize, true), (128, 4, true), (128, 2, false)]
    {
        let max_tail = PeerSendLayout::tail_bf16_elems(16, hidden, top_k)?;
        let workspace_bytes = (16 * hidden + max_tail) * size_of::<bf16>();
        for rows in 1..=16usize {
            let layout = PeerSendLayout::for_rows(rows, hidden, top_k)?;
            let x_bytes = rows * hidden * size_of::<bf16>();
            assert_eq!(layout.ids_off, x_bytes);
            assert_eq!(layout.ids_off % 16, 0);
            assert_eq!(layout.weights_off % 16, 0);
            assert!(layout.ids_off + layout.ids_bytes <= layout.weights_off);
            assert_eq!(layout.send_bytes, layout.weights_off + layout.weights_bytes);
            assert!(layout.send_bytes <= workspace_bytes);
            if exact_gap_free {
                assert_eq!(layout.weights_off, layout.ids_off + layout.ids_bytes);
                assert_eq!(layout.send_bytes, rows * (hidden * 2 + top_k * 6));
            }
        }
    }
    assert_eq!(PeerSendLayout::for_rows(1, 2048, 8)?.send_bytes, 4_144);
    assert_eq!(PeerSendLayout::for_rows(9, 2048, 8)?.send_bytes, 37_296);
    assert!(PeerSendLayout::for_rows(0, 2048, 8).is_err());
    assert!(PeerSendLayout::for_rows(1, 4, 8).is_err());
    Ok(())
}

#[test]
fn decode_phase_reset_recovers_after_completion_errors() -> Result<()> {
    let mut phase = DecodePhase::default();
    for rows in [1, 8, 16] {
        phase.begin(rows, 0x1000, 0x2000)?;
        assert!(phase.begin(rows, 0x1000, 0x2000).is_err());
        assert!(phase.check(0x3000, 0x2000).is_err());
        assert!(phase.check(0x1000, 0x4000).is_err());
        phase.check(0x1000, 0x2000)?;
        phase.reset();
        phase.begin(rows, 0x5000, 0x6000)?;
        phase.check(0x5000, 0x6000)?;
        assert_eq!(phase.rows, rows);
        phase.finish()?;
        assert!(phase.finish().is_err());
    }
    Ok(())
}

#[test]
fn expert_host_reserve_keeps_span_independent_of_cpu_ceiling() -> Result<()> {
    use crate::experts::{
        catalogue::ExpertLayout,
        plan::{PlanInputs, Reserves, ServiceCell, ServiceTable, SpanCandidate, select_span},
    };
    let geometry = ExpertGeometry {
        first_layer: 0,
        layers: 2,
        experts: 16,
        hidden: 32,
        intermediate: 64,
        top_k: 2,
    };
    let layout = ExpertLayout::new(crate::model_spec::ExpertFormat::Bf16, &geometry)?;
    let slot = layout.slot_bytes();
    let service = ServiceTable::new(
        "fixture",
        [(1u32, 1u32), (1, 2), (64, 1), (64, 2)]
            .into_iter()
            .map(|(rows, experts)| ServiceCell {
                workers: 2,
                rows,
                experts,
                seconds: f64::from(rows * experts) * 0.00002,
            })
            .collect(),
    )?;
    let mut admitted = Vec::new();
    for measured in [15, 16] {
        let h2d = slot as f64 / ((f64::from(measured) + 0.5) * 0.00002);
        let candidates = [32, 256]
            .map(|span_rows| {
                let cpu = residency::cpu_bound(geometry, slot, span_rows, &service, h2d)?;
                assert_eq!(cpu.3, measured);
                Ok(SpanCandidate {
                    span_rows,
                    wave_rows: 1,
                    reserves: Reserves {
                        device_fixed_bytes: 0,
                        host_fixed_bytes: host_workspace(geometry, span_rows, 2, 1)?,
                        residual_row_bytes: 64,
                        normalized_row_bytes: 64,
                        contribution_row_bytes: 128,
                        done_row_bytes: 8,
                        route_row_bytes: 36,
                        prefill_staging_slots: 1,
                        disk_ring_slots: 0,
                    },
                })
            })
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let inputs = PlanInputs {
            geometry,
            layout: layout.clone(),
            device_free_bytes: slot * 16,
            host_available_bytes: 439_520,
            h2d_bytes_per_second: h2d,
            d2h_bytes_per_second: h2d,
            copy_engines: 1,
            disk_readers: 1,
            host_registration_bytes: 0,
            service: &service,
            reserves: &candidates[0].reserves,
            span_rows: 256,
            wave_rows: 1,
        };
        let (_, capacity) = select_span(&inputs, 1, 256, &candidates)?;
        println!(
            "expert_host_reserve measured_cpu_ceiling={measured} admitted_span={} large_span_host_bytes={}",
            capacity.span_rows, candidates[1].reserves.host_fixed_bytes
        );
        admitted.push(capacity.span_rows);
    }
    assert_eq!(admitted, [32, 32]);
    let extra = |rows| -> Result<usize> {
        Ok(decode_device_bytes(geometry, geometry.intermediate, rows)?
            - decode_device_bytes(geometry, geometry.intermediate, 1)?)
    };
    let total = 24usize << 30;
    let mut request = crate::prefill::WorkspaceRequest {
        capacity: 16,
        requested: Some(16),
        free_after_resident: total / 16 + 16 * 64 + extra(8)?,
        total,
        alignment: 1,
        row_bytes: 64,
        fixed_transient_bytes: extra(8)?,
    };
    assert!(crate::prefill::WorkspaceBudget::new(request, |_| Ok(0), Ok).is_ok());
    request.fixed_transient_bytes = extra(16)?;
    assert!(crate::prefill::WorkspaceBudget::new(request, |_| Ok(0), Ok).is_err());
    assert!(decode_device_bytes(geometry, geometry.intermediate, 17).is_err());
    assert_eq!(extra(1)?, 0);
    // Each row includes a 4-byte router completion counter.
    let row =
        extra(2)? - (decode_scratch_bytes(2, 2, 64) - decode_scratch_bytes(1, 2, 64)) as usize;
    assert_eq!(row, (32 * 2 + 26) * 2 + 32 * 2 + 4 + 16 * 2);
    println!(
        "PASS expert_decode_row_reserve rows=1,8,16 extra8={} extra16={} legacy8_control=admitted full16=rejected",
        extra(8)?,
        extra(16)?
    );
    Ok(())
}

#[test]
fn expert_host_profile_obeys_the_current_cgroup_limit() -> Result<()> {
    use crate::experts::{
        catalogue::ExpertLayout,
        plan::{PlanInputs, Reserves, ServiceCell, ServiceTable, capacity},
    };
    let available = super::host_memory(Path::new("/"))?;
    ensure!(available > 0, "invalid expert host memory profile");
    let root = std::env::temp_dir().join(format!(
        "flyingfish-host-memory-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::create_dir(&root)?;
    let result = (|| -> Result<()> {
        let write = |path: &str, value: &str| -> Result<()> {
            let path = root.join(path);
            std::fs::create_dir_all(path.parent().context("fixture parent missing")?)?;
            Ok(std::fs::write(path, value)?)
        };
        let placement = |available| -> Result<Vec<_>> {
            let model = crate::model_spec::ModelSpec::minicpm(
                &serde_json::from_str(include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/data/minicpm-config.json"
                )))?,
                &serde_json::json!({"eos_token_id": [248044, 248046]}),
            )?;
            let geometry = ExpertGeometry::from_model(&model)?;
            let service = ServiceTable::new(
                "fixture",
                vec![
                    ServiceCell {
                        workers: 2,
                        rows: 1,
                        experts: 1,
                        seconds: 0.001,
                    },
                    ServiceCell {
                        workers: 2,
                        rows: 1,
                        experts: 2,
                        seconds: 0.002,
                    },
                    ServiceCell {
                        workers: 2,
                        rows: 1,
                        experts: 8,
                        seconds: 0.008,
                    },
                ],
            )?;
            let reserves = Reserves {
                device_fixed_bytes: 0,
                host_fixed_bytes: 0,
                residual_row_bytes: 0,
                normalized_row_bytes: 0,
                contribution_row_bytes: 0,
                done_row_bytes: 0,
                route_row_bytes: 0,
                prefill_staging_slots: 1,
                disk_ring_slots: 1,
            };
            [
                crate::model_spec::ExpertFormat::Bf16,
                crate::model_spec::ExpertFormat::Fp8W8a16,
            ]
            .into_iter()
            .map(|format| {
                let inputs = PlanInputs {
                    geometry,
                    layout: ExpertLayout::new(format, &geometry)?,
                    device_free_bytes: 12 << 30,
                    host_available_bytes: available,
                    h2d_bytes_per_second: 1e9,
                    d2h_bytes_per_second: 1e9,
                    copy_engines: 1,
                    disk_readers: 1,
                    host_registration_bytes: 0,
                    service: &service,
                    reserves: &reserves,
                    span_rows: 1,
                    wave_rows: 1,
                };
                let plan = capacity(&inputs, 1)?;
                Ok((plan.slots_per_layer, plan.host_blocks, plan.disk_bytes))
            })
            .collect()
        };
        let expected_placement = placement(5 << 30)?;
        write("proc/meminfo", "MemAvailable: 16777216 kB\n")?;
        write(
            "proc/self/limits",
            "Max resident set unlimited unlimited bytes\n",
        )?;
        let mounts = "1 0 0:1 / /other rw - cgroup none rw,cpuset\n2 0 0:2 /pod /controllers/memory\\040bank rw - cgroup none rw,cpu,memory\n3 0 0:3 / /sys/fs/cgroup rw - cgroup2 none rw\n";
        write("proc/self/mountinfo", mounts)?;
        write(
            "proc/self/cgroup",
            "2:cpuset:/other\n5:cpu,memory:/pod/job/leaf\n",
        )?;
        for (path, limit) in [
            ("controllers/memory bank/job", "6442450944\n"),
            ("controllers/memory bank/job/leaf", "8589934592\n"),
        ] {
            write(&format!("{path}/memory.limit_in_bytes"), limit)?;
            write(&format!("{path}/memory.usage_in_bytes"), "1073741824\n")?;
            write(
                &format!("{path}/memory.stat"),
                "total_active_file 0\ntotal_inactive_file 0\n",
            )?;
        }
        ensure!(
            super::host_memory(&root)? == 5 << 30,
            "v1 ancestor headroom did not bind"
        );
        println!(
            "PASS host_memory_fixture v1 ancestor_available_bytes={}",
            5u64 << 30
        );
        for (active, inactive) in [(0u64, 1u64 << 30), (2 << 30, 1 << 30), (3 << 30, 0), (0, 0)] {
            let cache = active + inactive;
            for path in [
                "controllers/memory bank/job",
                "controllers/memory bank/job/leaf",
            ] {
                write(
                    &format!("{path}/memory.usage_in_bytes"),
                    &format!("{}\n", (1 << 30) + cache),
                )?;
                write(
                    &format!("{path}/memory.stat"),
                    &format!("total_active_file {active}\ntotal_inactive_file {inactive}\n"),
                )?;
            }
            let available = super::host_memory(&root)?;
            ensure!(available == 5 << 30, "v1 cache changed headroom");
            ensure!(
                placement(available)? == expected_placement,
                "v1 cache changed placement"
            );
        }
        for path in [
            "controllers/memory bank/job",
            "controllers/memory bank/job/leaf",
        ] {
            write(
                &format!("{path}/memory.limit_in_bytes"),
                "9223372036854771712\n",
            )?;
        }
        ensure!(
            super::host_memory(&root)? == 16 << 30,
            "v1 unlimited bound MemAvailable"
        );
        write(
            "proc/self/cgroup",
            "0::/unified\n5:cpu,memory:/pod/job/leaf\n",
        )?;
        ensure!(
            super::host_memory(&root)? == 16 << 30,
            "hybrid memory controller was not selected"
        );
        write("proc/self/cgroup", "0::/job/leaf\n")?;
        for (path, limit) in [
            ("sys/fs/cgroup/job", "6442450944\n"),
            ("sys/fs/cgroup/job/leaf", "8589934592\n"),
        ] {
            write(&format!("{path}/memory.max"), limit)?;
            write(&format!("{path}/memory.current"), "1073741824\n")?;
            write(
                &format!("{path}/memory.stat"),
                "file 0\nshmem 0\ninactive_file 0\n",
            )?;
        }
        ensure!(
            super::host_memory(&root)? == 5 << 30,
            "v2 ancestor headroom did not bind"
        );
        for (cache, inactive) in [(1u64 << 30, 1 << 30), (3 << 30, 1 << 30), (0, 0)] {
            for path in ["sys/fs/cgroup/job", "sys/fs/cgroup/job/leaf"] {
                write(
                    &format!("{path}/memory.current"),
                    &format!("{}\n", (1 << 30) + cache),
                )?;
                write(
                    &format!("{path}/memory.stat"),
                    &format!("file {cache}\nshmem 0\ninactive_file {inactive}\n"),
                )?;
            }
            let available = super::host_memory(&root)?;
            ensure!(available == 5 << 30, "v2 cache changed headroom");
            ensure!(
                placement(available)? == expected_placement,
                "v2 cache changed placement"
            );
        }
        write("sys/fs/cgroup/job/memory.current", "3221225472\n")?;
        write(
            "sys/fs/cgroup/job/memory.stat",
            "file 3221225472\nshmem 1073741824\ninactive_file 1073741824\n",
        )?;
        ensure!(
            super::host_memory(&root)? == 5 << 30,
            "shared memory was reclaimed"
        );
        write("sys/fs/cgroup/job/memory.stat", "file 0\n")?;
        ensure!(
            super::host_memory(&root).is_err(),
            "missing shared-memory counter admitted"
        );
        write("sys/fs/cgroup/job/memory.stat", "file invalid\nshmem 0\n")?;
        ensure!(
            super::host_memory(&root).is_err(),
            "invalid file counter admitted"
        );
        for path in ["sys/fs/cgroup/job", "sys/fs/cgroup/job/leaf"] {
            write(&format!("{path}/memory.max"), "max\n")?;
        }
        ensure!(
            super::host_memory(&root)? == 16 << 30,
            "v2 unlimited bound MemAvailable"
        );
        write(
            "proc/self/limits",
            "Max resident set 6442450944 unlimited bytes\n",
        )?;
        write("proc/self/status", "VmRSS: 1048576 kB\n")?;
        ensure!(
            super::host_memory(&root)? == 5 << 30,
            "process resident limit did not bind"
        );
        ensure!(
            placement(super::host_memory(&root)?)? == expected_placement,
            "process resident limit changed placement for equal headroom"
        );
        write("proc/self/status", "VmRSS: 7340032 kB\n")?;
        ensure!(
            super::host_memory(&root)? == 0,
            "exhausted RSS limit admitted"
        );
        write("proc/self/status", "VmRSS: invalid kB\n")?;
        ensure!(super::host_memory(&root).is_err(), "invalid RSS admitted");
        write("proc/meminfo", "MemAvailable: 886853164 kB\n")?;
        write("sys/fs/cgroup/job/memory.max", "128849018880\n")?;
        write("sys/fs/cgroup/job/memory.current", "107595685888\n")?;
        write(
            "sys/fs/cgroup/job/memory.stat",
            "file 106297856000\nshmem 8192\n",
        )?;
        write("proc/self/status", "VmRSS: 1973172 kB\n")?;
        let model = crate::model_spec::ModelSpec::glm5_next(
            &serde_json::from_str(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/data/glm53-flash-config.json"
            )))?,
            &serde_json::Value::Null,
        )?;
        let geometry = ExpertGeometry::from_model(&model)?;
        let layout = ExpertLayout::new(
            model.expert_format.context("GLM format missing")?,
            &geometry,
        )?;
        let slot = layout.slot_bytes();
        ensure!(slot == 25_171_968, "GLM native slot extent changed");
        for limit in [42_949_672_960u64, 40] {
            write(
                "proc/self/limits",
                &format!("Max resident set {limit} {limit} bytes\n"),
            )?;
            let host = super::host_memory(&root)?;
            let host_bytes = host - host / 16;
            let capacity = crate::experts::plan::staging_capacity(
                geometry,
                layout.clone(),
                8,
                79_600_893_952,
                host_bytes,
                2,
            );
            if limit == 40 {
                ensure!(host == 0, "exhausted 40-byte RSS limit admitted");
                let error = capacity.expect_err("40-byte RSS limit admitted staging");
                ensure!(
                    error
                        .to_string()
                        .ends_with("host_bytes=0 slot=25171968 copies=2 required=151031808"),
                    "native staging failure omits budget operands: {error}"
                );
            } else {
                ensure!(host == 40_929_144_832, "GLM process headroom changed");
                ensure!(host_bytes == 38_371_073_280, "GLM host reserve changed");
                let (capacity, _) = capacity?;
                ensure!(
                    u64::from(capacity.host_blocks + capacity.spare_blocks + 6) * slot
                        <= host_bytes,
                    "GLM host allocation exceeds retained headroom"
                );
            }
        }
        println!(
            "PASS glm_host_staging rss_bytes=42949672960 sampled_rss_bytes=2020528128 host_bytes=38371073280 slot=25171968 copies=2 required=151031808 malformed_rss_bytes=40 rejected=true"
        );
        write(
            "proc/self/limits",
            "Max resident set unlimited unlimited bytes\n",
        )?;
        write("proc/self/cgroup", "2:cpuset:/other\n")?;
        ensure!(
            super::host_memory(&root).is_err(),
            "missing memory hierarchy admitted"
        );
        println!(
            "PASS host_memory_fixtures v1_finite=true v1_unlimited=true hybrid=true v2_finite=true v2_unlimited=true rss_finite=true rss_exhausted=true cache_independent_placement=true shmem_retained=true malformed_stats_rejected=true missing_controller_rejected=true"
        );
        Ok(())
    })();
    std::fs::remove_dir_all(&root)?;
    result
}

fn native_fixture(
    fixture: &MoeFixture,
    directory: &Path,
    format: crate::model_spec::ExpertFormat,
) -> Result<Arc<crate::experts::catalogue::ExpertCatalogue>> {
    use crate::checkpoint::{TensorEntry, TensorManifest, TensorScope};
    let geometry = ExpertGeometry::from_model(&hybrid_spec()?)?;
    let d = fixture.dims;
    let mut bytes = Vec::new();
    let mut manifest = TensorManifest {
        entries: Default::default(),
        resident_bytes: 0,
    };
    if format == crate::model_spec::ExpertFormat::Fp8W8a16 {
        for layer in 0..geometry.layers {
            for expert in 0..geometry.experts {
                for (tag, op, n, k) in [
                    (0usize, "gate_proj", d.inter, d.hidden),
                    (1, "up_proj", d.inter, d.hidden),
                    (2, "down_proj", d.hidden, d.inter),
                ] {
                    for scale in [false, true] {
                        let start = bytes.len() as u64;
                        if scale {
                            for row in 0..n {
                                let value = bf16::from_f32(
                                    (1.0 + ((row * 13
                                        + expert as usize * 7
                                        + layer as usize * 11
                                        + tag)
                                        % 31) as f32
                                        / 64.0)
                                        / 128.0,
                                );
                                bytes.extend(value.to_le_bytes());
                            }
                        } else {
                            bytes.extend((0..n * k).map(|index| {
                                let (row, col) = (index / k, index % k);
                                (((row * 37
                                    + col * 11
                                    + expert as usize * 17
                                    + layer as usize * 19
                                    + tag)
                                    % 61) as u8)
                                    | ((((col / 7 + row + expert as usize) % 2) as u8) << 7)
                            }));
                        }
                        let suffix = if scale { "weight_scale" } else { "weight" };
                        let role = format!("mlp.experts.{expert}.{op}.{suffix}");
                        manifest.entries.insert(format!("model.language_model.layers.{layer}.{role}"), TensorEntry {
                            dtype: if scale { safetensors::Dtype::BF16 } else { safetensors::Dtype::F8_E4M3 },
                            shape: if scale { vec![n, 1] } else { vec![n, k] },
                            shard: "expert-fixture.bin".into(), bytes: bytes.len() - start as usize,
                            load: false, data_start: 0, data_offsets: [start, bytes.len() as u64],
                            scope: TensorScope::Text, layer: Some(layer), role,
                            scale_name: (!scale).then(|| format!("model.language_model.layers.{layer}.mlp.experts.{expert}.{op}.weight_scale")),
                        });
                    }
                }
            }
        }
    }
    for layer in 0..geometry.layers {
        if format != crate::model_spec::ExpertFormat::Bf16 {
            break;
        }
        for (role, values, shape) in [
            (
                "gate_up_proj",
                &fixture.gate_up,
                vec![d.experts, 2 * d.inter, d.hidden],
            ),
            (
                "down_proj",
                &fixture.down,
                vec![d.experts, d.hidden, d.inter],
            ),
        ] {
            let start = bytes.len() as u64;
            bytes.extend(values.iter().flat_map(|value| {
                let value = if layer == 0 {
                    *value
                } else {
                    bf16::from_f32(-0.75 * value.to_f32())
                };
                value.to_bits().to_le_bytes()
            }));
            let end = bytes.len() as u64;
            manifest.entries.insert(
                format!("model.language_model.layers.{layer}.mlp.experts.{role}"),
                TensorEntry {
                    dtype: safetensors::Dtype::BF16,
                    shape,
                    shard: "expert-fixture.bin".into(),
                    bytes: (end - start) as usize,
                    load: false,
                    data_start: 0,
                    data_offsets: [start, end],
                    scope: TensorScope::Text,
                    layer: Some(layer),
                    role: format!("mlp.experts.{role}"),
                    scale_name: None,
                },
            );
        }
    }
    std::fs::create_dir_all(directory)?;
    let path = directory.join("expert-fixture.bin");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    std::io::Write::write_all(&mut file, &bytes)?;
    Ok(Arc::new(
        crate::experts::catalogue::ExpertCatalogue::from_manifest(&manifest, geometry, format)?,
    ))
}

pub(super) fn hybrid_spec() -> Result<crate::model_spec::ModelSpec> {
    use crate::model_spec::{LayerPlan, Mixer};
    let mut spec = crate::model_spec::ModelSpec::minicpm(
        &serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/minicpm-config.json"
        )))?,
        &serde_json::json!({"eos_token_id": [248044, 248046]}),
    )?;
    spec.hidden = 128;
    spec.intermediate = 64;
    spec.vocab = 32;
    spec.max_position = 32;
    spec.eos = vec![31];
    spec.vision = None;
    spec.moe = Some(crate::model_spec::MoeConfig {
        experts: 16,
        experts_per_tok: 4,
        expert_intermediate: 64,
        shared_intermediate: 64,
        score: crate::model_spec::RouterScore::Softmax,
        shared_gate: true,
    });
    spec.layers = vec![
        LayerPlan {
            geometry: None,
            mixer: Mixer::Gdn,
            state_index: 0,
            rope: false,
            sliding_window: None,
            ffn: crate::model_spec::FfnKind::Moe,
        },
        LayerPlan {
            geometry: None,
            mixer: Mixer::FullAttention,
            state_index: 0,
            rope: true,
            sliding_window: None,
            ffn: crate::model_spec::FfnKind::Moe,
        },
    ];
    Ok(spec)
}

fn hybrid_manifest(
    spec: &crate::model_spec::ModelSpec,
) -> Result<crate::checkpoint::TensorManifest> {
    use crate::checkpoint::{TensorEntry, TensorManifest, TensorScope};
    use crate::model_spec::Mixer;
    ensure!(
        spec.architecture == crate::model_spec::Architecture::MiniCpmV47,
        "hybrid fixture requires MiniCpmV47 architecture, received {:?}",
        spec.architecture
    );
    ensure!(
        spec.layers
            .iter()
            .map(|layer| layer.mixer)
            .eq([Mixer::Gdn, Mixer::FullAttention]),
        "hybrid fixture requires one GDN and one full-attention layer"
    );
    let g = crate::op_contract::GdnSpec::from_model(spec)?;
    let a = crate::op_contract::FullAttentionSpec::from_model(spec)?;
    let h = spec.hidden as usize;
    let m = spec.moe.context("hybrid fixture MoE missing")?;
    let i = m.shared_intermediate as usize;
    let mut manifest = TensorManifest {
        entries: Default::default(),
        resident_bytes: 0,
    };
    for (layer, plan) in spec.layers.iter().enumerate() {
        let mut shapes = vec![
            ("input_layernorm.weight", vec![h]),
            ("post_attention_layernorm.weight", vec![h]),
            ("mlp.gate.weight", vec![m.experts as usize, h]),
            ("mlp.shared_expert.gate_proj.weight", vec![i, h]),
            ("mlp.shared_expert.up_proj.weight", vec![i, h]),
            ("mlp.shared_expert.down_proj.weight", vec![h, i]),
            ("mlp.shared_expert_gate.weight", vec![1, h]),
        ];
        shapes.extend(match plan.mixer {
            Mixer::Gdn => vec![
                (
                    "linear_attn.in_proj_qkv.weight",
                    vec![g.conv_dim()? as usize, h],
                ),
                (
                    "linear_attn.in_proj_z.weight",
                    vec![(g.hv * g.dv) as usize, h],
                ),
                ("linear_attn.in_proj_a.weight", vec![g.hv as usize, h]),
                ("linear_attn.in_proj_b.weight", vec![g.hv as usize, h]),
                (
                    "linear_attn.out_proj.weight",
                    vec![h, (g.hv * g.dv) as usize],
                ),
                (
                    "linear_attn.conv1d.weight",
                    vec![g.conv_dim()? as usize, 1, g.conv_width as usize],
                ),
                ("linear_attn.A_log", vec![g.hv as usize]),
                ("linear_attn.dt_bias", vec![g.hv as usize]),
                ("linear_attn.norm.weight", vec![g.dv as usize]),
            ],
            Mixer::FullAttention => vec![
                (
                    "self_attn.q_proj.weight",
                    vec![(2 * a.hq * a.d) as usize, h],
                ),
                ("self_attn.k_proj.weight", vec![(a.hkv * a.d) as usize, h]),
                ("self_attn.v_proj.weight", vec![(a.hkv * a.d) as usize, h]),
                ("self_attn.o_proj.weight", vec![h, (a.hq * a.d) as usize]),
                ("self_attn.q_norm.weight", vec![a.d as usize]),
                ("self_attn.k_norm.weight", vec![a.d as usize]),
            ],
        });
        for (role, shape) in shapes {
            let bytes = shape.iter().product::<usize>() * size_of::<bf16>();
            manifest.resident_bytes += bytes;
            manifest.entries.insert(
                format!("model.language_model.layers.{layer}.{role}"),
                TensorEntry {
                    dtype: safetensors::Dtype::BF16,
                    shape,
                    shard: "synthetic.safetensors".into(),
                    bytes,
                    load: true,
                    data_start: 0,
                    data_offsets: [0, bytes as u64],
                    scope: TensorScope::Text,
                    layer: Some(layer as u32),
                    role: role.into(),
                    scale_name: None,
                },
            );
        }
    }
    Ok(manifest)
}

impl Engine {
    pub(crate) fn load_moe_test(
        fixture: &MoeFixture,
        runtime: &Path,
        workspace_rows: usize,
        format: crate::model_spec::ExpertFormat,
    ) -> Result<Self> {
        let geometry = ExpertGeometry::from_model(&hybrid_spec()?)?;
        let count = geometry.layers * geometry.experts;
        let layout = crate::experts::catalogue::ExpertLayout::new(format, &geometry)?;
        let capacity = Capacity {
            layout: layout.clone(),
            span_rows: fixture
                .dims
                .rows
                .clamp(8, crate::linear_plan::MAX_SKINNY_ROWS) as u32,
            wave_rows: 8,
            device_expert_bytes: layout.slot_bytes() * u64::from(count),
            peer_expert_bytes: 0,
            gpu0_layers: geometry.layers,
            host_expert_bytes: 0,
            slots_per_layer: vec![geometry.experts; geometry.layers as usize],
            host_blocks: 0,
            spare_blocks: 0,
            disk_bytes: 0,
            worker_count: 2,
            disk_workers: 0,
            prefill_staging_slots: 0,
            pinned: false,
        };
        Self::load_moe_placed(
            fixture,
            runtime,
            &capacity,
            &vec![(0..geometry.experts).collect::<Vec<_>>(); geometry.layers as usize],
            crate::experts::plan::Admission {
                budget_bytes_per_token: 0,
            },
            2,
            workspace_rows,
        )
    }

    pub(in crate::backend::cuda::engine) fn load_moe_placed(
        fixture: &MoeFixture,
        runtime: &Path,
        capacity: &Capacity,
        initial: &[Vec<u32>],
        admission: crate::experts::plan::Admission,
        engines: u32,
        workspace_rows: usize,
    ) -> Result<Self> {
        use crate::backend::cuda::{
            linear::LayerLinearPlan,
            weights::{StoredTensor, WeightCatalog},
        };
        let d = fixture.dims;
        ensure!(
            d.hidden == 128 && d.inter == 64 && d.experts == 16 && d.top_k == 4,
            "MoE hook expects H128/I64/E16/K4, received {d:?}"
        );
        ensure!(
            (8..=32).contains(&workspace_rows),
            "hybrid fixture workspace_rows={workspace_rows} outside8..32"
        );
        let mut model_spec = hybrid_spec()?;
        model_spec.expert_format = Some(match &capacity.layout {
            crate::experts::catalogue::ExpertLayout::Bf16Fused { .. } => {
                crate::model_spec::ExpertFormat::Bf16
            }
            crate::experts::catalogue::ExpertLayout::Bf16Separate { .. } => {
                crate::model_spec::ExpertFormat::Bf16Separate
            }
            crate::experts::catalogue::ExpertLayout::Fp8W8a16(_) => {
                crate::model_spec::ExpertFormat::Fp8W8a16
            }
            crate::experts::catalogue::ExpertLayout::Fp8W8a16Block(_, block_scale) => {
                crate::model_spec::ExpertFormat::Fp8W8a16Block {
                    block: block_scale.block(),
                    scale_dtype: block_scale.dtype(),
                }
            }
            crate::experts::catalogue::ExpertLayout::Fp4E2m1Block(..) => {
                anyhow::bail!("E2M1 hybrid fixtures are not bound yet")
            }
        });
        let config = model_spec.config();
        let manifest = hybrid_manifest(&model_spec)?;
        let geometry = ExpertGeometry::from_model(&model_spec)?;
        let service = tier_service(capacity.worker_count)?;
        let decode_rows = fixture.dims.rows.min(crate::linear_plan::MAX_SKINNY_ROWS);
        let cpu_bounds = if capacity.prefill_staging_slots > 0 {
            let bound = residency::cpu_bound(
                geometry,
                capacity.slot_bytes(),
                capacity.span_rows,
                &service,
                1_000_000_000.0,
            )?;
            (
                bound.0.max(decode_rows),
                bound.1.max(decode_rows * geometry.top_k as usize),
            )
        } else {
            (decode_rows, decode_rows * geometry.top_k as usize)
        };
        let device = Device::new(0)?;
        let peer = if capacity.gpu0_layers < geometry.layers {
            Some(match device.peer()? {
                Some(peer) => peer,
                None => Device::new(0)?,
            })
        } else {
            None
        };
        let stream = &device.stream;
        let catalogue = native_fixture(
            fixture,
            runtime,
            model_spec
                .expert_format
                .context("MoE expert_format missing")?,
        )?;
        let mut slots = Slots::load(
            &device,
            peer.as_ref(),
            geometry,
            residency::Load {
                capacity,
                admission,
                initial,
                catalogue,
                directory: runtime,
                engines,
                max_rows: cpu_bounds.0 as u32,
                swiglu_rounding: crate::model_spec::SwigluRounding::Twice,
                swiglu_limit: 0.0,
                routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
            },
        )?;
        std::fs::remove_file(runtime.join("expert-fixture.bin"))?;
        let mut frequency = vec![vec![0.0; geometry.experts as usize]; geometry.layers as usize];
        for (layer, experts) in initial.iter().enumerate() {
            for &expert in experts {
                frequency[layer][expert as usize] = 1.0;
            }
        }
        slots.begin_request(&frequency)?;
        device.upload.synchronize()?;
        if let Some(peer) = &peer {
            peer.upload.synchronize()?;
        }
        let mut tensors = std::collections::BTreeMap::new();
        for (name, entry) in &manifest.entries {
            let values = match entry.role.as_str() {
                "mlp.gate.weight" => fixture.router_w.clone(),
                "mlp.shared_expert.gate_proj.weight" => fixture.shared_gate.clone(),
                "mlp.shared_expert.up_proj.weight" => fixture.shared_up.clone(),
                "mlp.shared_expert.down_proj.weight" => fixture.shared_down.clone(),
                "mlp.shared_expert_gate.weight" => fixture.shared_expert_gate.clone(),
                role if role.ends_with("norm.weight") => vec![bf16::ONE; entry.bytes / 2],
                "linear_attn.A_log" | "linear_attn.dt_bias" => vec![bf16::ZERO; entry.bytes / 2],
                _ => (0..entry.bytes / 2)
                    .map(|index| {
                        let width = *entry.shape.last().unwrap();
                        bf16::from_f32(
                            (((index / width * 37 + index % width * 11 + 7) % 53) as f32 - 26.0)
                                / 4096.0,
                        )
                    })
                    .collect(),
            };
            ensure!(
                values.len() * 2 == entry.bytes,
                "hybrid fixture tensor extent: {name}"
            );
            tensors.insert(
                name.clone(),
                StoredTensor::Bf16(stream.clone_htod(&values)?),
            );
        }
        let bytes = manifest.resident_bytes;
        let catalog = WeightCatalog {
            #[cfg(feature = "cuda")]
            dense: None,
            linear_format: model_spec.linear_format,
            tensors,
            manifest,
            bytes,
            #[cfg(feature = "cuda")]
            upload_content_sha256: None,
            upload_hashed_bytes: None,
            load_wall_ms: 0.0,
        };
        let plan = crate::linear_plan::LayerDequantPlan::from_manifest(
            &catalog.manifest,
            0,
            model_spec.linear_format,
        )?;
        let mut linear = LayerLinearPlan::bind(&plan, &catalog, None, stream)?;
        let ffn = BoundFfn::bind(&model_spec, &mut linear)?;
        let weights = Weights {
            embed: Some(
                stream.clone_htod(
                    &(0..32 * d.hidden)
                        .map(|index| {
                            let (token, column) = (index / d.hidden, index % d.hidden);
                            if column < 8 {
                                if token % 8 == column {
                                    bf16::ONE
                                } else {
                                    bf16::ZERO
                                }
                            } else {
                                bf16::from_f32(
                                    (((token * 37 + column * 11) % 53) as f32 - 26.0) / 64.0,
                                )
                            }
                        })
                        .collect::<Vec<_>>(),
                )?,
            ),
            norm: Some(stream.clone_htod(&vec![bf16::ONE; d.hidden])?),
            head: stream.clone_htod(
                &(0..32 * d.hidden)
                    .map(|index| {
                        bf16::from_f32(
                            (((index / d.hidden * 17 + index % d.hidden * 7) % 61) as f32 - 30.0)
                                / 128.0,
                        )
                    })
                    .collect::<Vec<_>>(),
            )?,
            bytes: 0,
            layers: Vec::new(),
        };
        let ops = Ops::new(&device.ctx)?;
        let closure = Rc::new(super::super::super::closure::Closure::new(
            true,
            "MoE layer gate".into(),
        ));
        #[cfg(feature = "cuda")]
        let blas = Blas::new(&device, &ops, runtime, closure.clone())?;
        #[cfg(feature = "rocm")]
        let blas = Blas::new(stream.clone())?;
        let state = super::super::super::state::StateStore::new(&model_spec, 32, stream)?;
        let capture_checkpoint = Some(super::super::super::state::StateCheckpoint::new(
            &state,
            &model_spec,
            1,
            stream,
        )?);
        let peer_tail_elems = if peer.is_some() {
            PeerSendLayout::tail_bf16_elems(
                decode_rows,
                geometry.hidden as usize,
                geometry.top_k as usize,
            )?
        } else {
            0
        };
        let mut work =
            Work::new_with_peer_tail(stream, &model_spec, workspace_rows, peer_tail_elems)?;
        stream.memcpy_htod(&[1i32], &mut work.length)?;
        let flash = Some(Flash::new(
            stream.clone(),
            workspace_rows,
            model_spec.q_heads as usize,
            model_spec.kv_heads as usize,
            model_spec.head_dim as usize,
        )?);
        let (send_ptr, _send_guard) = work.n.device_ptr(stream);
        let send_cap_bytes = work.n.len() * size_of::<bf16>();
        #[cfg(feature = "cuda")]
        drop(_send_guard);
        let moe = Some(Decode::new(
            &device,
            DecodeOptions {
                peer: peer.as_ref().map(|peer| PeerBind {
                    device: peer,
                    gpu0_layers: capacity.gpu0_layers,
                    send_ptr,
                    send_cap_bytes,
                }),
                geometry,
                residual_output: model_spec.architecture != crate::model_spec::Architecture::Flint,
                max_rows: decode_rows,
                workers: capacity.worker_count,
                cpu_bounds,
                participants: crate::experts::plan::ParticipantTable::uniform(
                    u32::try_from(cpu_bounds.0)?,
                    u32::try_from(cpu_bounds.1)?,
                    capacity.worker_count,
                )?,
                format: model_spec
                    .expert_format
                    .context("MoE expert_format missing")?,
                shared_intermediate: model_spec
                    .moe
                    .context("MoE shared geometry missing")?
                    .shared_intermediate,
                swiglu_rounding: model_spec.swiglu_rounding(),
                swiglu_limit: model_spec.swiglu_limit(),
                routed_weight: model_spec.routed_weight_boundary(),
            },
        )?);
        let moe_prefill = if capacity.prefill_staging_slots > 0 {
            let reserves = crate::experts::plan::Reserves {
                device_fixed_bytes: 0,
                host_fixed_bytes: 0,
                residual_row_bytes: u64::from(geometry.hidden) * 2,
                normalized_row_bytes: u64::from(geometry.hidden) * 2,
                contribution_row_bytes: u64::from(geometry.top_k) * u64::from(geometry.hidden) * 2,
                done_row_bytes: u64::from(geometry.top_k) * 4,
                route_row_bytes: u64::from(geometry.top_k) * 10,
                prefill_staging_slots: capacity.prefill_staging_slots,
                disk_ring_slots: 0,
            };
            Some(Prefill::new(
                (&device, peer.as_ref()),
                &mut slots,
                capacity,
                reserves,
                service,
                (1_000_000_000.0, 1_000_000_000.0, 2, Vec::new()),
                (
                    model_spec
                        .moe
                        .context("MoE shared geometry missing")?
                        .shared_intermediate,
                    workspace_rows,
                    8,
                    (capacity.wave_rows, capacity.wave_rows),
                    (
                        model_spec.swiglu_rounding(),
                        model_spec.routed_weight_boundary(),
                    ),
                ),
            )?)
        } else {
            None
        };
        device.finish_upload()?;
        if let Some(peer) = &peer {
            peer.finish_upload()?;
            device.ctx.bind_to_thread()?;
        }
        let mut engine = Self {
            hidden_capture: None,
            capture_fail_after_layer: None,
            capture_fail_stage: None,
            #[cfg(test)]
            flint_window_capture: None,
            qwen: None,
            deepseek_mixer: None,
            deepseek_ffn: None,
            deepseek_engram: None,
            recurrent: None,
            diffusion: None,
            hc: None,
            glm_dsa: None,
            moe,
            moe_slots: Some(slots),
            moe_prefill,
            moe_test_ffn: Some(ffn),
            #[cfg(test)]
            flint_legacy_overflow_prefill: false,
            flint_frozen_decode_length: false,
            #[cfg(feature = "cuda")]
            mtp: None,
            #[cfg(feature = "cuda")]
            mtp_state: None,
            #[cfg(feature = "cuda")]
            mtp_epoch: 0,
            images: None,
            qwen_device_prefix: false,
            forward_rows: (1..=workspace_rows).collect(),
            graphs: Vec::new(),
            closure,
            poisoned: false,
            spec_budget: SpecBudget::Chain,
            spec_graph: false,
            tree_builder: TreeBuilder::Serial,
            compare_builders: false,
            profile_rounds: false,
            runtime: runtime.to_owned(),
            spec: None,
            graph: None,
            weights,
            state,
            capture_checkpoint,
            work,
            ops,
            blas,
            flash,
            decode: None,
            logits_f32: None,
            device,
            config,
            model_spec,
            capacity: 32,
            chunk: 8,
            workspace_rows,
            trace_layer: 0,
            position: 0,
            #[cfg(feature = "cuda")]
            prefix_identity: None,
            #[cfg(feature = "cuda")]
            prefix_identity_load_ms: 0.0,
            logits_cache: PrefixLogitsCache::default(),
            committed_prefix_ids: Vec::new(),
            pending_token: None,
            resident_ids_len: 0,
            last_input_kind: None,
        };
        let scratch = engine.device.stream.alloc_zeros(qwen::Runtime::workspace(
            &engine.model_spec,
            workspace_rows,
            32,
            #[cfg(feature = "cuda")]
            crate::gdn_plan::GdnPolicy::for_device(&engine.device.info),
            #[cfg(feature = "rocm")]
            crate::gdn_plan::GdnPolicy::P2,
            8,
            Some(&engine.device.stream),
        )?)?;
        engine.qwen = Some(qwen::Runtime::new(
            &engine,
            runtime,
            catalog,
            None,
            scratch,
            None,
            (8, false),
        )?);
        #[cfg(feature = "rocm")]
        engine.prepare_hip_fixture()?;
        for rows in 1..=workspace_rows {
            engine.prepare_qwen(rows)?;
            engine.prepare_prefill_rows(rows)?;
        }
        engine.device.finish_upload()?;
        Ok(engine)
    }

    pub(crate) fn run_layer_from_reference(
        &mut self,
        layer: usize,
        hidden_in: &[bf16],
        pos_axes: [&[i32]; 3],
        trace: &mut Trace,
    ) -> Result<()> {
        let hidden = self.model_spec.hidden as usize;
        ensure!(
            layer < self.model_spec.layers.len(),
            "layer injection layer={layer} outside layers={}",
            self.model_spec.layers.len()
        );
        ensure!(
            hidden_in.len().is_multiple_of(hidden),
            "layer injection hidden length={} width={hidden}",
            hidden_in.len()
        );
        let rows = hidden_in.len() / hidden;
        ensure!(
            rows > 0 && rows <= self.workspace_rows && pos_axes.iter().all(|a| a.len() == rows),
            "layer injection rows={rows} workspace={} axes={:?}",
            self.workspace_rows,
            pos_axes.map(<[i32]>::len)
        );
        self.inject_moe_test_positions(pos_axes)?;
        let stream = &self.device.stream;
        stream.memcpy_htod(hidden_in, &mut self.work.x.slice_mut(..hidden_in.len()))?;
        ensure!(
            self.qwen.is_some(),
            "layer injection requires the hybrid runtime"
        );
        self.moe_slots
            .as_mut()
            .context("injected layer has no residency")?
            .begin_token()?;
        self.run_layer(
            layer,
            crate::backend::Step {
                rows,
                input: crate::backend::Input::Ids,
                mask: crate::backend::Mask::Causal,
                head: Head::None,
            },
            &mut Some(Target {
                trace,
                prefix: "layer",
                skip: 0,
            }),
        )?;
        self.moe_slots
            .as_mut()
            .context("injected layer lost residency")?
            .finish_token()
    }

    pub(crate) fn record_test_stats(&self, trace: &mut Trace) -> Result<()> {
        trace.add(
            "stats.vision_encode_calls".into(),
            vec![1],
            vec![self.qwen.as_ref().map_or(0, |q| q.vision_encode_calls) as f32],
        )?;
        let stats = self
            .moe
            .as_ref()
            .context("MoE statistics executor missing")?
            .stats;
        for (name, count) in [
            "route_calls",
            "gpu_resident_hits",
            "cpu_worker_misses",
            "cpu_pool_calls",
            "combine_calls",
        ]
        .into_iter()
        .zip(stats)
        {
            trace.add(format!("stats.{name}"), vec![1], vec![count as f32])?;
        }
        self.moe_slots
            .as_ref()
            .context("MoE residency statistics missing")?
            .trace(trace)?;
        Ok(())
    }
}

pub(super) fn graph_state(engine: &Engine) -> Result<Vec<u32>> {
    let stream = &engine.device.stream;
    let mut values = vec![engine.position as u32];
    for kv in &engine.state.kv {
        for data in [&kv.k, &kv.v] {
            values.extend(
                stream
                    .clone_dtoh(data)?
                    .into_iter()
                    .map(|v| u32::from(v.to_bits())),
            );
        }
    }
    for state in &engine.state.gdn {
        for data in [&state.s, &state.history] {
            values.extend(stream.clone_dtoh(data)?.into_iter().map(f32::to_bits));
        }
    }
    if let Some(sequence) = &engine.state.sequence {
        values.extend(stream.clone_dtoh(sequence)?.into_iter().map(|v| v as u32));
    }
    values.extend(stream.clone_dtoh(&engine.work.token)?);
    values.extend(
        stream
            .clone_dtoh(&engine.work.position)?
            .into_iter()
            .map(|v| v as u32),
    );
    values.extend(
        stream
            .clone_dtoh(&engine.work.length)?
            .into_iter()
            .map(|v| v as u32),
    );
    Ok(values)
}

fn graph_gate(format: crate::model_spec::ExpertFormat) -> Result<()> {
    let runtime = PathBuf::from(
        std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("graph gate runtime missing")?,
    );
    let fixture = tier_fixture(3, 8);
    let geometry = ExpertGeometry::from_model(&hybrid_spec()?)?;
    let frequency = vec![vec![1.0; 16]; 2];
    let mut defects = 0;
    let mut control = false;
    for (resident, pinned, engines, cap, backed) in [
        (true, false, 1, 0, false),
        (false, false, 1, 2, false),
        (false, true, 2, 2, false),
        (false, false, 2, 0, false),
        (false, false, 1, 2, true),
        (false, true, 2, 2, true),
    ] {
        let mut capacity = tier_capacity(geometry, pinned, format)?;
        if cap == 0 {
            capacity.spare_blocks = 1;
            capacity.host_expert_bytes = 9 * capacity.slot_bytes();
        }
        if backed {
            capacity.host_blocks = geometry.layers * geometry.experts;
            capacity.spare_blocks = 0;
            capacity.host_expert_bytes =
                u64::from(capacity.host_blocks + capacity.spare_blocks) * capacity.slot_bytes();
            capacity.disk_bytes = 0;
            capacity.disk_workers = 0;
        }
        let mut pair = (0..2)
            .map(|arm| {
                let directory = runtime.join(format!(
                    "graphs-{resident}-{pinned}-{engines}-{cap}-{backed}-{arm}"
                ));
                if resident {
                    Engine::load_moe_test(&fixture, &directory, 8, format)
                } else {
                    Engine::load_moe_placed(
                        &fixture,
                        &directory,
                        &capacity,
                        &[vec![0, 1, 2, 3], vec![0, 1, 2, 3]],
                        crate::experts::plan::Admission {
                            budget_bytes_per_token: cap * capacity.slot_bytes(),
                        },
                        engines,
                        8,
                    )
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let mut candidate = pair.pop().context("graph candidate missing")?;
        let mut reference = pair.pop().context("graph reference missing")?;
        for engine in [&mut reference, &mut candidate] {
            engine.reset()?;
            engine
                .moe_slots
                .as_mut()
                .context("graph slots missing")?
                .begin_request(&frequency)?;
            engine.closure.freeze_fixed()?;
        }
        candidate.declare_graphs()?;
        let before = graph_state(&candidate)?;
        candidate.capture()?;
        let after = graph_state(&candidate)?;
        let capture_mismatches = before.iter().zip(&after).filter(|(a, b)| a != b).count()
            + before.len().abs_diff(after.len());
        defects += usize::from(capture_mismatches != 0);
        let expected_graphs = if resident { 1 } else { 6 };
        ensure!(
            candidate.graphs.len() == expected_graphs,
            "Runner FAIL graph declaration count"
        );
        candidate.closure.expect_captures(expected_graphs)?;
        candidate.check_graphs()?;
        reference.closure.ready(Instant::now())?;
        candidate.closure.ready(Instant::now())?;
        println!(
            "{} moe_graph_capture resident={resident} pinned={pinned} engines={engines} admission_cap={cap} backed={backed} graphs={expected_graphs} state_bit_mismatches={capture_mismatches}",
            if capture_mismatches == 0 {
                "PASS"
            } else {
                "DEFECT"
            }
        );
        for request in 0..2 {
            if request > 0 {
                for engine in [&mut reference, &mut candidate] {
                    engine.reset()?;
                    engine
                        .moe_slots
                        .as_mut()
                        .context("graph reset slots missing")?
                        .begin_request(&frequency)?;
                }
            }
            for step in 0..6 {
                let token = (3 + step * 7 + request * 11) % 32;
                reference.set_token(token)?;
                candidate.set_token(token)?;
                {
                    let _range = super::super::super::profile::Range::new(
                        cfg!(feature = "cuda"),
                        c"moe_graph_reference",
                    );
                    reference.decode(false, None)?;
                }
                {
                    let _range = super::super::super::profile::Range::new(
                        cfg!(feature = "cuda"),
                        c"moe_graph_candidate",
                    );
                    candidate.decode(true, None)?;
                }
                let a = reference.decode_logits()?;
                let b = candidate.decode_logits()?;
                ensure!(
                    a.iter().chain(&b).all(|v| v.is_finite()),
                    "Runner FAIL graph comparison is nonfinite"
                );
                let logits = a
                    .iter()
                    .zip(&b)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let a = graph_state(&reference)?;
                let b = graph_state(&candidate)?;
                let state =
                    a.iter().zip(&b).filter(|(a, b)| a != b).count() + a.len().abs_diff(b.len());
                let mut a = Trace::default();
                let mut b = Trace::default();
                reference.record_test_stats(&mut a)?;
                candidate.record_test_stats(&mut b)?;
                let mut placement = 0;
                for key in ["residency.tier", "residency.generation"] {
                    let (a, b) = (a.values(key)?, b.values(key)?);
                    placement += a
                        .iter()
                        .zip(b)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                }
                let mismatch = logits + state + placement;
                defects += usize::from(mismatch != 0);
                println!(
                    "{} moe_graph_decode resident={resident} pinned={pinned} engines={engines} admission_cap={cap} backed={backed} request={request} step={step} logits_bit_mismatches={logits} state_bit_mismatches={state} placement_mismatches={placement} graph_launches={expected_graphs}",
                    if mismatch == 0 { "PASS" } else { "DEFECT" }
                );
            }
        }
        if !resident && !control {
            for engine in [&mut reference, &mut candidate] {
                engine.reset()?;
                engine
                    .moe_slots
                    .as_mut()
                    .context("graph control slots missing")?
                    .begin_request(&frequency)?;
                engine.set_token(3)?;
            }
            reference.decode(false, None)?;
            {
                let segments = &mut candidate
                    .moe
                    .as_mut()
                    .context("control executor missing")?
                    .segments;
                let (first, tail) = segments.split_at_mut(1);
                std::mem::swap(&mut first[0].join, &mut tail[0].join);
            }
            candidate.decode(true, None)?;
            let a = reference.decode_logits()?;
            let b = candidate.decode_logits()?;
            ensure!(
                a.iter().chain(&b).all(|v| v.is_finite()),
                "Runner FAIL wrong-join control is nonfinite"
            );
            let mismatches = a
                .iter()
                .zip(&b)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            ensure!(
                mismatches > 0,
                "Runner FAIL wrong-join control was not detected"
            );
            println!("PASS moe_graph_control wrong_join_logits_bit_mismatches={mismatches}");
            control = true;
        }
    }
    ensure!(control, "Runner FAIL graph control did not execute");
    ensure!(defects == 0, "DEFECT MoE graph cells={defects}");
    println!(
        "PASS moe_graph_gate format={format:?} cases=6 requests=2 steps=6 bitwise=true control=wrong_join backing=partial,complete"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA and FLYINGFISH_RUNTIME_DIR"]
fn moe_graphs_match_eager_and_preserve_capture_state_gpu() -> Result<()> {
    graph_gate(crate::model_spec::ExpertFormat::Bf16)
}

#[test]
#[ignore = "requires CUDA and FLYINGFISH_RUNTIME_DIR"]
fn fp8_moe_graphs_match_eager_and_preserve_capture_state_gpu() -> Result<()> {
    graph_gate(crate::model_spec::ExpertFormat::Fp8W8a16)
}

fn step(rows: usize) -> crate::backend::Step<'static> {
    crate::backend::Step {
        rows,
        input: crate::backend::Input::Ids,
        mask: crate::backend::Mask::Causal,
        head: Head::None,
    }
}

pub(super) fn tier_fixture(variant: u32, rows: usize) -> MoeFixture {
    let mut values = fixture(MoeDims::new(rows, 16, 4, 128, 64), variant);
    values.router_w.fill(bf16::ZERO);
    for row in 0..rows {
        values.x[row * 128..row * 128 + rows].fill(bf16::ZERO);
        values.x[row * 128 + row] = bf16::ONE;
        let experts = routed_experts(row);
        for expert in 0..16 {
            let score = experts
                .iter()
                .position(|&id| id == expert)
                .map_or(-1.0 - expert as f32 / 32.0, |rank| 3.0 - rank as f32 / 2.0);
            values.router_w[expert as usize * 128 + row] = bf16::from_f32(score);
        }
    }
    values
}

fn routed_experts(row: usize) -> [u32; 4] {
    let offset = row as u32;
    [
        offset % 4,
        4 + (offset + 2) % 4,
        8 + (offset + 1) % 4,
        12 + (offset + 3) % 4,
    ]
}

pub(super) fn tier_service(workers: u32) -> Result<crate::experts::plan::ServiceTable> {
    crate::experts::plan::ServiceTable::new(
        "fixture",
        [1, 2, 5, 8, 16, 32, 64]
            .into_iter()
            .flat_map(|rows| {
                [1, 2, 4, 8]
                    .into_iter()
                    .map(move |experts| crate::experts::plan::ServiceCell {
                        workers,
                        rows,
                        experts,
                        seconds: f64::from(rows * experts) * 0.00002,
                    })
            })
            .collect(),
    )
}

pub(super) fn tier_capacity(
    geometry: ExpertGeometry,
    pinned: bool,
    format: crate::model_spec::ExpertFormat,
) -> Result<Capacity> {
    let layout = crate::experts::catalogue::ExpertLayout::new(format, &geometry)?;
    let slot = layout.slot_bytes();
    Ok(Capacity {
        layout,
        span_rows: 8,
        wave_rows: 8,
        device_expert_bytes: slot * 8,
        peer_expert_bytes: 0,
        gpu0_layers: geometry.layers,
        host_expert_bytes: slot * 10,
        slots_per_layer: vec![4, 4],
        host_blocks: 8,
        spare_blocks: 2,
        disk_bytes: slot * 16,
        worker_count: 2,
        disk_workers: 2,
        prefill_staging_slots: 0,
        pinned,
    })
}

fn tier_schedule() -> [(&'static [u32], &'static [usize]); 3] {
    [
        (&[0], &[1, 1, 1, 1, 1, 1, 2, 5, 8, 2]),
        (&[0], &[8, 2, 1]),
        (&[4, 5, 7, 8], &[8, 1]),
    ]
}

struct TierOracle {
    policy: crate::experts::runtime::Residency,
    reader: crate::experts::disk::DiskReader,
    catalogue: Arc<crate::experts::catalogue::ExpertCatalogue>,
}

impl TierOracle {
    fn new(fixture: &MoeFixture, directory: &Path, capacity: &Capacity) -> Result<Self> {
        use crate::experts::{
            arena::SlotLayout,
            disk::DiskReader,
            runtime::{LoadInputs, Residency},
        };
        let format = match &capacity.layout {
            crate::experts::catalogue::ExpertLayout::Bf16Fused { .. } => {
                crate::model_spec::ExpertFormat::Bf16
            }
            crate::experts::catalogue::ExpertLayout::Bf16Separate { .. } => {
                crate::model_spec::ExpertFormat::Bf16Separate
            }
            crate::experts::catalogue::ExpertLayout::Fp8W8a16(_) => {
                crate::model_spec::ExpertFormat::Fp8W8a16
            }
            crate::experts::catalogue::ExpertLayout::Fp8W8a16Block(_, block_scale) => {
                crate::model_spec::ExpertFormat::Fp8W8a16Block {
                    block: block_scale.block(),
                    scale_dtype: block_scale.dtype(),
                }
            }
            crate::experts::catalogue::ExpertLayout::Fp4E2m1Block(..) => {
                anyhow::bail!("E2M1 hybrid fixtures are not bound yet")
            }
        };
        let catalogue = native_fixture(fixture, directory, format)?;
        let geometry = catalogue.geometry();
        let layout = SlotLayout::new(geometry, capacity)?;
        let host = if layout.host_blocks() > 0 {
            Some(HostBlocks::new(
                layout.host_blocks(),
                usize::try_from(capacity.slot_bytes() / 2)?,
            )?)
        } else {
            None
        };
        let reader = DiskReader::open(&catalogue, directory, 1)?;
        let (policy, _) = Residency::load(LoadInputs {
            geometry,
            capacity: capacity.clone(),
            admission: crate::experts::plan::Admission {
                budget_bytes_per_token: 2 * capacity.slot_bytes(),
            },
            max_rows: fixture.dims.rows as u32,
            initial: &vec![vec![0, 1, 2, 3]; geometry.layers as usize],
            host: host.as_ref(),
            catalogue: &catalogue,
            reader: &reader,
        })?;
        std::fs::remove_file(directory.join("expert-fixture.bin"))?;
        Ok(Self {
            policy,
            reader,
            catalogue,
        })
    }

    fn routes(rows: usize) -> Vec<crate::experts::runtime::Route> {
        (0..rows)
            .flat_map(|row| {
                routed_experts(row)
                    .into_iter()
                    .enumerate()
                    .map(move |(rank, expert)| crate::experts::runtime::Route {
                        row: row as u32,
                        choice: (row * 4 + rank) as u32,
                        expert,
                        weight: bf16::from_f32(0.25).to_f32(),
                    })
            })
            .collect()
    }

    fn cpu_values(
        &self,
        layer: u32,
        input: &[bf16],
        reference: &Trace,
        active: &[u32],
    ) -> Result<Vec<bf16>> {
        use crate::cpu::expert_ffn::{
            Contribution, Fp8Contribution, Isa, Scratch, contribution, fp8_contribution,
            native_bytes,
        };
        let shape = Shape::new(128, 64)?;
        let mut scratch = Scratch::new(&shape);
        let native = ExpertWeights {
            gate_up: 0..16384,
            gate_up_stride: 128,
            down: 16384..24576,
            down_stride: 64,
        };
        let mut output = reference
            .values("layer.moe.contributions")?
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let ids = reference.values("layer.moe.ids")?;
        let weights = reference.values("layer.moe.weights")?;
        for (choice, &active) in active.iter().enumerate() {
            if active == 1 {
                continue;
            }
            let slot = self.reader.read_expert(
                layer,
                ids[choice] as u32,
                &self.catalogue,
                self.catalogue.slot_bytes(),
            )?;
            let values = self.reader.values(slot)?;
            let row = choice / 4;
            if let Some(slot_layout) = self.catalogue.layout().fp8_slot() {
                fp8_contribution(
                    Isa::detect(),
                    Fp8Contribution {
                        form: crate::cpu::expert_ffn::Form::Lane16,
                        shape: &shape,
                        slot: native_bytes(&values),
                        slot_layout,
                        hidden: &input[row * 128..(row + 1) * 128],
                        routing: bf16::from_f32(weights[choice]).to_f32(),

                        rounding: crate::model_spec::SwigluRounding::Twice,
                        limit: 0.0,
                        routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                    },
                    &mut output[choice * 128..(choice + 1) * 128],
                    &mut scratch,
                )?;
            } else {
                contribution(
                    Isa::detect(),
                    Contribution {
                        form: crate::cpu::expert_ffn::Form::Lane16,
                        shape: &shape,
                        arena: &values,
                        weights: &native,
                        hidden: &input[row * 128..(row + 1) * 128],
                        routing: bf16::from_f32(weights[choice]).to_f32(),

                        rounding: crate::model_spec::SwigluRounding::Twice,
                        limit: 0.0,
                        routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
                    },
                    &mut output[choice * 128..(choice + 1) * 128],
                    &mut scratch,
                )?;
            }
        }
        Ok(output)
    }

    fn compare_state(&self, engine: &Engine) -> Result<usize> {
        use crate::experts::plan::Tier;
        let mut actual = Trace::default();
        engine.record_test_stats(&mut actual)?;
        let tiers = actual.values("residency.tier")?;
        let generations = actual.values("residency.generation")?;
        let mut mismatches = 0;
        for layer in 0..2 {
            for expert in 0..16 {
                let tier = self.policy.ledger().tier(layer, expert)?;
                let value = match tier {
                    Tier::Gpu => 0.0,
                    Tier::Host => 1.0,
                    Tier::Disk => 2.0,
                };
                let generation = if tier == Tier::Gpu {
                    self.policy.generation(layer, expert)?
                } else {
                    0
                };
                let index = layer as usize * 16 + expert as usize;
                mismatches +=
                    usize::from(tiers[index] != value || generations[index] != generation as f32);
            }
        }
        Ok(mismatches)
    }
}

#[test]
fn decode_residency_fixture_covers_tiers_requests_and_cap() -> Result<()> {
    use crate::experts::runtime::{MoveSource, Route};
    let directory = std::env::temp_dir().join(format!(
        "flyingfish-decode-residency-{}",
        std::process::id()
    ));
    let fixture = tier_fixture(3, 8);
    let geometry = ExpertGeometry::from_model(&hybrid_spec()?)?;
    let mut oracle = TierOracle::new(
        &fixture,
        &directory,
        &tier_capacity(geometry, false, crate::model_spec::ExpertFormat::Bf16)?,
    )?;
    for layer in 0..2 {
        for expert in 0..16 {
            let slot = oracle.reader.read_expert(
                layer,
                expert,
                &oracle.catalogue,
                oracle.catalogue.slot_bytes(),
            )?;
            let values = oracle.reader.values(slot)?;
            let expert = expert as usize;
            let expected = fixture.gate_up[expert * 16384..(expert + 1) * 16384]
                .iter()
                .chain(&fixture.down[expert * 8192..(expert + 1) * 8192])
                .map(|&value| {
                    if layer == 0 {
                        value
                    } else {
                        bf16::from_f32(value.to_f32() * -0.75)
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(values.values(), expected);
        }
    }
    let mut event = 1;
    let (
        mut host,
        mut disk,
        mut admissions,
        mut host_moves,
        mut disk_moves,
        mut chains,
        mut compact,
    ) = (0, 0, 0, 0, 0, 0, false);
    let mut writebacks = 0;
    for (target, rows) in tier_schedule() {
        let mut frequency = vec![vec![0.0; 16]; 2];
        for layer in &mut frequency {
            for &expert in target {
                layer[expert as usize] = 1.0;
            }
        }
        let moves = oracle.policy.begin_request(&frequency)?.to_vec();
        for movement in moves {
            writebacks += usize::from(movement.dest_block.is_some());
            host_moves += usize::from(matches!(movement.source, MoveSource::Host(_)));
            disk_moves += usize::from(matches!(movement.source, MoveSource::Disk));
            chains += usize::from(movement.chain_slot.is_some());
            oracle.policy.record_copy(&[movement.slot], event)?;
            event += 1;
        }
        for &rows in rows {
            oracle.policy.begin_token()?;
            let routes: Vec<Route> = TierOracle::routes(rows);
            for layer in 0..2 {
                let dispatch = oracle.policy.dispatch(layer, &routes)?;
                host += dispatch.host.len();
                disk += dispatch.disk.len();
                compact |= dispatch.miss_rows.first().is_some_and(|&row| row > 0);
            }
            let updates = oracle.policy.finish_token()?.to_vec();
            assert!(updates.len() <= 2);
            admissions += updates.len();
            for update in updates {
                writebacks += usize::from(update.dest_block.is_some());
                host_moves += usize::from(update.from_host);
                disk_moves += usize::from(!update.from_host);
                oracle.policy.record_copy(&[update.slot], event)?;
                event += 1;
            }
        }
        oracle.policy.end_request()?;
    }
    assert!(
        host > 0
            && disk > 0
            && admissions > 0
            && host_moves > 0
            && disk_moves > 0
            && chains > 0
            && compact
    );
    drop(oracle);
    std::fs::remove_dir(directory)?;
    println!(
        "PASS decode_residency_cpu_preflight host_choices={host} disk_choices={disk} admissions={admissions} host_moves={host_moves} disk_moves={disk_moves} writebacks={writebacks} chains={chains} compact_nonzero_base={compact} requests=3 cap=2 GPU_executed=false"
    );
    Ok(())
}

fn direct(engine: &Engine, layer: usize, input: &[bf16], rows: usize) -> Result<Trace> {
    let (h, k, i) = (128, 4, 64);
    let stream = &engine.device.stream;
    let ctx = &engine.device.ctx;
    let BoundFfn::Moe {
        router,
        ref shared,
        shared_gate,
        ..
    } = engine.moe_test_ffn.clone().context("test FFN absent")?
    else {
        anyhow::bail!("dense test FFN");
    };
    let shared_gate = shared_gate.context("test FFN has no shared gate")?;
    let x = stream.clone_htod(input)?;
    let ids = stream.alloc_zeros::<i32>(rows * k)?;
    let weights = stream.alloc_zeros::<bf16>(rows * k)?;
    let active = stream.clone_htod(&vec![1u32; rows * k])?;
    let mut generation = stream.alloc_zeros::<u64>(rows * k)?;
    let contributions = stream.alloc_zeros::<bf16>(rows * k * h)?;
    let shared_out = stream.alloc_zeros::<bf16>(rows * h)?;
    let done = stream.alloc_zeros::<u32>(rows * k)?;
    let y = stream.alloc_zeros::<bf16>(rows * h)?;
    let scratch = stream.alloc_zeros::<u8>(decode_scratch_bytes(rows, k, i) as usize)?;
    let slots = engine
        .moe_slots
        .as_ref()
        .context("test slots absent")?
        .bindings(layer, stream)?;
    let xv = qwen::matrix(&x, stream, 0, rows, h)?;
    let idv = view(&ids, stream, 0, [rows as u32, k as u32], [k as u64, 1])?;
    let weightv = qwen::matrix(&weights, stream, 0, rows, k)?;
    let experts = router.shape[0];
    let logits = stream.alloc_zeros::<bf16>(rows * experts as usize)?;
    RouteOps::new(ctx, stream, experts, rows)?.route(
        xv,
        router,
        &mut TensorMut(idv),
        &mut TensorMut(weightv),
        &mut TensorMut(view(
            &logits,
            stream,
            0,
            [rows as u32, experts],
            [u64::from(experts), 1],
        )?),
        crate::model_spec::RouterScore::Softmax,
    )?;
    let routed = stream.clone_dtoh(&ids)?;
    let published = &engine
        .moe_slots
        .as_ref()
        .context("direct reference slots absent")?
        .published;
    let generations = routed
        .iter()
        .map(|&id| published[layer * 16 + id as usize])
        .collect::<Vec<_>>();
    stream.memcpy_htod(&generations, &mut generation)?;
    let mut shared_view = TensorMut(qwen::matrix(&shared_out, stream, 0, rows, h)?);
    let mut done_view = TensorMut(view(
        &done,
        stream,
        0,
        [rows as u32, k as u32],
        [k as u64, 1],
    )?);
    let input = MoeDecode {
        x: xv,
        ids: idv,
        active: view(&active, stream, 0, [rows as u32, k as u32], [k as u64, 1])?,
        weights: weightv,
        slots_gate_up: slots.gate_up,
        slots_down: slots.down,
        slot_valid: slots.valid,
        slot_generation: slots.generation,
        assignment_gen: view(
            &generation,
            stream,
            0,
            [rows as u32, k as u32],
            [k as u64, 1],
        )?,
        shared_gate: shared[0],
        shared_up: shared[1],
        shared_down: shared[2],
        shared_expert_gate: Some(shared_gate),
        contributions: view(
            &contributions,
            stream,
            0,
            [rows as u32, k as u32, h as u32],
            [(k * h) as u64, h as u64, 1],
        )?,
        shared_out: &mut shared_view,
        done: &mut done_view,
    };
    let op = DecodeOps::new(ctx)?;
    let mut y_view = TensorMut(qwen::matrix(&y, stream, 0, rows, h)?);
    let scratch = qwen::scratch(&scratch, stream);
    let geometry = crate::experts::catalogue::ExpertGeometry {
        first_layer: 0,
        layers: 1,
        experts: slots.valid.shape[0],
        hidden: h as u32,
        intermediate: i as u32,
        top_k: k as u32,
    };
    let layout = crate::experts::catalogue::ExpertLayout::new(
        engine
            .model_spec
            .expert_format
            .context("MoE reference expert format missing")?,
        &geometry,
    )?;
    op.decode(
        stream,
        &input,
        &mut y_view,
        &scratch,
        layout.fp8_slot(),
        i as u32,
    )?;
    stream.synchronize()?;
    let mut trace = Trace::default();
    trace.add(
        "layer.moe.ids".into(),
        vec![rows, k],
        stream
            .clone_dtoh(&ids)?
            .into_iter()
            .map(|v| v as f32)
            .collect(),
    )?;
    trace.add(
        "layer.moe.done".into(),
        vec![rows, k],
        stream
            .clone_dtoh(&done)?
            .into_iter()
            .map(|v| v as f32)
            .collect(),
    )?;
    for (name, data, shape) in [
        ("layer.moe.weights", weights, vec![rows, k]),
        ("layer.moe.contributions", contributions, vec![rows, k, h]),
        ("layer.moe.shared", shared_out, vec![rows, h]),
        ("layer.ffn", y, vec![rows, h]),
    ] {
        trace.add(
            name.into(),
            shape,
            stream
                .clone_dtoh(&data)?
                .into_iter()
                .map(bf16::to_f32)
                .collect(),
        )?;
    }
    Ok(trace)
}

fn compare(a: &Trace, b: &Trace) -> Result<usize> {
    let mut mismatch = 0;
    for name in [
        "layer.moe.ids",
        "layer.moe.weights",
        "layer.moe.contributions",
        "layer.moe.shared",
        "layer.moe.done",
        "layer.ffn",
    ] {
        let (a, b) = (a.values(name)?, b.values(name)?);
        ensure!(
            a.len() == b.len(),
            "reference/candidate trace extent differs for {name}"
        );
        ensure!(
            a.iter().chain(&b).all(|value| value.is_finite()),
            "Runner FAIL reference/candidate trace nonfinite for {name}"
        );
        mismatch += a
            .iter()
            .zip(&b)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
    }
    Ok(mismatch)
}

fn combine_expected(engine: &Engine, contributions: &[bf16], rows: usize) -> Result<Vec<bf16>> {
    use crate::backend::cuda::moe::combine::MoeCombine;

    let stream = &engine.device.stream;
    let decode = engine.moe.as_ref().context("reference decode missing")?;
    let x = qwen::matrix(&engine.work.n, stream, 0, rows, 128)?;
    let ffn = engine
        .moe_test_ffn
        .clone()
        .context("reference FFN missing")?;
    let expected_contributions = stream.clone_htod(contributions)?;
    let expected_out = stream.alloc_zeros::<bf16>(rows * 128)?;
    let expected_scale = stream.alloc_zeros::<bf16>(rows)?;
    let BoundFfn::Moe { shared_gate, .. } = ffn else {
        anyhow::bail!("dense CPU control FFN");
    };
    let shared_gate = shared_gate.context("CPU control FFN has no shared gate")?;
    let scale = view(&expected_scale, stream, 0, [rows as u32, 1], [1, 1])?;
    let combine = CombineOps::new(&engine.device.ctx)?;
    combine.shared_scale(stream, x, shared_gate, &mut TensorMut(scale))?;
    combine.combine(
        stream,
        &MoeCombine {
            ids: view(&decode.ids, stream, 0, [rows as u32, 4], [4, 1])?,
            final_active: view(&decode.final_active, stream, 0, [rows as u32, 4], [4, 1])?,
            done: view(&decode.done, stream, 0, [rows as u32, 4], [4, 1])?,
            contributions: view(
                &expected_contributions,
                stream,
                0,
                [rows as u32, 4, 128],
                [512, 128, 1],
            )?,
            shared_out: qwen::matrix(&decode.shared, stream, 0, rows, 128)?,
            x,
            shared_expert_gate: Some(shared_gate),
            scale: Some(TensorMut(scale)),
            y: TensorMut(qwen::matrix(&expected_out, stream, 0, rows, 128)?),
        },
    )?;
    stream.synchronize()?;
    let output = stream.clone_dtoh(&expected_out)?;
    Ok(output)
}

fn tier_cell(
    engine: &mut Engine,
    oracle: &mut TierOracle,
    reference: &Trace,
    layer: u32,
    input: &[bf16],
    rows: usize,
) -> Result<(usize, usize, usize)> {
    let ids = reference.values("layer.moe.ids")?;
    let weights = reference.values("layer.moe.weights")?;
    let routes = ids
        .iter()
        .zip(&weights)
        .enumerate()
        .map(
            |(choice, (&expert, &weight))| crate::experts::runtime::Route {
                row: choice as u32 / 4,
                choice: choice as u32,
                expert: expert as u32,
                weight: bf16::from_f32(weight).to_f32(),
            },
        )
        .collect::<Vec<_>>();
    ensure!(
        routes
            .iter()
            .all(|route| route.expert
                == routed_experts(route.row as usize)[route.choice as usize % 4]),
        "Runner FAIL tier reference ids differ from the frozen 0.5-margin logits"
    );
    let dispatch = oracle.policy.dispatch(layer, &routes)?.clone();
    let expected = oracle.cpu_values(layer, input, reference, &dispatch.active)?;
    engine
        .device
        .stream
        .memcpy_htod(input, &mut engine.work.n.slice_mut(..input.len()))?;
    let mut trace = Trace::default();
    engine.trace_layer = layer as usize;
    let residual = engine.device.stream.clone_dtoh(
        &engine
            .work
            .x
            .slice(..rows * engine.model_spec.hidden as usize),
    )?;
    engine.run_ffn(
        layer as usize,
        step(rows),
        &mut Some(Target {
            trace: &mut trace,
            prefix: "layer",
            skip: 0,
        }),
    )?;
    for (name, length) in [
        ("layer.moe.ids", rows * 4),
        ("layer.moe.weights", rows * 4),
        ("layer.moe.shared", rows * 128),
        ("layer.moe.contributions", rows * 4 * 128),
        ("layer.moe.done", rows * 4),
        ("layer.ffn", rows * 128),
    ] {
        let values = trace.values(name)?;
        let baseline = reference.values(name)?;
        ensure!(
            values.len() == length
                && baseline.len() == length
                && values
                    .iter()
                    .chain(&baseline)
                    .all(|value| value.is_finite()),
            "Runner FAIL tier trace {name} extent/nonfinite layer={layer} rows={rows} expected={length} actual={} reference={}",
            values.len(),
            baseline.len()
        );
    }
    let mut mismatches = 0;
    for name in ["layer.moe.ids", "layer.moe.weights", "layer.moe.shared"] {
        mismatches += trace
            .values(name)?
            .iter()
            .zip(reference.values(name)?)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
    }
    mismatches += trace
        .values("layer.moe.contributions")?
        .iter()
        .zip(&expected)
        .filter(|(a, b)| a.to_bits() != b.to_f32().to_bits())
        .count();
    mismatches += trace
        .values("layer.moe.done")?
        .iter()
        .filter(|&&done| done != 1.0)
        .count();
    let expected_out = combine_expected(engine, &expected, rows)?;
    let expected_out = expected_out
        .into_iter()
        .zip(&residual)
        .map(|(ffn, residual)| bf16::from_f32(ffn.to_f32() + residual.to_f32()))
        .collect::<Vec<_>>();
    mismatches += trace
        .values("layer.ffn")?
        .iter()
        .zip(expected_out)
        .filter(|(a, b)| a.to_bits() != b.to_f32().to_bits())
        .count();
    println!(
        "{} decode_residency_cell layer={layer} rows={rows} gpu_choices={} host_choices={} disk_choices={} compacted_rows={:?} bit_mismatches={mismatches}",
        if mismatches == 0 { "PASS" } else { "DEFECT" },
        dispatch.active.iter().sum::<u32>(),
        dispatch.host.len(),
        dispatch.disk.len(),
        dispatch.miss_rows
    );
    Ok((
        usize::from(mismatches != 0),
        dispatch.host.len(),
        dispatch.disk.len(),
    ))
}

fn tier_gate(
    variants: &[u32],
    cases: &[(bool, u32)],
    format: crate::model_spec::ExpertFormat,
) -> Result<()> {
    let runtime = PathBuf::from(
        std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("tier gate runtime missing")?,
    );
    let geometry = ExpertGeometry::from_model(&hybrid_spec()?)?;
    let mut defects = 0;
    for &variant in variants {
        let fixture = tier_fixture(variant, 8);
        let reference = Engine::load_moe_test(
            &fixture,
            &runtime.join(format!("reference-{variant}")),
            8,
            format,
        )?;
        let mut references = std::collections::BTreeMap::new();
        for layer in 0..2 {
            for rows in [1, 2, 5, 8] {
                let input = &fixture.x[..rows * 128];
                let trace = direct(&reference, layer, input, rows)?;
                ensure!(
                    compare(&trace, &direct(&reference, layer, input, rows)?)? == 0,
                    "Runner FAIL tier same-form reference repeat differs layer={layer} rows={rows}"
                );
                references.insert((layer, rows), trace);
            }
        }
        for (pinned, engines, backed) in cases
            .iter()
            .flat_map(|&(pinned, engines)| [(pinned, engines, false), (pinned, engines, true)])
        {
            let directory = runtime.join(format!("tier-{variant}-{pinned}-{engines}-{backed}"));
            let mut capacity = tier_capacity(geometry, pinned, format)?;
            if backed {
                capacity.host_blocks = geometry.layers * geometry.experts;
                capacity.spare_blocks = 0;
                capacity.host_expert_bytes =
                    u64::from(capacity.host_blocks + capacity.spare_blocks) * capacity.slot_bytes();
                capacity.disk_bytes = 0;
                capacity.disk_workers = 0;
            }
            let mut oracle = TierOracle::new(&fixture, &directory.join("oracle"), &capacity)?;
            let mut engine = Engine::load_moe_placed(
                &fixture,
                &directory,
                &capacity,
                &[vec![0, 1, 2, 3], vec![0, 1, 2, 3]],
                crate::experts::plan::Admission {
                    budget_bytes_per_token: 2 * capacity.slot_bytes(),
                },
                engines,
                8,
            )?;
            engine
                .moe_slots
                .as_mut()
                .context("tier slots missing")?
                .end_request()?;
            let (mut host_choices, mut disk_choices, mut admissions) = (0, 0, 0);
            let mut event = 1;
            let _range = super::super::super::profile::Range::new(
                cfg!(feature = "cuda"),
                c"engine_moe_residency",
            );
            for (request, (target, rows)) in tier_schedule().into_iter().enumerate() {
                let mut frequency = vec![vec![0.0; 16]; 2];
                for layer in &mut frequency {
                    for &expert in target {
                        layer[expert as usize] = 1.0;
                    }
                }
                let moves = oracle.policy.begin_request(&frequency)?.to_vec();
                for movement in &moves {
                    oracle.policy.record_copy(&[movement.slot], event)?;
                    event += 1;
                }
                {
                    let slots = engine
                        .moe_slots
                        .as_mut()
                        .context("tier request slots missing")?;
                    slots.delay_copies();
                    slots.begin_request(&frequency)?;
                }
                defects += oracle.compare_state(&engine)?;
                for &rows in rows {
                    oracle.policy.begin_token()?;
                    engine
                        .moe_slots
                        .as_mut()
                        .context("tier token slots missing")?
                        .begin_token()?;
                    for layer in 0..2 {
                        let (cell, host, disk) = tier_cell(
                            &mut engine,
                            &mut oracle,
                            &references[&(layer, rows)],
                            layer as u32,
                            &fixture.x[..rows * 128],
                            rows,
                        )?;
                        defects += cell;
                        host_choices += host;
                        disk_choices += disk;
                    }
                    let updates = oracle.policy.finish_token()?.to_vec();
                    ensure!(
                        updates.len() <= 2,
                        "Runner FAIL CPU oracle exceeds frozen admission cap"
                    );
                    ensure!(
                        !backed || updates.iter().all(|update| update.dest_block.is_none()),
                        "DEFECT complete backing requested a victim write-back"
                    );
                    admissions += updates.len();
                    for update in &updates {
                        oracle.policy.record_copy(&[update.slot], event)?;
                        event += 1;
                    }
                    let slots = engine
                        .moe_slots
                        .as_mut()
                        .context("tier admission slots missing")?;
                    slots.delay_copies();
                    slots.finish_token()?;
                    let mismatch = oracle.compare_state(&engine)?;
                    defects += mismatch;
                    println!(
                        "{} decode_residency_admission variant={variant} pinned={pinned} engines={engines} backed={backed} request={request} rows={rows} moves={} cap=2 state_mismatches={mismatch}",
                        if mismatch == 0 { "PASS" } else { "DEFECT" },
                        updates.len()
                    );
                }
                oracle.policy.end_request()?;
                engine
                    .moe_slots
                    .as_mut()
                    .context("tier request end slots missing")?
                    .end_request()?;
            }
            let mut stats = Trace::default();
            engine.record_test_stats(&mut stats)?;
            let pending = stats.values("stats.copy_wait_pending")?[0];
            ensure!(
                pending > 0.0,
                "Runner FAIL delayed copy control did not execute an outstanding first-read wait"
            );
            ensure!(
                host_choices > 0 && admissions > 0 && (disk_choices == 0) == backed,
                "Runner FAIL tier coverage host={host_choices} disk={disk_choices} admissions={admissions} backed={backed}"
            );
            println!(
                "PASS decode_residency_coverage variant={variant} pinned={pinned} engines={engines} backed={backed} host_choices={host_choices} disk_choices={disk_choices} admissions={admissions} pending_first_reads={pending} requests=3 end_request=true"
            );
        }
    }
    ensure!(defects == 0, "DEFECT decode residency cells={defects}");
    println!(
        "PASS decode_residency_gate variants={} cases={} rows=1,2,5,8 format={format:?} host=true disk=true admissions=true writeback=true requests=3 graphs=false backing=partial,complete",
        variants.len(),
        cases.len() * 2
    );
    Ok(())
}

#[test]
#[ignore = "authorized GPU window"]
fn decode_residency_smoke_gpu() -> Result<()> {
    tier_gate(&[3], &[(false, 1)], crate::model_spec::ExpertFormat::Bf16)
}

#[test]
#[ignore = "authorized GPU window"]
fn decode_residency_gpu() -> Result<()> {
    tier_gate(
        &[3, 7],
        &[(false, 1), (false, 2), (true, 1), (true, 2)],
        crate::model_spec::ExpertFormat::Bf16,
    )
}

#[test]
#[ignore = "GPU window and FLYINGFISH_RUNTIME_DIR required"]
fn fp8_decode_residency_gpu() -> Result<()> {
    tier_gate(
        &[3, 7],
        &[(false, 1), (true, 2)],
        crate::model_spec::ExpertFormat::Fp8W8a16,
    )
}

fn cpu_control(
    engine: &mut Engine,
    fixture: &MoeFixture,
    input: &[bf16],
    rows: usize,
    reference: &Trace,
) -> Result<usize> {
    use crate::cpu::expert_ffn::{Contribution, Isa, Scratch, contribution};
    let ffn = engine
        .moe_test_ffn
        .clone()
        .context("CPU control FFN missing")?;
    let slots = engine
        .moe_slots
        .as_ref()
        .context("CPU control slots missing")?;
    let host = HostBlocks::new(16, 24576)?;
    for expert in 0..16 {
        let mut claim = host.claim(expert)?;
        let values = claim.slice()?;
        let expert = expert as usize;
        values[..16384].copy_from_slice(&fixture.gate_up[expert * 16384..(expert + 1) * 16384]);
        values[16384..].copy_from_slice(&fixture.down[expert * 8192..(expert + 1) * 8192]);
    }
    let shape = Shape::new(128, 64)?;
    let mut scratch = Scratch::new(&shape);
    let local = ExpertWeights {
        gate_up: 0..16384,
        gate_up_stride: 128,
        down: 16384..24576,
        down_stride: 64,
    };
    let experts = (0..16)
        .map(|expert| {
            let range = host.block_range(expert)?;
            Ok(ExpertWeights {
                gate_up: range.start..range.start + 16384,
                gate_up_stride: 128,
                down: range.start + 16384..range.end,
                down_stride: 64,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let ids = reference.values("layer.moe.ids")?;
    let weights = reference.values("layer.moe.weights")?;
    let generations = ids
        .iter()
        .map(|&expert| slots.published[expert as usize])
        .collect::<Vec<_>>();
    let mut contributions = reference
        .values("layer.moe.contributions")?
        .into_iter()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    let mut active = vec![1u32; rows * 4];
    let mut assignments = Vec::new();
    let mut miss_rows = Vec::new();
    for row in (0..rows).step_by(2) {
        let choice = row * 4 + (row * 3 + 1) % 4;
        let expert = ids[choice] as u32;
        let lease = host.lease(&[expert])?;
        contribution(
            Isa::detect(),
            Contribution {
                form: crate::cpu::expert_ffn::Form::Lane16,
                shape: &shape,
                arena: lease.read(expert)?,
                weights: &local,
                hidden: &input[row * 128..(row + 1) * 128],
                routing: bf16::from_f32(weights[choice]).to_f32(),

                rounding: crate::model_spec::SwigluRounding::Twice,
                limit: 0.0,
                routed_weight: crate::model_spec::RoutedWeightBoundary::AfterDownBf16,
            },
            &mut contributions[choice * 128..(choice + 1) * 128],
            &mut scratch,
        )?;
        active[choice] = 0;
        assignments.push(Assignment {
            row: miss_rows.len() as u32,
            choice: choice as u32,
            expert,
            weight: bf16::from_f32(weights[choice]).to_f32(),
        });
        miss_rows.push(row as u32);
    }
    engine
        .device
        .stream
        .memset_zeros(&mut engine.work.out.slice_mut(..rows * 128))?;
    let decode = engine.moe.as_mut().context("CPU control decode missing")?;
    let stream = &engine.device.stream;
    let x = qwen::matrix(&engine.work.n, stream, 0, rows, 128)?;
    decode.route(&engine.device, 0, &ffn, x)?;
    let duplicate_route = decode.route(&engine.device, 0, &ffn, x).is_err();
    let alternate = stream.clone_htod(input)?;
    let mut probe_out = TensorMut(qwen::matrix(&engine.work.out, stream, 0, rows, 128)?);
    let dispatch = || Dispatch {
        active: &active,
        generation: &generations,
        host: Some(host.clone()),
        experts: &experts,
        assignments: &assignments,
        miss_rows: &miss_rows,
        host_dma: None,
        disk: &[],
        disk_cpu: None,
        layer: 0,
    };
    let different_input = decode
        .finish(
            &engine.device,
            ffn.clone(),
            slots.bindings(0, stream)?,
            dispatch(),
            (
                qwen::matrix(&alternate, stream, 0, rows, 128)?,
                &mut probe_out,
            ),
        )
        .is_err();
    super::residency::delay_stream(&decode.mailbox)?;
    decode.finish(
        &engine.device,
        ffn.clone(),
        slots.bindings(0, stream)?,
        Dispatch {
            active: &active,
            generation: &generations,
            host: Some(host.clone()),
            experts: &experts,
            assignments: &assignments,
            miss_rows: &miss_rows,
            host_dma: None,
            disk: &[],
            disk_cpu: None,
            layer: 0,
        },
        (
            x,
            &mut TensorMut(qwen::matrix(&engine.work.out, stream, 0, rows, 128)?),
        ),
    )?;
    ensure!(
        !decode.mailbox_ready.try_is_complete()?,
        "DEFECT decode CPU upload blocked the host until completion"
    );
    stream.synchronize()?;
    let duplicate_finish = decode
        .finish(
            &engine.device,
            ffn,
            slots.bindings(0, stream)?,
            dispatch(),
            (x, &mut probe_out),
        )
        .is_err();
    ensure!(
        duplicate_route && different_input && duplicate_finish,
        "DEFECT MoE phase controls duplicate_route={duplicate_route} different_input={different_input} duplicate_finish={duplicate_finish}"
    );
    let mut actual = Trace::default();
    decode.trace(stream, 0, &mut actual, "layer")?;
    let observed = actual.values("layer.moe.contributions")?;
    let mismatches = observed
        .iter()
        .zip(&contributions)
        .filter(|(a, b)| a.to_bits() != b.to_f32().to_bits())
        .count();
    let done_mismatches = actual
        .values("layer.moe.done")?
        .iter()
        .filter(|&&d| d != 1.0)
        .count();
    let expected = combine_expected(engine, &contributions, rows)?;
    let observed = stream.clone_dtoh(&engine.work.out.slice(..rows * 128))?;
    let output_mismatches = expected
        .iter()
        .zip(&observed)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let defects = mismatches + done_mismatches + output_mismatches;
    println!(
        "{} eager_moe_cpu_control rows={rows} cpu_choices={} compacted_rows={} contribution_mismatches={mismatches} done_mismatches={done_mismatches} output_mismatches={output_mismatches} upload_pending=true delay_ms=100",
        if defects == 0 { "PASS" } else { "DEFECT" },
        assignments.len(),
        miss_rows.len()
    );
    Ok(usize::from(defects != 0))
}

fn gate(row_cases: &[usize], variants: &[u32], cpu: bool) -> Result<()> {
    let runtime =
        std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("MoE gate runtime directory missing")?;
    let runtime = Path::new(&runtime);
    std::fs::create_dir_all(runtime)?;
    let mut defects = 0;
    for &variant in variants {
        let fixture = fixture(MoeDims::new(8, 16, 4, 128, 64), variant);
        let mut engine =
            Engine::load_moe_test(&fixture, runtime, 8, crate::model_spec::ExpertFormat::Bf16)?;
        for &rows in row_cases {
            let input = &fixture.x[..rows * 128];
            let reference = direct(&engine, 0, input, rows)?;
            ensure!(
                compare(&reference, &direct(&engine, 0, input, rows)?)? == 0,
                "Runner FAIL same-form reference repeat differs rows={rows} variant={variant}"
            );
            let mut trace = Trace::default();
            let stream = &engine.device.stream;
            stream.memset_zeros(&mut engine.work.x.slice_mut(..input.len()))?;
            stream.memcpy_htod(input, &mut engine.work.n.slice_mut(..input.len()))?;
            {
                engine
                    .moe_slots
                    .as_mut()
                    .context("gate slots absent")?
                    .begin_token()?;
                let _range = super::super::super::profile::Range::new(
                    cfg!(feature = "cuda"),
                    c"engine_moe_ffn",
                );
                engine.run_ffn(
                    0,
                    step(rows),
                    &mut Some(Target {
                        trace: &mut trace,
                        prefix: "layer",
                        skip: 0,
                    }),
                )?;
            }
            engine
                .moe_slots
                .as_mut()
                .context("gate slots lost")?
                .finish_token()?;
            let mismatches = compare(&reference, &trace)?;
            defects += usize::from(mismatches != 0);
            println!(
                "{} eager_moe_all_resident rows={rows} variant={variant} bit_mismatches={mismatches}",
                if mismatches == 0 { "PASS" } else { "DEFECT" }
            );
            if cpu {
                defects += cpu_control(&mut engine, &fixture, input, rows, &reference)?;
            }
        }
        let mut stats = Trace::default();
        engine.record_test_stats(&mut stats)?;
        let calls = row_cases.len() * if cpu { 2 } else { 1 };
        let cpu_choices = if cpu {
            row_cases.iter().map(|r| r.div_ceil(2)).sum::<usize>()
        } else {
            0
        };
        let gpu_choices =
            row_cases.iter().sum::<usize>() * 4 * if cpu { 2 } else { 1 } - cpu_choices;
        for (name, expected) in [
            ("route_calls", calls),
            ("combine_calls", calls),
            ("gpu_resident_hits", gpu_choices),
            ("cpu_worker_misses", cpu_choices),
            ("cpu_pool_calls", if cpu { row_cases.len() } else { 0 }),
        ] {
            ensure!(
                stats.values(&format!("stats.{name}"))? == [expected as f32],
                "DEFECT stats.{name} expected={expected}"
            );
        }
        let axes = [0, 1];
        let _range =
            super::super::super::profile::Range::new(cfg!(feature = "cuda"), c"engine_moe_hybrid");
        for layer in 0..2 {
            let mut trace = Trace::default();
            engine.trace_layer = layer;
            engine.run_layer_from_reference(layer, &fixture.x[..256], [&axes; 3], &mut trace)?;
            ensure!(
                trace.values("layer.ffn")?.len() == 256,
                "hybrid layer injection did not execute FFN layer={layer}"
            );
            ensure!(
                trace
                    .values("layer.debug.mixer_out")?
                    .iter()
                    .all(|value| value.is_finite()),
                "Runner FAIL hybrid layer mixer output nonfinite layer={layer}"
            );
            println!(
                "PASS eager_moe_hybrid_injection layer={layer} mixer={:?} rows=2 q_heads=16 kv_heads=2 head_dim=256 rotary_dim=64 hybrid_bound=true",
                engine.model_spec.layers[layer].mixer
            );
        }
        engine
            .moe_slots
            .as_mut()
            .context("gate slots lost at request end")?
            .end_request()?;
    }
    ensure!(defects == 0, "eager MoE candidate defects={defects}");
    println!(
        "PASS eager_moe_gate all_resident=true cpu_control={cpu} variants={} row_cases={} graphs=false",
        variants.len(),
        row_cases.len()
    );
    Ok(())
}

#[test]
#[ignore = "authorized GPU window"]
fn eager_moe_smoke_gpu() -> Result<()> {
    gate(&[2], &[3], false)
}

#[test]
#[ignore = "authorized GPU window"]
fn eager_moe_decode_gpu() -> Result<()> {
    gate(&[1, 2, 5, 8, 2], &[3, 7], true)
}

#[test]
fn dispatch_covers_original_choices_once_after_row_compaction() -> Result<()> {
    for rows in [1, 8, 16] {
        let choices = rows * 32;
        let mut active = vec![1; choices];
        active[choices - 1] = 0;
        let generations = vec![7; choices];
        let weights = vec![bf16::ONE; choices];
        let host = Some(HostBlocks::new(1, 32)?);
        let dispatch = Dispatch {
            active: &active,
            generation: &generations,
            host,
            experts: &[ExpertWeights {
                gate_up: 0..16,
                gate_up_stride: 4,
                down: 16..32,
                down_stride: 4,
            }],
            assignments: &[Assignment {
                row: 0,
                choice: (choices - 1) as u32,
                expert: 0,
                weight: 1.0,
            }],
            miss_rows: &[(rows - 1) as u32],
            host_dma: None,
            disk: &[],
            disk_cpu: None,
            layer: 0,
        };
        validate_dispatch(rows, 32, &weights, &dispatch)?;
        assert!(validate_dispatch(17, 32, &weights, &dispatch).is_err());
    }
    let mut active = [1u32; 20];
    active[1] = 0;
    active[9] = 0;
    let weights = (0..20)
        .map(|i| bf16::from_f32((i + 1) as f32 / 32.0))
        .collect::<Vec<_>>();
    let generations = [7u64; 20];
    let host = Some(HostBlocks::new(1, 32)?);
    let experts = [ExpertWeights {
        gate_up: 0..16,
        gate_up_stride: 4,
        down: 16..32,
        down_stride: 4,
    }];
    let valid = [
        Assignment {
            row: 0,
            choice: 1,
            expert: 0,
            weight: weights[1].to_f32(),
        },
        Assignment {
            row: 1,
            choice: 9,
            expert: 0,
            weight: weights[9].to_f32(),
        },
    ];
    let check = |active: &[u32], assignments: &[Assignment], rows: &[u32]| {
        validate_dispatch(
            5,
            4,
            &weights,
            &Dispatch {
                active,
                generation: &generations,
                host: host.clone(),
                experts: &experts,
                assignments,
                miss_rows: rows,
                host_dma: None,
                disk: &[],
                disk_cpu: None,
                layer: 0,
            },
        )
    };
    check(&active, &valid, &[0, 2])?;
    assert!(check(&active, &valid[..1], &[0, 2]).is_err());
    assert!(check(&active, &[valid[0], valid[0], valid[1]], &[0, 2]).is_err());
    let mut bad = valid;
    bad[1].row = 0;
    assert!(check(&active, &bad, &[0, 2]).is_err());
    bad = valid;
    bad[1].choice = 8;
    assert!(check(&active, &bad, &[0, 2]).is_err());
    bad = valid;
    bad[1].expert = 1;
    assert!(check(&active, &bad, &[0, 2]).is_err());
    assert!(check(&active, &valid, &[0, 5]).is_err());
    assert!(check(&active, &valid, &[2, 0]).is_err());
    active[1] = 1;
    assert!(check(&active, &valid, &[0, 2]).is_err());
    println!(
        "PASS eager_moe_dispatch_controls exact_cover=true compacted_rows=true original_choice=true bounds=true"
    );
    Ok(())
}

#[test]
fn injected_layers_bind_hybrid_mixer_geometry() -> Result<()> {
    use crate::model_spec::{Architecture, Mixer};
    let spec = hybrid_spec()?;
    let manifest = hybrid_manifest(&spec)?;
    assert_eq!(spec.architecture, Architecture::MiniCpmV47);
    assert_eq!(
        (spec.q_heads, spec.kv_heads, spec.head_dim, spec.rotary_dim),
        (16, 2, 256, 64)
    );
    let g = crate::op_contract::GdnSpec::from_model(&spec)?;
    assert_eq!(
        (g.hk, g.hv, g.dk, g.dv, g.conv_width),
        (16, 32, 128, 128, 4)
    );
    for (layer, mixer) in [Mixer::Gdn, Mixer::FullAttention].into_iter().enumerate() {
        let crate::model::LayerProgram::Standard(stages) =
            crate::model::layer_program(&spec, layer)?
        else {
            anyhow::bail!("hybrid spec must use the standard layer program");
        };
        assert!(matches!(
            stages[1],
            crate::model::LayerStage::Mixer(bound) if bound == mixer
        ));
        let plan = crate::linear_plan::LayerDequantPlan::from_manifest(
            &manifest,
            layer as u32,
            spec.linear_format,
        )?;
        assert_eq!(plan.bytes, 0);
        assert!(plan.slots.is_empty());
    }
    for (name, shape) in [
        ("0.linear_attn.in_proj_qkv.weight", [8192, 128]),
        ("0.linear_attn.out_proj.weight", [128, 4096]),
        ("1.self_attn.q_proj.weight", [8192, 128]),
        ("1.self_attn.k_proj.weight", [512, 128]),
        ("1.self_attn.o_proj.weight", [128, 4096]),
    ] {
        assert_eq!(
            manifest.entries[&format!("model.language_model.layers.{name}")].shape,
            shape
        );
    }
    assert!(qwen::Runtime::workspace(&spec, 8, 32, crate::gdn_plan::GdnPolicy::P2, 8, None)? > 0);
    let mut old = spec.clone();
    old.architecture = Architecture::Llama;
    assert!(hybrid_manifest(&old).is_err());
    old = spec.clone();
    old.head_dim = 128;
    old.q_width = old.q_heads * old.head_dim;
    assert!(hybrid_manifest(&old).is_err());
    println!(
        "PASS eager_moe_hybrid_preflight architecture=MiniCpmV47 layers=Gdn,FullAttention q16_kv2_d256_rotary64=true gdn_hk16_hv32=true legacy_rejected=true d128_rejected=true GPU_executed=false"
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA and FLYINGFISH_RUNTIME_DIR"]
fn split_combine_matches_single_gpu_and_detects_perturbed_routed_peer_gpu() -> Result<()> {
    let runtime = PathBuf::from(
        std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("split combine runtime missing")?,
    );
    let fixture = tier_fixture(3, 8);
    let geometry = ExpertGeometry::from_model(&hybrid_spec()?)?;
    let mut defects = 0;
    for format in [
        crate::model_spec::ExpertFormat::Bf16,
        crate::model_spec::ExpertFormat::Fp8W8a16,
    ] {
        let layout = crate::experts::catalogue::ExpertLayout::new(format, &geometry)?;
        let slot = layout.slot_bytes();
        let peer_cap = Capacity {
            layout,
            span_rows: 8,
            wave_rows: 8,
            device_expert_bytes: slot * 16,
            peer_expert_bytes: slot * 16,
            gpu0_layers: 1,
            host_expert_bytes: 0,
            slots_per_layer: vec![16, 16],
            host_blocks: 0,
            spare_blocks: 0,
            disk_bytes: 0,
            worker_count: 2,
            disk_workers: 0,
            prefill_staging_slots: 0,
            pinned: false,
        };
        let mut single = Engine::load_moe_test(
            &fixture,
            &runtime.join(format!("split-single-{format:?}")),
            8,
            format,
        )?;
        let all_experts = (0..16).collect::<Vec<_>>();
        let mut two_gpu = Engine::load_moe_placed(
            &fixture,
            &runtime.join(format!("split-peer-{format:?}")),
            &peer_cap,
            &[all_experts.clone(), all_experts],
            crate::experts::plan::Admission {
                budget_bytes_per_token: 0,
            },
            1,
            8,
        )?;
        let peer_freq = vec![vec![1.0; 16]; 2];
        for engine in [&mut single, &mut two_gpu] {
            engine.reset()?;
            engine
                .moe_slots
                .as_mut()
                .context("split combine slots missing")?
                .begin_request(&peer_freq)?;
            engine.closure.freeze_fixed()?;
            engine.declare_graphs()?;
            engine.capture()?;
            engine.closure.expect_captures(engine.graphs.len())?;
            engine.check_graphs()?;
            engine.closure.ready(Instant::now())?;
        }
        ensure!(
            single
                .moe_slots
                .as_ref()
                .context("single slots missing")?
                .all_resident()
                && single.graph.is_some()
                && !single.has_moe_segments()
                && single
                    .moe
                    .as_ref()
                    .context("single decode missing")?
                    .peer
                    .is_none(),
            "Runner FAIL single-GPU resident fixture invariant"
        );
        ensure!(
            two_gpu
                .moe_slots
                .as_ref()
                .context("two-GPU slots missing")?
                .all_resident()
                && two_gpu.graph.is_some()
                && !two_gpu.has_moe_segments()
                && two_gpu
                    .moe
                    .as_ref()
                    .context("two-GPU decode missing")?
                    .peer
                    .is_some(),
            "Runner FAIL two-GPU full residency graph invariant"
        );
        let run_seq = |engine: &mut Engine,
                       freq: &[Vec<f64>],
                       graph: bool|
         -> Result<(Vec<f32>, Vec<u32>, [u64; 5])> {
            engine.reset()?;
            engine
                .moe_slots
                .as_mut()
                .context("seq slots missing")?
                .begin_request(freq)?;
            let before = engine.moe.as_ref().context("seq decode missing")?.stats;
            for step in 0..4 {
                engine.set_token((step * 5 + 1) % 32)?;
                engine.decode(graph, None)?;
            }
            let after = engine.moe.as_ref().context("seq decode missing")?.stats;
            let mut delta = [0u64; 5];
            for i in 0..5 {
                delta[i] = after[i] - before[i];
            }
            Ok((engine.decode_logits()?, graph_state(engine)?, delta))
        };
        let (ref_logits, ref_state, single_eager_stats) = run_seq(&mut single, &peer_freq, false)?;
        let (single_graph_logits, single_graph_state, single_graph_stats) =
            run_seq(&mut single, &peer_freq, true)?;
        ensure!(
            single_eager_stats[1] == 32
                && single_eager_stats[2] == 0
                && single_graph_stats[1] == 32
                && single_graph_stats[2] == 0,
            "Runner FAIL single-GPU resident fixture stats eager={single_eager_stats:?} graph={single_graph_stats:?}"
        );
        let (peer_eager_logits, peer_eager_state, peer_eager_stats) =
            run_seq(&mut two_gpu, &peer_freq, false)?;
        let (peer_graph_logits, peer_graph_state, peer_graph_stats) =
            run_seq(&mut two_gpu, &peer_freq, true)?;
        ensure!(
            peer_eager_stats[1] == 32
                && peer_eager_stats[2] == 0
                && peer_graph_stats[1] == 32
                && peer_graph_stats[2] == 0,
            "Runner FAIL two-GPU full residency stats eager={peer_eager_stats:?} graph={peer_graph_stats:?}"
        );
        for (label, logits, state) in [
            ("single_graph", &single_graph_logits, &single_graph_state),
            ("two_gpu_eager", &peer_eager_logits, &peer_eager_state),
            ("two_gpu_graph", &peer_graph_logits, &peer_graph_state),
        ] {
            let logit_diff = ref_logits
                .iter()
                .zip(logits)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let state_diff = ref_state.iter().zip(state).filter(|(a, b)| a != b).count()
                + ref_state.len().abs_diff(state.len());
            defects += usize::from(logit_diff != 0 || state_diff != 0);
            println!(
                "{} split_combine_parity format={format:?} arm={label} logit_bit_mismatches={logit_diff} state_bit_mismatches={state_diff}",
                if logit_diff == 0 && state_diff == 0 {
                    "PASS"
                } else {
                    "DEFECT"
                }
            );
        }
        two_gpu
            .moe
            .as_mut()
            .context("two-GPU decode missing")?
            .peer
            .as_mut()
            .context("two-GPU peer missing")?
            .perturb_word = Some((0, 0x7bff));
        two_gpu.clear_moe_graphs();
        two_gpu.closure = Rc::new(super::super::super::closure::Closure::new(
            true,
            "MoE split combine".into(),
        ));
        two_gpu.reset()?;
        two_gpu
            .moe_slots
            .as_mut()
            .context("two-GPU slots missing")?
            .begin_request(&peer_freq)?;
        two_gpu.closure.freeze_fixed()?;
        two_gpu.declare_graphs()?;
        two_gpu.capture()?;
        two_gpu.closure.expect_captures(two_gpu.graphs.len())?;
        two_gpu.check_graphs()?;
        two_gpu.closure.ready(Instant::now())?;
        for (label, graph) in [("perturbed_eager", false), ("perturbed_graph", true)] {
            let (logits, state, _) = run_seq(&mut two_gpu, &peer_freq, graph)?;
            let logit_diff = ref_logits
                .iter()
                .zip(&logits)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let state_diff = ref_state.iter().zip(&state).filter(|(a, b)| a != b).count();
            ensure!(
                logit_diff > 0 && state_diff > 0,
                "Runner FAIL planted routed_peer perturbation was not detected in {label} format={format:?}"
            );
            println!(
                "PASS split_combine_control format={format:?} arm={label} logit_bit_mismatches={logit_diff} state_bit_mismatches={state_diff}"
            );
        }
        for fault in [
            crate::backend::cuda::engine::CaptureFault::Begin,
            crate::backend::cuda::engine::CaptureFault::End,
            crate::backend::cuda::engine::CaptureFault::Instantiate,
        ] {
            two_gpu.poisoned = false;
            two_gpu.clear_moe_graphs();
            two_gpu.closure = Rc::new(super::super::super::closure::Closure::new(
                true,
                "MoE split combine fault".into(),
            ));
            two_gpu.reset()?;
            two_gpu
                .moe_slots
                .as_mut()
                .context("two-GPU slots missing")?
                .begin_request(&peer_freq)?;
            two_gpu.closure.freeze_fixed()?;
            two_gpu.declare_graphs()?;
            two_gpu.capture_fail_stage = Some(fault);
            let err = two_gpu
                .capture()
                .expect_err("injected multi-device capture fault did not error");
            two_gpu.capture_fail_stage = None;
            let chain = format!("{err:#}");
            ensure!(
                chain.contains("MoE capture transaction failed")
                    && chain.contains("injected capture")
                    && two_gpu.graph.is_none()
                    && !two_gpu.has_moe_segments()
                    && !two_gpu.has_moe_peer_tail(),
                "Runner FAIL multi-device capture fault fell back or lost context: {chain}"
            );
            println!(
                "PASS multi_device_capture_no_fallback format={format:?} fault={fault:?} segments=false"
            );
        }

        let single_partial_cap = Capacity {
            layout: crate::experts::catalogue::ExpertLayout::new(format, &geometry)?,
            span_rows: 8,
            wave_rows: 8,
            device_expert_bytes: slot * 24,
            peer_expert_bytes: 0,
            gpu0_layers: 2,
            host_expert_bytes: slot * 32,
            slots_per_layer: vec![8, 16],
            host_blocks: 32,
            spare_blocks: 0,
            disk_bytes: 0,
            worker_count: 2,
            disk_workers: 0,
            prefill_staging_slots: 0,
            pinned: true,
        };
        let peer_partial_cap = Capacity {
            layout: crate::experts::catalogue::ExpertLayout::new(format, &geometry)?,
            span_rows: 8,
            wave_rows: 8,
            device_expert_bytes: slot * 8,
            peer_expert_bytes: slot * 16,
            gpu0_layers: 1,
            host_expert_bytes: slot * 32,
            slots_per_layer: vec![8, 16],
            host_blocks: 32,
            spare_blocks: 0,
            disk_bytes: 0,
            worker_count: 2,
            disk_workers: 0,
            prefill_staging_slots: 0,
            pinned: true,
        };
        let partial_initial = [(0..8).collect::<Vec<_>>(), (0..16).collect::<Vec<_>>()];
        let partial_freq = vec![
            (0..16)
                .map(|id| if id < 8 { 10.0 } else { 1.0 })
                .collect::<Vec<_>>(),
            vec![1.0; 16],
        ];
        let mut single_partial = Engine::load_moe_placed(
            &fixture,
            &runtime.join(format!("split-single-partial-{format:?}")),
            &single_partial_cap,
            &partial_initial,
            crate::experts::plan::Admission {
                budget_bytes_per_token: 0,
            },
            1,
            8,
        )?;
        let mut two_partial = Engine::load_moe_placed(
            &fixture,
            &runtime.join(format!("split-peer-partial-{format:?}")),
            &peer_partial_cap,
            &partial_initial,
            crate::experts::plan::Admission {
                budget_bytes_per_token: 0,
            },
            1,
            8,
        )?;
        for engine in [&mut single_partial, &mut two_partial] {
            engine.reset()?;
            engine
                .moe_slots
                .as_mut()
                .context("partial split combine slots missing")?
                .begin_request(&partial_freq)?;
            engine.closure.freeze_fixed()?;
            engine.declare_graphs()?;
            engine.capture()?;
            engine.closure.expect_captures(engine.graphs.len())?;
            engine.check_graphs()?;
            engine.closure.ready(Instant::now())?;
        }
        ensure!(
            !single_partial
                .moe_slots
                .as_ref()
                .context("single partial slots missing")?
                .all_resident()
                && single_partial.graph.is_none()
                && single_partial.has_moe_segments()
                && !single_partial.has_moe_peer_tail()
                && single_partial.graphs.len() == 6,
            "Runner FAIL single-GPU partial fixture invariant"
        );
        ensure!(
            !two_partial
                .moe_slots
                .as_ref()
                .context("two-GPU partial slots missing")?
                .all_resident()
                && two_partial.graph.is_none()
                && two_partial.has_moe_segments()
                && two_partial.has_moe_peer_tail()
                && two_partial.graphs.len() == 4
                && two_partial
                    .moe
                    .as_ref()
                    .context("two-GPU partial decode missing")?
                    .peer
                    .is_some(),
            "Runner FAIL two-GPU partial peer-tail graph invariant"
        );
        let (part_ref_logits, part_ref_state, part_single_eager_stats) =
            run_seq(&mut single_partial, &partial_freq, false)?;
        let (part_single_graph_logits, part_single_graph_state, part_single_graph_stats) =
            run_seq(&mut single_partial, &partial_freq, true)?;
        let (part_peer_eager_logits, part_peer_eager_state, part_peer_eager_stats) =
            run_seq(&mut two_partial, &partial_freq, false)?;
        let (part_peer_graph_logits, part_peer_graph_state, part_peer_graph_stats) =
            run_seq(&mut two_partial, &partial_freq, true)?;
        ensure!(
            part_single_eager_stats[1] == 24
                && part_single_eager_stats[2] == 8
                && part_single_graph_stats == part_single_eager_stats
                && part_peer_eager_stats == part_single_eager_stats
                && part_peer_graph_stats == part_single_eager_stats,
            "Runner FAIL partial peer-tail stats single_eager={part_single_eager_stats:?} single_graph={part_single_graph_stats:?} peer_eager={part_peer_eager_stats:?} peer_graph={part_peer_graph_stats:?}"
        );
        for (label, logits, state) in [
            (
                "partial_single_graph",
                &part_single_graph_logits,
                &part_single_graph_state,
            ),
            (
                "partial_two_gpu_eager",
                &part_peer_eager_logits,
                &part_peer_eager_state,
            ),
            (
                "partial_two_gpu_graph",
                &part_peer_graph_logits,
                &part_peer_graph_state,
            ),
        ] {
            let logit_diff = part_ref_logits
                .iter()
                .zip(logits)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let state_diff = part_ref_state
                .iter()
                .zip(state)
                .filter(|(a, b)| a != b)
                .count()
                + part_ref_state.len().abs_diff(state.len());
            defects += usize::from(logit_diff != 0 || state_diff != 0);
            println!(
                "{} split_combine_parity format={format:?} arm={label} logit_bit_mismatches={logit_diff} state_bit_mismatches={state_diff}",
                if logit_diff == 0 && state_diff == 0 {
                    "PASS"
                } else {
                    "DEFECT"
                }
            );
        }
        two_partial
            .moe
            .as_mut()
            .context("two-GPU partial decode missing")?
            .peer
            .as_mut()
            .context("two-GPU partial peer missing")?
            .perturb_word = Some((0, 0x7bff));
        two_partial.clear_moe_graphs();
        two_partial.closure = Rc::new(super::super::super::closure::Closure::new(
            true,
            "MoE split combine partial".into(),
        ));
        two_partial.reset()?;
        two_partial
            .moe_slots
            .as_mut()
            .context("two-GPU partial slots missing")?
            .begin_request(&partial_freq)?;
        two_partial.closure.freeze_fixed()?;
        two_partial.declare_graphs()?;
        two_partial.capture()?;
        two_partial
            .closure
            .expect_captures(two_partial.graphs.len())?;
        two_partial.check_graphs()?;
        two_partial.closure.ready(Instant::now())?;
        for (label, graph) in [
            ("partial_perturbed_eager", false),
            ("partial_perturbed_graph", true),
        ] {
            let (logits, state, _) = run_seq(&mut two_partial, &partial_freq, graph)?;
            let logit_diff = part_ref_logits
                .iter()
                .zip(&logits)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let state_diff = part_ref_state
                .iter()
                .zip(&state)
                .filter(|(a, b)| a != b)
                .count();
            ensure!(
                logit_diff > 0 && state_diff > 0,
                "Runner FAIL planted partial routed_peer perturbation was not detected in {label} format={format:?}"
            );
            println!(
                "PASS split_combine_control format={format:?} arm={label} logit_bit_mismatches={logit_diff} state_bit_mismatches={state_diff}"
            );
        }
        for fault in [
            crate::backend::cuda::engine::CaptureFault::Begin,
            crate::backend::cuda::engine::CaptureFault::End,
            crate::backend::cuda::engine::CaptureFault::Instantiate,
        ] {
            two_partial.poisoned = false;
            two_partial.clear_moe_graphs();
            two_partial.closure = Rc::new(super::super::super::closure::Closure::new(
                true,
                "MoE split combine partial fault".into(),
            ));
            two_partial.reset()?;
            two_partial
                .moe_slots
                .as_mut()
                .context("two-GPU partial slots missing")?
                .begin_request(&partial_freq)?;
            two_partial.closure.freeze_fixed()?;
            two_partial.declare_graphs()?;
            two_partial.capture_fail_stage = Some(fault);
            let err = two_partial
                .capture()
                .expect_err("injected partial peer-tail capture fault did not error");
            two_partial.capture_fail_stage = None;
            let chain = format!("{err:#}");
            ensure!(
                chain.contains("MoE capture transaction failed")
                    && chain.contains("injected capture")
                    && two_partial.graph.is_none()
                    && !two_partial.has_moe_segments()
                    && !two_partial.has_moe_peer_tail(),
                "Runner FAIL partial peer-tail capture fault fell back or lost context: {chain}"
            );
            println!(
                "PASS multi_device_capture_no_fallback format={format:?} fault={fault:?} segments=true"
            );
        }
    }
    ensure!(defects == 0, "DEFECT split combine cells={defects}");
    println!(
        "PASS split_combine_gate formats=Bf16,Fp8W8a16 eager_graph_bitwise=true perturbed_routed_peer_detected=true"
    );
    Ok(())
}

pub(super) fn placement_compare(
    label: &str,
    a: &[f32],
    b: &[f32],
    bf16_ulp_limit: Option<u16>,
) -> Result<usize> {
    ensure!(
        !a.is_empty() && a.len() == b.len(),
        "Runner FAIL {label} extent"
    );
    ensure!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "Runner FAIL {label} nonfinite"
    );
    let mismatches = a
        .iter()
        .zip(b)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let first = a
        .iter()
        .zip(b)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let ordered = |value: f32| {
        let bits = bf16::from_f32(value).to_bits();
        if bits & 0x8000 == 0 {
            bits | 0x8000
        } else {
            !bits
        }
    };
    let max_bf16_ulp = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| ordered(a).abs_diff(ordered(b)))
        .max()
        .context("empty comparison")?;
    let passed = match bf16_ulp_limit {
        Some(limit) => max_bf16_ulp <= limit,
        None => mismatches == 0,
    };
    println!(
        "{} placement_comparison {}",
        if passed { "PASS" } else { "DEFECT" },
        serde_json::json!({
            "label":label,"elements":a.len(),"bit_mismatches":mismatches,"first":first,
            "first_a":first.map(|i|a[i]),"first_b":first.map(|i|b[i]),"max_abs":max_abs,"max_bf16_ulp":max_bf16_ulp,
            "strict_bits":bf16_ulp_limit.is_none(),"bf16_ulp_limit":bf16_ulp_limit,"passed":passed
        })
    );
    Ok(usize::from(!passed))
}

fn placement_graph(
    stream: &Arc<CudaStream>,
    launch: impl FnOnce() -> Result<()>,
) -> Result<CudaGraph> {
    stream.synchronize()?;
    #[cfg(feature = "cuda")]
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    #[cfg(feature = "rocm")]
    let capture = stream.begin_capture()?;
    let launched = launch();
    #[cfg(feature = "cuda")]
    let captured = stream.end_capture(sys::CUgraphInstantiate_flags(0));
    #[cfg(feature = "rocm")]
    let captured = capture.finish().map(Some);
    launched?;
    let graph = captured?.context("Runner FAIL empty placement graph")?;
    graph.upload()?;
    stream.synchronize()?;
    Ok(graph)
}

pub(super) fn placement_components(engine: &Engine, model: &Path) -> Result<usize> {
    use crate::experts::{
        catalogue::ExpertCatalogue,
        disk::{DiskExperts, DiskReader},
    };
    let device = &engine.device;
    let stream = &device.stream;
    let format = engine
        .model_spec
        .expert_format
        .context("placement expert_format missing")?;
    let geometry = ExpertGeometry::from_model(&engine.model_spec)?;
    ensure!(
        geometry.hidden == 2048
            && geometry.intermediate == 512
            && geometry.top_k == 8
            && geometry.experts == 256,
        "Runner FAIL placement geometry {geometry:?}"
    );
    let manifest = crate::checkpoint::MappedCheckpoint::open(model)?
        .manifest(&engine.model_spec, crate::checkpoint::LoadPolicy::TextOnly)?;
    let catalogue = Arc::new(ExpertCatalogue::from_manifest(&manifest, geometry, format)?);
    let reader = DiskReader::open(&catalogue, model, 1)?;
    let elements = usize::try_from(catalogue.slot_bytes() / 2)?;
    let host = HostBlocks::new(1, elements)?;
    let mut gpu_weights = Vec::new();
    for expert in 0..8 {
        let index = reader.read_expert(0, expert, &catalogue, catalogue.slot_bytes())?;
        let lease = reader.values(index)?;
        if expert == 0 {
            host.claim(0)?.slice()?.copy_from_slice(&lease);
        }
        gpu_weights.push(stream.clone_htod(lease.values())?);
    }
    let mut gate_ptrs = vec![0u64; 256];
    let mut down_ptrs = vec![0u64; 256];
    let mut valid = vec![0u32; 256];
    let mut generations = vec![0u64; 256];
    for (expert, weights) in gpu_weights.iter().enumerate() {
        gate_ptrs[expert] = weights.device_ptr(stream).0;
        down_ptrs[expert] = gate_ptrs[expert] + catalogue.layout().down_offset();
        valid[expert] = 1;
        generations[expert] = 1;
    }
    let gates = stream.clone_htod(&gate_ptrs)?;
    let downs = stream.clone_htod(&down_ptrs)?;
    let valid = stream.clone_htod(&valid)?;
    let generations = stream.clone_htod(&generations)?;
    let table = SlotBindings {
        gate_up: view(&gates, stream, 0, [256], [1])?,
        down: view(&downs, stream, 0, [256], [1])?,
        valid: view(&valid, stream, 0, [256], [1])?,
        generation: view(&generations, stream, 0, [256], [1])?,
    };
    let input = (0..2048)
        .map(|index| bf16::from_f32(((index * 37 % 131) as f32 - 65.0) / 128.0))
        .collect::<Vec<_>>();
    let x_buffer = stream.clone_htod(&input)?;
    let mut y = stream.alloc_zeros::<bf16>(2048)?;
    let x = qwen::matrix(&x_buffer, stream, 0, 1, 2048)?;
    let mut y_view = TensorMut(qwen::matrix(&y, stream, 0, 1, 2048)?);
    let ffn = engine.moe_ffn_binding(0)?;
    let mut decode = placement_decode(engine)?;
    let ids = (0..8).collect::<Vec<i32>>();
    let weights = [bf16::from_f32(0.125); 8];
    stream.memcpy_htod(&ids, &mut decode.ids)?;
    stream.memcpy_htod(&weights, &mut decode.weights)?;
    stream.memcpy_htod(&[1u32; 8], &mut decode.active)?;
    stream.memcpy_htod(&[1u64; 8], &mut decode.generation)?;
    decode.contribute_device(device, 0, ffn.clone(), table, x)?;
    decode.combine_device(device, 0, ffn.clone(), x, &mut y_view)?;
    let route = placement_graph(stream, || decode.route_device(device, 0, &ffn, x))?;
    let contribute = placement_graph(stream, || {
        decode.contribute_device(device, 0, ffn.clone(), table, x)
    })?;
    let join = placement_graph(stream, || {
        decode.combine_device(device, 0, ffn.clone(), x, &mut y_view)
    })?;
    decode.segments.push(graphs::Segments {
        route,
        contribute,
        join,
    });
    let native = [ExpertWeights::from_slot(
        Shape::new(2048, 512)?,
        host.block_range(0)?,
        catalogue.layout(),
    )?];
    let mut disk = DiskExperts::new(catalogue.clone(), model, 1, 1)?;
    let assignment = [Assignment {
        row: 0,
        choice: 0,
        expert: 0,
        weight: weights[0].to_f32(),
    }];
    let disk_route = [crate::experts::runtime::Route {
        row: 0,
        choice: 0,
        expert: 0,
        weight: weights[0].to_f32(),
    }];
    let mut defects = 0;
    let mut eager = Vec::new();
    for tier in 0..3 {
        let mut reference: Option<Vec<f32>> = None;
        for (repeat, graph) in [
            (0, false),
            (1, false),
            (2, false),
            (0, true),
            (1, true),
            (2, true),
        ] {
            stream.memcpy_htod(&ids, &mut decode.ids)?;
            stream.memcpy_htod(&weights, &mut decode.weights)?;
            let active = if tier == 0 {
                [1; 8]
            } else {
                [0, 1, 1, 1, 1, 1, 1, 1]
            };
            decode.read_route(device, 0, &ffn, x)?;
            let dispatch = Dispatch {
                active: &active,
                generation: &[1; 8],
                host: Some(host.clone()),
                experts: &native,
                assignments: if tier == 1 { &assignment } else { &[] },
                miss_rows: if tier == 0 { &[] } else { &[0] },
                host_dma: None,
                disk: if tier == 2 { &disk_route } else { &[] },
                disk_cpu: Some(&mut disk),
                layer: 0,
            };
            let before = decode.stats;
            let _range = super::super::super::profile::Range::new(
                cfg!(feature = "cuda"),
                c"placement_component",
            );
            stream.memset_zeros(&mut y)?;
            decode.finish_with(
                device,
                ffn.clone(),
                table,
                dispatch,
                (x, &mut y_view),
                graph.then_some(0),
            )?;
            stream.synchronize()?;
            ensure!(
                decode.stats[1] - before[1] == if tier == 0 { 8 } else { 7 }
                    && decode.stats[2] - before[2] == u64::from(tier != 0),
                "Runner FAIL forced tier coverage"
            );
            let done = stream.clone_dtoh(&decode.done)?;
            ensure!(done == [1; 8], "Runner FAIL placement completion words");
            ensure!(
                stream.clone_dtoh(&decode.ids)? == ids
                    && stream.clone_dtoh(&decode.weights)? == weights,
                "Runner FAIL placement changed frozen routing"
            );
            let mut values = stream
                .clone_dtoh(&decode.contributions)?
                .into_iter()
                .map(bf16::to_f32)
                .collect::<Vec<_>>();
            values.extend(stream.clone_dtoh(&y)?.into_iter().map(bf16::to_f32));
            println!(
                "PASS placement_path tier={tier} graph={graph} repeat={repeat} gpu_choices={} cpu_choices={} disk_choices={} graph_launches={} input=[1,2048] contributions=[1,8,2048] output=[1,2048]",
                decode.stats[1] - before[1],
                u64::from(tier == 1),
                u64::from(tier == 2),
                if graph { 2 } else { 0 }
            );
            if let Some(previous) = &reference {
                defects += placement_compare(
                    &format!(
                        "tier{tier}_{}",
                        if graph { "graph_eager" } else { "self_repeat" }
                    ),
                    previous,
                    &values,
                    None,
                )?;
            } else {
                reference = Some(values);
            }
        }
        eager.push(reference.context("missing eager placement")?);
    }
    defects += placement_compare(
        "gpu_host_contribution",
        &eager[0][..2048],
        &eager[1][..2048],
        Some(1),
    )?;
    defects += placement_compare(
        "host_disk_contribution",
        &eager[1][..2048],
        &eager[2][..2048],
        None,
    )?;
    defects += placement_compare(
        "gpu_host_output",
        &eager[0][16384..],
        &eager[1][16384..],
        Some(1),
    )?;
    defects += placement_compare(
        "host_disk_output",
        &eager[1][16384..],
        &eager[2][16384..],
        None,
    )?;
    Ok(defects)
}

fn placement_decode(engine: &Engine) -> Result<Decode> {
    let geometry = ExpertGeometry::from_model(&engine.model_spec)?;
    let format = engine
        .model_spec
        .expert_format
        .context("placement expert format")?;
    let device = &engine.device;
    Decode::new(
        device,
        DecodeOptions {
            peer: None,
            geometry,
            residual_output: engine.model_spec.architecture
                != crate::model_spec::Architecture::Flint,
            max_rows: 1,
            workers: 1,
            cpu_bounds: (1, 8),
            participants: crate::experts::plan::ParticipantTable::uniform(1, 8, 1)?,
            format,
            shared_intermediate: engine
                .model_spec
                .moe
                .context("MoE shared geometry missing")?
                .shared_intermediate,
            swiglu_rounding: engine.model_spec.swiglu_rounding(),
            swiglu_limit: engine.model_spec.swiglu_limit(),
            routed_weight: engine.model_spec.routed_weight_boundary(),
        },
    )
}

fn published_residency(engine: &Engine, model: &Path) -> Result<usize> {
    use crate::experts::{catalogue::ExpertCatalogue, plan::Tier};
    let device = &engine.device;
    let stream = &device.stream;
    let geometry = ExpertGeometry::from_model(&engine.model_spec)?;
    let format = engine
        .model_spec
        .expert_format
        .context("published expert format")?;
    let manifest = crate::checkpoint::MappedCheckpoint::open(model)?
        .manifest(&engine.model_spec, crate::checkpoint::LoadPolicy::TextOnly)?;
    let catalogue = Arc::new(ExpertCatalogue::from_manifest(&manifest, geometry, format)?);
    let slot = catalogue.slot_bytes();
    let capacity = Capacity {
        layout: catalogue.layout().clone(),
        span_rows: 8,
        wave_rows: 8,
        device_expert_bytes: slot * u64::from(geometry.layers * 4),
        peer_expert_bytes: 0,
        gpu0_layers: geometry.layers,
        host_expert_bytes: slot * u64::from(geometry.layers * 4 + 2),
        slots_per_layer: vec![4; geometry.layers as usize],
        host_blocks: geometry.layers * 4,
        spare_blocks: 2,
        disk_bytes: slot * u64::from(geometry.layers * (geometry.experts - 8)),
        worker_count: 1,
        disk_workers: 1,
        prefill_staging_slots: 0,
        pinned: true,
    };
    let initial = vec![vec![0, 1, 2, 3]; geometry.layers as usize];
    let mut slots = Slots::load(
        device,
        None,
        geometry,
        residency::Load {
            capacity: &capacity,
            admission: crate::experts::plan::Admission {
                budget_bytes_per_token: slot * 2,
            },
            initial: &initial,
            catalogue,
            directory: model,
            engines: 2,
            max_rows: 1,
            swiglu_rounding: engine.model_spec.swiglu_rounding(),
            swiglu_limit: engine.model_spec.swiglu_limit(),
            routed_weight: engine.model_spec.routed_weight_boundary(),
        },
    )?;
    device.finish_upload()?;
    slots.freeze_probe_admissions()?;
    let mut decode = placement_decode(engine)?;
    let ffn = engine.moe_ffn_binding(0)?;
    let input = (0..2048)
        .map(|i| bf16::from_f32(((i * 37 % 131) as f32 - 65.0) / 128.0))
        .collect::<Vec<_>>();
    let x = stream.clone_htod(&input)?;
    let mut y = stream.alloc_zeros::<bf16>(2048)?;
    let x = qwen::matrix(&x, stream, 0, 1, 2048)?;
    let mut out = TensorMut(qwen::matrix(&y, stream, 0, 1, 2048)?);
    let ids = [0i32, 3, 4, 7, 8, 11, 12, 15];
    let weights = [bf16::from_f32(0.125); 8];
    let reference_slots = engine
        .moe_slots
        .as_ref()
        .context("resident reference slots")?;
    ensure!(
        reference_slots.all_resident(),
        "Runner FAIL published reference is not resident"
    );
    let generations = ids.map(|id| reference_slots.published[id as usize]);
    stream.memcpy_htod(&ids, &mut decode.ids)?;
    stream.memcpy_htod(&weights, &mut decode.weights)?;
    stream.memcpy_htod(&[1u32; 8], &mut decode.active)?;
    stream.memcpy_htod(&generations, &mut decode.generation)?;
    decode.contribute_device(
        device,
        0,
        ffn.clone(),
        reference_slots.bindings(0, stream)?,
        x,
    )?;
    let reference = stream
        .clone_dtoh(&decode.contributions)?
        .into_iter()
        .map(bf16::to_f32)
        .collect::<Vec<_>>();
    stream.memcpy_htod(&[0u32; 8], &mut decode.active)?;
    let table = slots.bindings(0, stream)?;
    let route = placement_graph(stream, || decode.route_device(device, 0, &ffn, x))?;
    let contribute = placement_graph(stream, || {
        decode.contribute_device(device, 0, ffn.clone(), table, x)
    })?;
    let join = placement_graph(stream, || {
        decode.combine_device(device, 0, ffn.clone(), x, &mut out)
    })?;
    decode.segments.push(graphs::Segments {
        route,
        contribute,
        join,
    });
    let mut defects = 0;
    let mut previous_generation = Vec::new();
    for (request, target) in [[0u32, 1, 2, 3], [8, 9, 10, 11], [12, 13, 14, 15]]
        .into_iter()
        .enumerate()
    {
        let mut frequency = vec![vec![0.0; geometry.experts as usize]; geometry.layers as usize];
        for layer in &mut frequency {
            layer[..4].fill(1.0);
        }
        frequency[0].fill(0.0);
        for id in target {
            frequency[0][id as usize] = 1.0;
        }
        slots.delay_copies();
        slots.begin_request(&frequency)?;
        let placement = slots.capture_placement()?;
        let tiers = ids.map(|id| placement[id as usize].0);
        ensure!(
            tiers.contains(&Tier::Gpu) && tiers.iter().any(|tier| *tier != Tier::Gpu),
            "Runner FAIL forced mixed placement request={request} tiers={tiers:?}"
        );
        if request == 0 {
            ensure!(
                tiers.contains(&Tier::Host) && tiers.contains(&Tier::Disk),
                "Runner FAIL native host/disk tiers absent"
            );
        }
        let mut eager: Option<Vec<f32>> = None;
        for graph in [false, true] {
            for repeat in 0..3 {
                slots.begin_token()?;
                stream.memcpy_htod(&ids, &mut decode.ids)?;
                stream.memcpy_htod(&weights, &mut decode.weights)?;
                stream.memset_zeros(&mut y)?;
                if graph {
                    slots.run_graph(device, &mut decode, 0, 0, ffn.clone(), (x, &mut out))?;
                } else {
                    decode.read_route(device, 0, &ffn, x)?;
                    slots.finish_routed(device, &mut decode, 0, ffn.clone(), (x, &mut out))?;
                }
                let values = stream
                    .clone_dtoh(&decode.contributions)?
                    .into_iter()
                    .map(bf16::to_f32)
                    .collect::<Vec<_>>();
                ensure!(
                    stream.clone_dtoh(&decode.done)? == [1; 8],
                    "Runner FAIL native completion words"
                );
                defects += placement_compare(
                    "published_resident_vs_forced",
                    &reference,
                    &values,
                    Some(1),
                )?;
                if let Some(expected) = &eager {
                    defects +=
                        placement_compare("published_tier_repeat_graph", expected, &values, None)?;
                } else {
                    eager = Some(values);
                }

                slots.finish_token()?;
                println!(
                    "PASS published_residency request={request} graph={graph} repeat={repeat} tiers={tiers:?} native_slot_bytes={slot} input=[1,2048] contributions=[1,8,2048]"
                );
            }
        }
        if request > 0 {
            ensure!(
                slots.published != previous_generation,
                "DEFECT request refill did not advance generations"
            );
        }
        previous_generation.clone_from(&slots.published);
        slots.end_request()?;
    }
    let mut trace = Trace::default();
    slots.trace(&mut trace)?;
    ensure!(
        trace.values("stats.copy_wait_pending")?[0] > 0.0,
        "Runner FAIL publication delay was not observed"
    );
    println!(
        "PUBLISHED_RESIDENCY cells=18 defects={defects} pinned=true requests=3 refill_generation=true"
    );
    Ok(defects)
}

#[test]
#[ignore = "GPU lease, published MiniCPM-V-4.7 FP8 and pinned FLYINGFISH_RUNTIME_DIR"]
fn published_residency_gpu() -> Result<()> {
    let model = PathBuf::from(std::env::var_os("FF_MODEL").context("published model missing")?);
    let runtime = PathBuf::from(
        std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("published runtime missing")?,
    );
    let settings = Settings {
        runtime_dir: Some(runtime),
        prefix_cache: false,
        device_budget_gib: None,
        draft_model: None,
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: true,
        kv_format: crate::kv::KvFormat::Bf16,
    };
    let engine = Engine::load(&model, 0, 8192, Some(129), settings)?;
    engine.check_ready()?;
    let mut defects = placement_components(&engine, &model)?;
    defects += published_residency(&engine, &model)?;
    ensure!(defects == 0, "DEFECT published residency cells={defects}");
    Ok(())
}

std::thread_local! {
    pub(super) static MEMORY_BUDGET: std::cell::Cell<Option<(usize, u64)>> = const { std::cell::Cell::new(None) };
}

#[test]
#[ignore = "GPU lease, published FP8 model, frozen model-definition outputs and pinned runtime"]
fn published_streamed_model_gpu() -> Result<()> {
    let fixtures: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/minicpm_reference.json"
    )))?;
    let cases = fixtures["cases"]
        .as_array()
        .context("published reference cases")?;
    let model = PathBuf::from(std::env::var_os("FF_MODEL").context("published model missing")?);
    let reference = PathBuf::from(
        std::env::var_os("FF_DENSE_REFERENCE").context("published reference missing")?,
    );
    let runtime = PathBuf::from(
        std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("published runtime missing")?,
    );
    let settings = Settings {
        runtime_dir: Some(runtime),
        prefix_cache: false,
        device_budget_gib: None,
        draft_model: None,
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: true,
        kv_format: crate::kv::KvFormat::Bf16,
    };
    MEMORY_BUDGET.set(Some((24 << 30, 4 << 30)));
    let loaded = Engine::load(&model, 0, 8192, Some(129), settings);
    MEMORY_BUDGET.set(None);
    let mut engine = loaded?;
    engine.check_ready()?;
    let cap = &engine
        .moe_prefill
        .as_ref()
        .context("streamed prefill missing")?
        .capacity;
    ensure!(
        cap.pinned
            && cap.disk_bytes > 0
            && cap.host_blocks > 0
            && cap.slots_per_layer.iter().all(|&n| n < 256),
        "Runner FAIL forced planner did not produce pinned-host/disk partial residency: {cap:?}"
    );
    println!(
        "PUBLISHED_STREAMED_PLAN slots={:?} host_blocks={} disk_bytes={} pinned={} device_limit={} host_limit={}",
        cap.slots_per_layer,
        cap.host_blocks,
        cap.disk_bytes,
        cap.pinned,
        24u64 << 30,
        4u64 << 30
    );
    let mut defects = 0;
    let mut cells = 0;
    for rows in [1024usize, 4096] {
        let case = cases
            .iter()
            .find(|case| case["rows"].as_u64() == Some(rows as u64))
            .context("published fixture rows")?;
        let max_limit = case["limits"]["max_normalized"]
            .as_f64()
            .context("frozen max limit")?;
        let rms_limit = case["limits"]["rms_normalized"]
            .as_f64()
            .context("frozen RMS limit")?;
        ensure!(
            max_limit.is_finite() && max_limit > 0.0 && rms_limit.is_finite() && rms_limit > 0.0,
            "Runner FAIL invalid frozen limits"
        );
        let bytes =
            std::fs::read(reference.join(format!("text{rows}-chunk{rows}-repeat0.safetensors")))?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        let tensor = tensors.tensor("logits")?;
        ensure!(
            tensor.dtype() == safetensors::Dtype::F32 && tensor.shape() == [4, 248144],
            "Runner FAIL reference must be F32[4,248144]"
        );
        let expected = tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| f32::from_le_bytes(word))
            .collect::<Vec<_>>();
        ensure!(
            expected.iter().all(|v| v.is_finite()),
            "Runner FAIL reference nonfinite"
        );
        let ids = (0..rows)
            .map(|i| ((i * 269 + 23) % 248144) as u32)
            .collect::<Vec<_>>();
        let mut eager = Vec::new();
        for (repeat, graph) in [false, true, false].into_iter().enumerate() {
            engine.reset()?;
            let mut token = engine.prefill(&ids, None)?;
            let mut maximum = 0.0f64;
            let mut rms = 0.0f64;
            let mut min_margin = f32::INFINITY;
            let mut cell_defects = 0;
            for step in 0..4 {
                if step > 0 {
                    engine.set_token([115, 1301, 3809][step - 1])?;
                    token = engine.decode(graph, None)?;
                }
                ensure!(
                    engine.position == rows + step,
                    "DEFECT streamed absolute position"
                );
                let logits = engine.decode_logits()?;
                ensure!(
                    logits.len() == 248144 && logits.iter().all(|v| v.is_finite()),
                    "Runner FAIL streamed logits"
                );
                let bits = logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                if repeat == 0 {
                    eager.push((token, bits));
                } else {
                    cell_defects += usize::from(eager[step] != (token, bits));
                }
                let reference = &expected[step * 248144..(step + 1) * 248144];
                let scale = (reference.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>()
                    / 248144.0)
                    .sqrt();
                ensure!(
                    scale.is_finite() && scale > 0.0,
                    "Runner FAIL reference scale"
                );
                let mut error_max = 0.0f64;
                let mut error_square = 0.0f64;
                for (&actual, &wanted) in logits.iter().zip(reference) {
                    let error = f64::from(actual) - f64::from(wanted);
                    error_max = error_max.max(error.abs());
                    error_square += error * error;
                }
                maximum = maximum.max(error_max / scale);
                rms = rms.max((error_square / 248144.0).sqrt() / scale);
                let mut order = (0..248144).collect::<Vec<_>>();
                order.sort_unstable_by(|&a, &b| {
                    reference[b].total_cmp(&reference[a]).then(a.cmp(&b))
                });
                let margin = reference[order[0]] - reference[order[1]];
                min_margin = min_margin.min(margin);
                cell_defects += usize::from(
                    token as usize != order[0] && f64::from(margin) > 2.0 * max_limit * scale,
                );
            }
            cell_defects += usize::from(maximum > max_limit || rms > rms_limit);
            let position = engine.position;
            ensure!(
                engine
                    .prefill(&vec![1; engine.capacity - position + 1], None)
                    .is_err()
                    && engine.position == position,
                "DEFECT streamed prefill exceeded capacity or mutated position"
            );
            let stats = engine
                .moe_prefill
                .as_ref()
                .context("streamed prefill stats")?
                .stats;
            ensure!(
                stats[2] > 0 && stats[6] + stats[7] > 0,
                "Runner FAIL native streaming/refill not exercised: {stats:?}"
            );
            engine.check_ready()?;
            defects += cell_defects;
            cells += 1;
            println!(
                "PUBLISHED_STREAMED_MODEL rows={rows} decode_prefix={rows} graph={graph} repeat={repeat} max_normalized={maximum:e} rms_normalized={rms:e} max_limit={max_limit} rms_limit={rms_limit} min_margin={min_margin} stats={stats:?} defects={cell_defects}"
            );
        }
    }
    println!(
        "PUBLISHED_STREAMED_MODEL cells={cells} defects={defects} frozen_model_definition_limits=true"
    );
    ensure!(defects == 0, "DEFECT streamed model cells={defects}");
    Ok(())
}

pub(super) fn chain_state(model: &Engine) -> Result<Vec<u8>> {
    let stream = &model.device.stream;
    let mut values = Vec::new();
    for layer in &model.state.gdn {
        for buffer in [&layer.s, &layer.history] {
            for value in stream.clone_dtoh(buffer)? {
                ensure!(value.is_finite(), "nonfinite chain recurrent state");
                values.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
    }
    for cache in &model.state.kv {
        for buffer in [&cache.k, &cache.v] {
            for value in stream.clone_dtoh(buffer)? {
                ensure!(value.is_finite(), "nonfinite chain KV");
                values.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
    }
    Ok(values)
}

pub(super) fn restore_chain_state(model: &mut Engine, mut values: &[u8]) -> Result<()> {
    let stream = &model.device.stream;
    for layer in &mut model.state.gdn {
        for buffer in [&mut layer.s, &mut layer.history] {
            let (saved, tail) = values
                .split_at_checked(buffer.len() * size_of::<f32>())
                .context("chain recurrent state restore extent")?;
            let saved: Vec<_> = saved
                .as_chunks::<4>()
                .0
                .iter()
                .map(|value| f32::from_le_bytes(*value))
                .collect();
            stream.memcpy_htod(&saved, buffer)?;
            values = tail;
        }
    }
    for cache in &mut model.state.kv {
        for buffer in [&mut cache.k, &mut cache.v] {
            let (saved, tail) = values
                .split_at_checked(buffer.len() * size_of::<bf16>())
                .context("chain KV restore extent")?;
            let saved: Vec<_> = saved
                .as_chunks::<2>()
                .0
                .iter()
                .map(|value| bf16::from_bits(u16::from_le_bytes(*value)))
                .collect();
            stream.memcpy_htod(&saved, buffer)?;
            values = tail;
        }
    }
    ensure!(values.is_empty(), "chain state restore has trailing bytes");
    Ok(())
}

#[test]
#[ignore = "requires GPU, published target/draft and an owned FLYINGFISH_RUNTIME_DIR"]
fn minicpm_chain_verify_gpu() -> Result<()> {
    let mut defects = 0;
    for chunk in [33, 9, 17] {
        if let Err(error) = minicpm_chain_verify(chunk) {
            if !error.to_string().starts_with("DEFECT") {
                return Err(error);
            }
            println!("DEFECT minicpm_chain_chunk chunk={chunk} error={error}");
            defects += 1;
        }
    }
    ensure!(defects == 0, "DEFECT hybrid chain chunks={defects}");
    Ok(())
}

fn minicpm_chain_verify(chunk: usize) -> Result<()> {
    ensure!(
        std::env::var("FF_Q38_GPU_WINDOW").as_deref() == Ok("granted"),
        "GPU window not acknowledged"
    );
    let target = PathBuf::from(std::env::var("FF_MODEL")?);
    let draft = PathBuf::from(std::env::var("FF_ARGMAX_DRAFT")?);
    let settings = Settings {
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: true,
        draft_model: Some(draft),
        runtime_dir: Some(PathBuf::from(std::env::var("FLYINGFISH_RUNTIME_DIR")?)),
        prefix_cache: false,
        device_budget_gib: None,
        kv_format: crate::kv::KvFormat::Bf16,
    };
    Engine::calibrate(&target, 0, 256, Some(chunk), settings.clone())?;
    let mut model = Engine::load(&target, 0, 256, Some(chunk), settings)?;
    let stream = model.device.stream.clone();
    let (rows, draft_rows, taps) = {
        let spec = model.spec.as_ref().context("chain draft missing")?;
        (
            spec.max_verify_rows,
            spec.draft.config.verify_rows()?,
            spec.draft.config.target_layer_ids.clone(),
        )
    };
    ensure!(
        rows == 8 && draft_rows == 16 && model.model_spec.moe.is_some(),
        "production block15 MoE fixture required"
    );
    let snapshots = model
        .state
        .gdn
        .iter()
        .map(|live| {
            Ok(super::super::super::state::GdnState {
                s: stream.alloc_zeros((rows - 1) * live.s.len())?,
                history: stream.alloc_zeros((rows - 1) * live.history.len())?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let h = model.config.hidden_size;
    let width = h * taps.len();
    ensure!(
        chain_state(&model)?.iter().all(|&value| value == 0),
        "graph setup changed initial target state"
    );
    let mut initial_draft = Trace::default();
    model
        .spec
        .as_ref()
        .context("chain missing")?
        .draft
        .snapshot_kv(&mut initial_draft, 0, draft_rows)?;
    for layer in 0..model
        .spec
        .as_ref()
        .context("chain missing")?
        .draft
        .config
        .num_hidden_layers
    {
        for name in [format!("kv_k_{layer}"), format!("kv_v_{layer}")] {
            ensure!(
                initial_draft
                    .values(&name)?
                    .iter()
                    .all(|value| value.to_bits() == 0),
                "graph setup changed initial draft KV layer={layer}"
            );
        }
    }
    let mut defects = 0;
    let mut wrong_layer = 0;
    let mut perturbed_gdn_control = 0;
    for prefix in [7usize, 17, 65, 125, 129] {
        model.reset()?;
        let ids: Vec<_> = (0..prefix).map(|row| (row * 263 + 17) as u32).collect();
        let mut trace = Trace::all_layers();
        if prefix <= model.chunk {
            model.prefill_tail(&ids, &mut trace, ids.len())?;
            let capture = model
                .hidden_capture
                .as_ref()
                .context("prefill tap binding missing")?;
            let values = stream.clone_dtoh(&capture.values.slice(..prefix * width))?;
            for (slot, layer) in taps.iter().enumerate() {
                let expected = trace.values(&format!("layer.{layer}"))?;
                ensure!(
                    expected.len() == prefix * h,
                    "prefill trace extent mismatch"
                );
                defects += values
                    .chunks_exact(width)
                    .enumerate()
                    .map(|(row, values)| {
                        values[slot * h..(slot + 1) * h]
                            .iter()
                            .zip(&expected[row * h..(row + 1) * h])
                            .filter(|(a, b)| a.to_bits() != bf16::from_f32(**b).to_bits())
                            .count()
                    })
                    .sum::<usize>();
            }
            let saved = values;
            let first = taps[0];
            let capture = model
                .hidden_capture
                .as_mut()
                .context("tap control binding missing")?;
            ensure!(
                capture.layer_slots[first + 1].is_none(),
                "wrong-layer control overlaps a tap"
            );
            capture.layer_slots[first] = None;
            capture.layer_slots[first + 1] = Some(0);
            model.reset()?;
            model.prefill(&ids, None)?;
            let capture = model
                .hidden_capture
                .as_mut()
                .context("tap control binding missing")?;
            let changed = stream.clone_dtoh(&capture.values.slice(..prefix * width))?;
            wrong_layer += changed
                .chunks_exact(width)
                .zip(saved.chunks_exact(width))
                .map(|(a, b)| {
                    a[..h]
                        .iter()
                        .zip(&b[..h])
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count()
                })
                .sum::<usize>();
            capture.layer_slots[first + 1] = None;
            capture.layer_slots[first] = Some(0);
            model.reset()?;
            model.prefill(&ids, None)?;
        } else {
            model.prefill(&ids, None)?;
        }
        let inputs: Vec<_> = (0..rows)
            .map(|row| ((prefix + row) * 541 + 37) as u32)
            .collect();
        model.set_token(inputs[0])?;
        model
            .spec
            .as_mut()
            .context("chain missing")?
            .checkpoint
            .as_mut()
            .context("chain checkpoint missing")?
            .save(&model.state, &model.work.token, prefix, &stream, None)?;
        let mut expected_states = vec![chain_state(&model)?];
        let mut logits = Vec::new();
        let mut captures = Vec::new();
        let mut predictions = Vec::new();
        let mut scalar_trace = Trace::all_layers();
        for (row, token) in inputs.iter().enumerate() {
            model.set_token(*token)?;
            model.advance(false, (row == 0).then_some((&mut scalar_trace, "scalar")))?;
            logits.extend(stream.clone_dtoh(&model.work.logits)?);
            predictions.push(model.token()?);
            captures.extend(
                stream.clone_dtoh(
                    &model
                        .hidden_capture
                        .as_ref()
                        .context("decode taps missing")?
                        .values
                        .slice(..width),
                )?,
            );
            expected_states.push(chain_state(&model)?);
        }
        model
            .spec
            .as_ref()
            .context("chain missing")?
            .checkpoint
            .as_ref()
            .context("chain checkpoint missing")?
            .restore(&mut model.state, &mut model.work.token, &stream)?;
        model.bind_gdn_snapshots(&snapshots, rows)?;
        model.position = prefix;
        model.committed_prefix_ids.truncate(prefix);
        model.set_token(inputs[0])?;
        stream.memcpy_htod(&[prefix as i32], &mut model.work.position)?;
        stream.memcpy_htod(&[prefix as i32 + 1], &mut model.work.length)?;
        stream.memcpy_htod(&inputs, &mut model.work.ids.slice_mut(..rows))?;
        model.run_step(
            crate::backend::Step {
                rows,
                input: crate::backend::Input::Ids,
                mask: crate::backend::Mask::Causal,
                head: Head::All,
            },
            None,
        )?;
        model
            .spec
            .as_ref()
            .context("chain missing")?
            .checkpoint
            .as_ref()
            .context("chain checkpoint missing")?
            .restore(&mut model.state, &mut model.work.token, &stream)?;
        model.bind_gdn_verify_inputs(rows)?;
        let mut batch_trace = Trace::all_layers();
        model.position = prefix;
        model.committed_prefix_ids.truncate(prefix);
        model.set_token(inputs[0])?;
        stream.memcpy_htod(&[prefix as i32], &mut model.work.position)?;
        stream.memcpy_htod(&[prefix as i32 + 1], &mut model.work.length)?;
        stream.memcpy_htod(&inputs, &mut model.work.ids.slice_mut(..rows))?;
        model.run_step(
            crate::backend::Step {
                rows,
                input: crate::backend::Input::Ids,
                mask: crate::backend::Mask::Causal,
                head: Head::All,
            },
            Some(Target {
                trace: &mut batch_trace,
                prefix: "batch",
                skip: 0,
            }),
        )?;
        let full_verify_state = chain_state(&model)?;
        for layer in 0..model.model_spec.layers.len() {
            for point in ["norm1", "mixer_out", "o_proj", "post_attention", "norm2"] {
                let expected = scalar_trace.values(&format!("scalar.{layer}.debug.{point}"))?;
                let actual = batch_trace.values(&format!("batch.{layer}.debug.{point}"))?;
                ensure!(
                    actual.len() == rows * expected.len(),
                    "chain boundary trace extent"
                );
                let mismatches = actual
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let maximum = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                println!(
                    "CHAIN_BOUNDARY chunk={chunk} prefix={prefix} layer={layer} point={point} row0_mismatches={mismatches} max_abs={maximum}"
                );
                defects += mismatches;
            }
        }
        let spec = model.spec.as_ref().context("chain missing")?;
        let actual = stream.clone_dtoh(&spec.logits)?;
        let logit_mismatches = actual
            .iter()
            .zip(&logits)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let actual = stream.clone_dtoh(
            &model
                .hidden_capture
                .as_ref()
                .context("verify taps missing")?
                .values
                .slice(..rows * width),
        )?;
        let tap_mismatches = actual
            .iter()
            .zip(&captures)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let id_mismatches = stream
            .clone_dtoh(&spec.predictions)?
            .iter()
            .zip(&predictions)
            .filter(|(a, b)| a != b)
            .count();
        defects += logit_mismatches + tap_mismatches + id_mismatches;
        for accepted in (0..=rows).rev() {
            restore_chain_state(&mut model, &full_verify_state)?;
            let spec = model.spec.as_ref().context("chain missing")?;
            spec.checkpoint
                .as_ref()
                .context("chain checkpoint missing")?
                .select_prefix(&mut model.state, &snapshots, accepted, &stream)?;
            let snapshot_gdn = model
                .state
                .gdn
                .iter()
                .map(|g| Ok((stream.clone_dtoh(&g.s)?, stream.clone_dtoh(&g.history)?)))
                .collect::<Result<Vec<_>>>()?;
            restore_chain_state(&mut model, &full_verify_state)?;
            let spec = model.spec.as_ref().context("chain missing")?;
            spec.checkpoint
                .as_ref()
                .context("chain checkpoint missing")?
                .restore_for_replay(&mut model.state, accepted, rows, &stream)?;
            if accepted > 0 && accepted < rows {
                model.replay_gdn_prefix(prefix, accepted)?;
            }
            let mut snapshot_mismatches = 0;
            for (live, (exp_s, exp_h)) in model.state.gdn.iter().zip(&snapshot_gdn) {
                let live_s = stream.clone_dtoh(&live.s)?;
                let live_h = stream.clone_dtoh(&live.history)?;
                snapshot_mismatches += live_s
                    .iter()
                    .zip(exp_s)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count()
                    + live_h
                        .iter()
                        .zip(exp_h)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
            }
            let actual = chain_state(&model)?;
            let mismatches = actual
                .iter()
                .zip(&expected_states[accepted])
                .filter(|(a, b)| a != b)
                .count();
            defects += mismatches + snapshot_mismatches;
            println!(
                "{} minicpm_chain_prefix prefix={prefix} accepted={accepted} state_bit_mismatches={mismatches} snapshot_gdn_bit_mismatches={snapshot_mismatches}",
                if mismatches + snapshot_mismatches == 0 {
                    "PASS"
                } else {
                    "DEFECT"
                }
            );
            if accepted == crate::gdn_plan::MIN_VERIFY_ROWS as usize + 1 {
                let conv_dim = crate::op_contract::GdnSpec::from_model(&model.model_spec)?
                    .conv_dim()? as usize;
                let word_idx = (accepted - 1) * conv_dim;
                let orig = {
                    let qkv = &mut model.spec.as_mut().context("chain missing")?.gdn_inputs[0].qkv;
                    let orig = stream.clone_dtoh(&qkv.slice(word_idx..word_idx + 1))?;
                    let flipped = [bf16::from_f32(orig[0].to_f32() + 1.0)];
                    stream.memcpy_htod(&flipped, &mut qkv.slice_mut(word_idx..word_idx + 1))?;
                    orig
                };
                model
                    .spec
                    .as_ref()
                    .context("chain missing")?
                    .checkpoint
                    .as_ref()
                    .context("chain checkpoint missing")?
                    .restore_for_replay(&mut model.state, accepted, rows, &stream)?;
                model.replay_gdn_prefix(prefix, accepted)?;
                let perturbed_s = stream.clone_dtoh(&model.state.gdn[0].s)?;
                let perturbed_h = stream.clone_dtoh(&model.state.gdn[0].history)?;
                let control_diff = perturbed_s
                    .iter()
                    .zip(&snapshot_gdn[0].0)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count()
                    + perturbed_h
                        .iter()
                        .zip(&snapshot_gdn[0].1)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                perturbed_gdn_control += control_diff;
                stream.memcpy_htod(
                    &orig,
                    &mut model.spec.as_mut().context("chain missing")?.gdn_inputs[0]
                        .qkv
                        .slice_mut(word_idx..word_idx + 1),
                )?;
            }
        }
        println!(
            "{} minicpm_chain_verify prefix={prefix} rows={rows} logits_bit_mismatches={logit_mismatches} capture_bit_mismatches={tap_mismatches} id_mismatches={id_mismatches}",
            if logit_mismatches + tap_mismatches + id_mismatches == 0 {
                "PASS"
            } else {
                "DEFECT"
            }
        );
    }
    let mut ids = crate::serve::detok::Shared::load(&target)?.encode(
        "Explain the thermodynamic efficiency limit of a regenerative Brayton cycle, \
         derive the pressure ratio that maximizes specific work for an ideal gas with gamma = 1.4, \
         and write a Rust function `fn optimal_pressure_ratio(t_min: f64, t_max: f64, gamma: f64) -> f64` \
         with unit assertions. Then compare SQLite WAL checkpointing against PostgreSQL MVCC vacuuming \
         under concurrent read-heavy OLTP workloads, summarize the Byzantine fault tolerance safety proof \
         for three-phase commit vs HotStuff quorum certificates, and translate the summary into concise \
         Chinese, Japanese, and French bullet points with numerical latency examples. Finally, analyze \
         CRISPR-Cas9 off-target cleavage kinetics using Michaelis-Menten enzyme rate equations, derive \
         the discrete Fourier transform convolution theorem over finite cyclic groups Z/NZ, implement \
         an iterative Cooley-Tukey radix-2 FFT butterfly loop in C99, outline the Black-Scholes-Merton \
         partial differential equation boundary conditions for a European call option under constant volatility, \
         and discuss tectonic plate subduction zone seismic wave attenuation across oceanic lithosphere.",
    )?;
    ids.truncate(224);
    ensure!(
        ids.len() == 224,
        "chain verify prompt too short: {}",
        ids.len()
    );
    model.reset()?;
    model.prefill(&ids, None)?;
    let mut greedy = Vec::new();
    let mut greedy_states = vec![chain_state(&model)?];
    let mut greedy_logits = Vec::new();
    let mut greedy_taps = Vec::new();
    model
        .spec
        .as_mut()
        .context("chain missing")?
        .checkpoint
        .as_mut()
        .context("checkpoint missing")?
        .save(&model.state, &model.work.token, ids.len(), &stream, None)?;
    let mut greedy_trace = Trace::all_layers();
    for row in 0..rows {
        greedy.push(model.decode(false, (row == 2).then_some((&mut greedy_trace, "greedy")))?);
        greedy_states.push(chain_state(&model)?);
        greedy_logits.push(stream.clone_dtoh(&model.work.logits)?);
        greedy_taps.extend(
            stream.clone_dtoh(
                &model
                    .hidden_capture
                    .as_ref()
                    .context("greedy taps missing")?
                    .values
                    .slice(..width),
            )?,
        );
    }
    let greedy_taps = stream.clone_htod(&greedy_taps)?;
    let (cold_schedule, schedule) = {
        let spec = model.spec.as_ref().context("chain missing")?;
        (spec.cold_verify_schedule, spec.verify_schedule)
    };
    let slots_layout = model
        .moe_slots
        .as_ref()
        .context("chain slots missing")?
        .storage
        .layout();
    let slots_total: u32 = (0..model.model_spec.layers.len())
        .map(|layer| slots_layout.slot_count(layer as u32))
        .sum::<Result<u32>>()?;
    let total_experts = u32::try_from(model.model_spec.layers.len())?
        * model.model_spec.moe.context("chain MoE missing")?.experts;
    let k_max = schedule.max_rows;
    println!(
        "minicpm_chain_schedule {}",
        serde_json::json!({
            "expert_format": model.model_spec.expert_format,
            "slots_total": slots_total,
            "total_experts": total_experts,
            "cold_schedule": cold_schedule,
            "prefill_schedule": schedule,
        })
    );
    ensure!(
        slots_total == total_experts || schedule != cold_schedule,
        "Runner FAIL prefill did not refresh verify_schedule from cold prior: {schedule:?}"
    );
    ensure!(
        crate::dspark::select_verify_rows(
            &[1.0; 15],
            crate::dspark::MIN_MOE_VERIFY_ROWS,
            &schedule,
        )? == k_max,
        "Runner FAIL schedule max_rows={k_max} not enforced at unit confidence"
    );
    if model.model_spec.expert_format == Some(crate::model_spec::ExpertFormat::Fp8W8a16) {
        ensure!(
            k_max == rows,
            "Runner FAIL FP8 chain schedule capped max_rows={k_max} below {rows}"
        );
    } else if slots_total < total_experts {
        ensure!(
            (crate::dspark::MIN_MOE_VERIFY_ROWS..rows).contains(&k_max),
            "Runner FAIL partial-residency BF16 schedule did not cap max_rows below {rows}: {schedule:?}"
        );
    }
    let max_forced = if k_max < rows { k_max + 1 } else { rows };
    let cases: Vec<(Option<usize>, usize, usize, Option<u32>)> = if k_max == rows {
        vec![
            (Some(2), 1, 8, None),
            (Some(3), 2, 8, None),
            (Some(4), 3, 8, None),
            (Some(5), 2, 8, None),
            (Some(5), 4, 8, None),
            (Some(6), 5, 8, None),
            (Some(7), 6, 8, None),
            (Some(8), 0, 8, None),
            (Some(8), 3, 8, None),
            (Some(8), 7, 8, None),
            (Some(8), 7, 1, None),
            (Some(8), 7, 4, None),
            (Some(8), 7, 8, Some(greedy[3])),
            (None, 8, 8, None),
        ]
    } else {
        let mut list = Vec::new();
        for w in crate::dspark::MIN_MOE_VERIFY_ROWS..=k_max {
            list.push((Some(w), w - 1, rows, None));
        }
        list.push((Some(k_max), 0, rows, None));
        list.push((Some(k_max), k_max / 2, rows, None));
        list.push((Some(k_max), k_max - 1, 1, None));
        list.push((Some(k_max), k_max - 1, (k_max / 2).max(1), None));
        list.push((
            Some(k_max),
            k_max - 1,
            rows,
            Some(greedy[(k_max / 2).min(k_max - 1)]),
        ));
        list.push((Some(k_max + 1), k_max, rows, None));
        list.push((Some(k_max + 1), 1, rows, None));
        list.push((None, rows, rows, None));
        list
    };
    let mut executed_graph_widths = std::collections::BTreeSet::new();
    let mut dynamic_widths = std::collections::BTreeSet::new();
    for graph in [false, true, false] {
        model.set_spec_graph(graph)?;
        for &(verify_width, matched, limit, eos) in &cases {
            model.reset()?;
            model.prefill(&ids, None)?;
            ensure!(
                model
                    .spec
                    .as_ref()
                    .context("chain missing")?
                    .verify_schedule
                    == schedule,
                "Runner FAIL post-prefill verify_schedule drifted across resets"
            );
            model
                .spec
                .as_ref()
                .context("chain missing")?
                .checkpoint
                .as_ref()
                .context("checkpoint missing")?
                .restore(&mut model.state, &mut model.work.token, &stream)?;
            let anchor = model.token()?;
            model.set_token(anchor)?;
            model.set_spec_eos(&eos.into_iter().collect::<Vec<_>>())?;
            if let Some(w) = verify_width {
                let mut proposals = greedy[..w - 1].to_vec();
                if matched < w - 1 {
                    proposals[matched] = (proposals[matched] + 1) % model.model_spec.vocab;
                }
                model
                    .spec
                    .as_mut()
                    .context("chain missing")?
                    .forced_proposals = Some(proposals);
            }
            let graph_range = super::super::super::profile::Range::new(
                graph && cfg!(feature = "cuda"),
                c"minicpm_chain_graph",
            );
            let traced = !graph && verify_width == Some(5) && matched == 2;
            let mut commit_trace = Trace::all_layers();
            let round = model.spec_round(limit, traced.then_some(&mut commit_trace), "chain")?;
            if traced {
                let names = (0..model.model_spec.layers.len())
                    .flat_map(|layer| {
                        ["norm1", "mixer_out", "o_proj", "post_attention", "norm2"]
                            .map(move |point| format!("{layer}.debug.{point}"))
                    })
                    .chain(std::iter::once("norm".to_owned()));
                for name in names {
                    let expected = greedy_trace.values(&format!("greedy.{name}"))?;
                    let actual = commit_trace.values(&format!("chain.{name}"))?;
                    ensure!(
                        actual.len() == 5 * expected.len(),
                        "commit boundary trace extent"
                    );
                    let actual = &actual[2 * expected.len()..3 * expected.len()];
                    let mismatches = actual
                        .iter()
                        .zip(&expected)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    let maximum = actual
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    println!(
                        "CHAIN_COMMIT_BOUNDARY chunk={chunk} row=2 point={name} mismatches={mismatches} max_abs={maximum}"
                    );
                    defects += mismatches;
                }
            }
            drop(graph_range);
            let actual_width = round.predictions.len();
            if let Some(w) = verify_width {
                ensure!(
                    actual_width == w,
                    "Runner FAIL verify width mismatch: got {actual_width} expected {w}"
                );
            } else {
                ensure!(
                    (crate::dspark::MIN_MOE_VERIFY_ROWS..=k_max).contains(&actual_width),
                    "Runner FAIL dynamic verify width {actual_width} exceeded schedule max_rows={k_max}"
                );
                dynamic_widths.insert(actual_width);
            }
            if graph {
                executed_graph_widths.insert(actual_width);
            }
            let committed = round.output.len();
            ensure!(
                committed > 0 && committed <= actual_width,
                "Runner FAIL transaction committed extent"
            );
            let mut expected = if let Some(w) = verify_width {
                greedy[..(matched + 1).min(w).min(limit)].to_vec()
            } else {
                greedy[..committed].to_vec()
            };
            if let Some(index) = expected.iter().position(|id| Some(*id) == eos) {
                expected.truncate(index + 1);
            }
            let output_mismatches = usize::from(round.output != expected)
                + usize::from(round.accepted + 1 != committed);
            let boundary_mismatches = usize::from(model.position != ids.len() + committed)
                + usize::from(model.pending_token != expected.last().copied())
                + usize::from(
                    model.committed_prefix_ids
                        != ids
                            .iter()
                            .copied()
                            .chain(std::iter::once(anchor))
                            .chain(expected.iter().copied().take(committed - 1))
                            .collect::<Vec<_>>(),
                );
            let actual = chain_state(&model)?;
            let state_mismatches = actual
                .iter()
                .zip(&greedy_states[committed])
                .filter(|(a, b)| a != b)
                .count();
            let actual = stream.clone_dtoh(&model.work.logits)?;
            let logit_mismatches = actual
                .iter()
                .zip(&greedy_logits[committed - 1])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let draft = &mut model.spec.as_mut().context("chain missing")?.draft;
            let mut actual = Trace::default();
            draft.snapshot_kv(&mut actual, ids.len(), draft_rows)?;
            draft.restore_kv(ids.len(), 0)?;
            draft.inject(
                &greedy_taps,
                committed,
                ids.len(),
                super::super::super::draft::Compute {
                    ops: &model.ops,
                    blas: &mut model.blas,
                },
                None,
            )?;
            let mut expected_kv = Trace::default();
            draft.snapshot_kv(&mut expected_kv, ids.len(), draft_rows)?;
            let mut draft_mismatches = 0;
            for layer in 0..draft.config.num_hidden_layers {
                for name in [format!("kv_k_{layer}"), format!("kv_v_{layer}")] {
                    draft_mismatches += actual
                        .values(&name)?
                        .iter()
                        .zip(expected_kv.values(&name)?)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                }
            }
            let mismatches = output_mismatches
                + boundary_mismatches
                + state_mismatches
                + logit_mismatches
                + draft_mismatches;
            defects += mismatches;
            println!(
                "{} minicpm_chain_commit graph={graph} verify_width={actual_width} forced={verify_width:?} matched={matched} limit={limit} eos={eos:?} accepted={} committed={committed} output_mismatches={output_mismatches} boundary_mismatches={boundary_mismatches} state_bit_mismatches={state_mismatches} logits_bit_mismatches={logit_mismatches} draft_kv_bit_mismatches={draft_mismatches}",
                if mismatches == 0 { "PASS" } else { "DEFECT" },
                round.accepted
            );
        }
    }
    ensure!(
        executed_graph_widths == (crate::dspark::MIN_MOE_VERIFY_ROWS..=max_forced).collect(),
        "Runner FAIL not all expected verify widths 2..={max_forced} executed in graph mode: {executed_graph_widths:?}"
    );
    if k_max < rows {
        ensure!(
            !dynamic_widths.contains(&(k_max + 1)),
            "Runner FAIL dynamic selection chose forced-only width {} above k_max={k_max}",
            k_max + 1
        );
    }
    for graph in [false, true, false] {
        model.set_spec_graph(graph)?;
        model.reset()?;
        model.prefill(&ids, None)?;
        let before = chain_state(&model)?;
        let mut draft_before = Trace::default();
        model
            .spec
            .as_ref()
            .context("chain missing")?
            .draft
            .snapshot_kv(&mut draft_before, ids.len(), draft_rows)?;
        let pending = model.pending_token;
        let missing = if graph {
            model
                .spec
                .as_mut()
                .context("chain missing")?
                .forced_proposals = Some(greedy[..rows - 1].to_vec());
            model.spec.as_mut().context("chain missing")?.inject_graphs[7].take()
        } else {
            model.capture_fail_after_layer = Some((0, 1));
            None
        };
        let error = model
            .spec_round(8, None, "chain_error")
            .err()
            .context("Runner FAIL transaction fault was ignored")?;
        if graph {
            ensure!(
                error
                    .to_string()
                    .contains("MoE commit graph rows=8 missing before Ready"),
                "Runner FAIL graph fault reached a different failure: {error}"
            );
            model.spec.as_mut().context("chain missing")?.inject_graphs[7] = missing;
        } else {
            ensure!(
                error.to_string().contains("injected capture failure"),
                "Runner FAIL eager fault reached a different failure: {error}"
            );
        }
        model.capture_fail_after_layer = None;
        ensure!(
            model.poisoned,
            "Runner FAIL transaction fault was not exposed"
        );
        let mut error_mismatches = usize::from(chain_state(&model)? != before)
            + usize::from(
                model.position != ids.len()
                    || model.pending_token != pending
                    || model.committed_prefix_ids != ids,
            );
        let mut draft_after = Trace::default();
        let draft = &model.spec.as_ref().context("chain missing")?.draft;
        draft.snapshot_kv(&mut draft_after, ids.len(), draft_rows)?;
        for layer in 0..draft.config.num_hidden_layers {
            for name in [format!("kv_k_{layer}"), format!("kv_v_{layer}")] {
                error_mismatches += draft_before
                    .values(&name)?
                    .iter()
                    .zip(draft_after.values(&name)?)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
            }
        }
        defects += error_mismatches;
        println!(
            "{} minicpm_chain_error graph={graph} restored_bit_mismatches={error_mismatches} poisoned=true",
            if error_mismatches == 0 {
                "PASS"
            } else {
                "DEFECT"
            }
        );
        model.reset()?;
    }
    ensure!(
        wrong_layer > 0,
        "Runner FAIL wrong-layer capture control was not detected"
    );
    ensure!(
        perturbed_gdn_control > 0,
        "Runner FAIL planted gdn_inputs perturbation control was not detected"
    );
    println!(
        "{} minicpm_chain_verify_end chunk={chunk} rows=8 graph_widths=2..={max_forced} prefixes=7,17,65,125,129,224 accepted=0..8 wrong_layer_mismatches={wrong_layer} perturbed_gdn_mismatches={perturbed_gdn_control} defects={defects}",
        if defects == 0 { "PASS" } else { "DEFECT" }
    );
    ensure!(
        defects == 0,
        "DEFECT production chain verification cells={defects}"
    );
    Ok(())
}

#[test]
#[ignore = "GPU lease, published hybrid FP8, media fixtures/reference, output and pinned runtime"]
fn hybrid_media_identity_gpu() -> Result<()> {
    let model = PathBuf::from(std::env::var_os("FF_MODEL").context("media model missing")?);
    let specification = crate::model_spec::ModelSpec::read(&model)?;
    let qwen = specification.architecture == crate::model_spec::Architecture::Qwen35Text;
    ensure!(
        qwen || specification.architecture == crate::model_spec::Architecture::MiniCpmV47,
        "media gate architecture unsupported"
    );
    let vocab = specification.vocab as usize;
    let cases: &[&str] = if qwen {
        &["image_hockey", "image_boat"]
    } else {
        &[
            "image_hockey_16x",
            "image_boat_16x",
            "video_football_16x",
            "video_control_same_grid_16x",
            "video_control_clip2_16x",
        ]
    };
    let input_dir = PathBuf::from(
        std::env::var_os("FF_MINICPM_REFERENCE_DIR").context("media inputs missing")?,
    );
    let reference =
        PathBuf::from(std::env::var_os("FF_DENSE_REFERENCE").context("media reference missing")?);
    let output =
        PathBuf::from(std::env::var_os("FF_DENSE_EVIDENCE").context("media output missing")?);
    let settings = Settings {
        runtime_dir: Some(PathBuf::from(
            std::env::var_os("FLYINGFISH_RUNTIME_DIR").context("media runtime missing")?,
        )),
        prefix_cache: false,
        device_budget_gib: None,
        draft_model: None,
        tree_builder: TreeBuilder::Waves,
        spec_budget: SpecBudget::Chain,
        spec_graph: true,
        kv_format: crate::kv::KvFormat::Bf16,
    };
    #[cfg(feature = "rocm")]
    let limits: serde_json::Value = serde_json::from_str(if qwen {
        include_str!("../../../tests/fixtures/q38_reference.json")
    } else {
        include_str!("../../../tests/fixtures/minicpm_reference.json")
    })?;
    #[cfg(feature = "rocm")]
    ensure!(
        limits["media_limits"].is_object(),
        "media limits are not frozen"
    );
    Engine::calibrate(&model, 0, 4352, Some(129), settings.clone())?;
    let mut engine = Engine::load(&model, 0, 4352, Some(129), settings)?;
    ensure!(
        engine.model_spec.vocab as usize == vocab,
        "media vocabulary mismatch"
    );
    let mut first_logits = Vec::new();
    let mut defects = 0;
    for &case in cases {
        let filename = if qwen {
            format!("{case}-input.safetensors")
        } else {
            format!("{case}.safetensors")
        };
        let bytes = std::fs::read(input_dir.join(filename))?;
        let input = Input::read(&SafeTensors::deserialize(&bytes)?, case)?;
        let bytes = std::fs::read(reference.join(format!("{case}.safetensors")))?;
        let source = SafeTensors::deserialize(&bytes)?;
        ensure!(
            source.tensor("logits")?.shape() == [32, vocab],
            "media reference logits must be [32,vocab]"
        );
        let expected = floats(&source, "logits")?;
        let forced = integers(&source, "tokens")?;
        ensure!(
            forced.len() == 32 && forced.iter().all(|&id| id < vocab as u32),
            "media reference token extent/range"
        );
        let mut eager = Vec::new();
        for (repeat, graph) in [false, true, false].into_iter().enumerate() {
            engine.reset()?;
            let mut token = engine.prefill_multimodal(&input.packed(), None)?;
            let mut captured = Vec::with_capacity(expected.len());
            let mut predictions = Vec::with_capacity(32);
            let (mut max_error, mut rms_error) = (0.0f64, 0.0f64);
            let mut min_margin = f32::INFINITY;
            let mut cell_defects = 0;
            for step in 0..32 {
                if step > 0 {
                    engine.set_token(forced[step - 1])?;
                    token = engine.decode(graph, None)?;
                }
                ensure!(
                    engine.position == input.ids.len() + step,
                    "media absolute position mismatch"
                );
                let actual = engine.decode_logits()?;
                ensure!(
                    actual.len() == vocab && actual.iter().all(|v| v.is_finite()),
                    "nonfinite or incomplete media logits"
                );
                let bits = actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                if repeat == 0 {
                    eager.push((token, bits));
                } else {
                    cell_defects += usize::from(eager[step] != (token, bits));
                }
                let wanted = &expected[step * vocab..(step + 1) * vocab];
                let scale = (wanted.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>()
                    / vocab as f64)
                    .sqrt();
                ensure!(scale.is_finite() && scale > 0.0, "media reference RMS");
                let (mut largest, mut square) = (0.0f64, 0.0f64);
                let mut top = [f32::NEG_INFINITY; 2];
                for (&a, &b) in actual.iter().zip(wanted) {
                    let error = f64::from(a) - f64::from(b);
                    largest = largest.max(error.abs());
                    square += error * error;
                    if b > top[0] {
                        top = [b, top[0]];
                    } else if b > top[1] {
                        top[1] = b;
                    }
                }
                max_error = max_error.max(largest / scale);
                rms_error = rms_error.max((square / vocab as f64).sqrt() / scale);
                min_margin = min_margin.min(top[0] - top[1]);
                predictions.push(token);
                captured.extend(actual);
            }
            #[cfg(feature = "rocm")]
            {
                let bound = &limits["media_limits"][case];
                let maximum = bound["max_normalized"]
                    .as_f64()
                    .context("media max bound")?;
                let rms = bound["rms_normalized"]
                    .as_f64()
                    .context("media RMS bound")?;
                ensure!(
                    maximum.is_finite() && rms.is_finite(),
                    "nonfinite media bounds"
                );
                cell_defects += usize::from(max_error > maximum || rms_error > rms);
            }
            let calls = engine
                .qwen
                .as_ref()
                .context("media runtime absent")?
                .vision_encode_calls;
            ensure!(
                calls == 1,
                "media encoder calls={calls}, expected one per request"
            );
            engine.check_ready()?;
            if repeat == 0 {
                first_logits.push(captured[..vocab].to_vec());
                println!(
                    "MEDIA_DECODED {case} {:?}",
                    crate::tokenizer::decode(&model, &predictions)?
                );
            }
            let mut trace = Trace::default();
            trace.set_device(&engine.device.info.name)?;
            trace.add("logits".into(), vec![32, vocab], captured)?;
            trace.add(
                "tokens".into(),
                vec![32],
                predictions.iter().map(|&v| v as f32).collect(),
            )?;
            trace.save(&output.join(format!("{case}-repeat{repeat}.safetensors")))?;
            println!(
                "MEDIA_IDENTITY case={case} graph={graph} repeat={repeat} encoder_calls={calls} steps=32 max_normalized={max_error:e} rms_normalized={rms_error:e} min_reference_margin={min_margin} defects={cell_defects}"
            );
            defects += cell_defects;
        }
    }
    ensure!(
        first_logits
            .iter()
            .enumerate()
            .all(|(i, a)| first_logits[i + 1..].iter().all(|b| a != b)),
        "different media produced identical full-vocabulary outputs"
    );
    println!(
        "MEDIA_IDENTITY cells={} defects={defects} media_inputs={} encoder_calls={} graph_eager_repeat=true",
        3 * cases.len(),
        cases.len(),
        3 * cases.len()
    );
    ensure!(defects == 0, "media identity or model-definition DEFECT");
    Ok(())
}

pub(super) fn floats(source: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    let tensor = source.tensor(name)?;
    let values = match tensor.dtype() {
        Dtype::BF16 => tensor
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| bf16::from_bits(u16::from_le_bytes(*v)).to_f32())
            .collect::<Vec<_>>(),
        Dtype::F32 => tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect(),
        dtype => anyhow::bail!("reference {name} has floating dtype {dtype:?}"),
    };
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite reference {name}"
    );
    Ok(values)
}

pub(super) fn integers(source: &SafeTensors<'_>, name: &str) -> Result<Vec<u32>> {
    let tensor = source.tensor(name)?;
    match tensor.dtype() {
        Dtype::I64 => tensor
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|v| Ok(u32::try_from(i64::from_le_bytes(*v))?))
            .collect(),
        Dtype::I32 => tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| Ok(u32::try_from(i32::from_le_bytes(*v))?))
            .collect(),
        dtype => anyhow::bail!("reference {name} has integer dtype {dtype:?}"),
    }
}

pub(super) struct Input {
    pub(super) ids: Vec<u32>,
    pub(super) modalities: Vec<u8>,
    pub(super) pixels: Vec<f32>,
    pub(super) grids: Vec<ImageGrid>,
}

impl Input {
    pub(super) fn read(source: &SafeTensors<'_>, case: &str) -> Result<Self> {
        let ids = integers(source, "input_ids")?;
        ensure!(
            source.tensor("input_ids")?.shape() == [1, ids.len()],
            "input ids layout"
        );
        let mut input = Self {
            ids,
            modalities: Vec::new(),
            pixels: Vec::new(),
            grids: Vec::new(),
        };
        let (pixels, sizes) = match case.split('_').next() {
            Some("text") => return Ok(input),
            Some("image") => ("pixel_values", "target_sizes"),
            Some("video") => ("pixel_values_videos", "target_sizes_videos"),
            _ => anyhow::bail!("unknown native fixture case={case}"),
        };
        input.modalities = integers(source, "mm_token_type_ids")?
            .into_iter()
            .map(|v| Ok(u8::try_from(v)?))
            .collect::<Result<_>>()?;
        input.pixels = floats(source, pixels)?;
        let sizes = integers(source, sizes)?;
        ensure!(
            sizes.len().is_multiple_of(2),
            "reference target_sizes layout"
        );
        input.grids = sizes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|s| ImageGrid {
                t: 1,
                h: s[0],
                w: s[1],
            })
            .collect();
        Ok(input)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn video(source: &SafeTensors<'_>, model: &Path, case: &str) -> Result<Self> {
        use crate::serve::{
            detok::Shared,
            image::{Processor, VideoOptions},
        };
        let reference = Self::read(source, case)?;
        let directory = PathBuf::from(
            std::env::var_os("FF_VIDEO_REF_DIR").context("FF_VIDEO_REF_DIR missing")?,
        );
        let clip = match case {
            "video_football_16x" => "football_4s",
            "video_control_clip2_16x" => "sample_demo_1",
            "video_control_same_grid_16x" => "boat_pan_4s",
            _ => anyhow::bail!("unknown video fixture case={case}"),
        };
        let floor: serde_json::Value = serde_json::from_slice(&std::fs::read(
            directory.join("preprocess_video_noise_floor.json"),
        )?)?;
        let measured = floor["cases"]
            .as_object()
            .context("video pixel floor cases missing")?;
        ensure!(!measured.is_empty(), "video pixel floor has no cases");
        let tolerance = measured
            .iter()
            .try_fold(0.0f64, |bound, (name, cell)| -> Result<f64> {
                Ok(bound.max(
                    cell["tv_vs_pil_diff"]["max_abs"]
                        .as_f64()
                        .with_context(|| format!("video pixel floor {name} max_abs missing"))?,
                ))
            })? as f32
            + 4.0 * f32::EPSILON;
        ensure!(
            tolerance.is_finite() && tolerance > 0.0,
            "invalid video pixel floor {case}"
        );
        let spec = crate::model_spec::ModelSpec::read(model)?;
        let vision = spec.vision.as_ref().context("video vision spec missing")?;
        let first = reference
            .ids
            .iter()
            .position(|&id| id == vision.start_token)
            .context("video opening marker missing")?;
        let last = reference
            .ids
            .iter()
            .rposition(|&id| id == vision.end_token)
            .context("video closing marker missing")?;
        ensure!(first < last, "video marker extent {case}");
        let mut prompt = reference.ids[..first].to_vec();
        prompt.push(vision.video_token);
        prompt.extend_from_slice(&reference.ids[last + 1..]);
        let tokenizer = Shared::load(model)?;
        let processor =
            Processor::load(model, Some(&tokenizer))?.context("video processor missing")?;
        let bytes = std::fs::read(directory.join("inputs").join(format!("{clip}.mp4")))?;
        let url = format!("data:video/mp4;base64,{}", base64::encode(bytes));
        let planned = processor.plan_video(
            &url,
            VideoOptions {
                max_num_frames: 4,
                stack_frames: 1,
                max_slice_nums: Some(1),
            },
            4352,
        )?;
        let video = processor.prepare_planned_video(planned)?;
        let frames = video.frames.len();
        let (ids, prepared) = processor.expand(&prompt, Vec::new(), vec![video], 4352)?;
        let input = Self {
            ids,
            modalities: prepared.modality_ids,
            pixels: prepared.pixel_values,
            grids: prepared.grids,
        };
        ensure!(
            input.ids == reference.ids
                && input.modalities == reference.modalities
                && input.grids == reference.grids,
            "video preprocessing ids/modality/grid mismatch {case}"
        );
        ensure!(
            input.pixels.len() == reference.pixels.len()
                && input.pixels.iter().all(|value| value.is_finite()),
            "video pixel extent/finite check {case}"
        );
        let error = input
            .pixels
            .iter()
            .zip(&reference.pixels)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        ensure!(
            error <= tolerance,
            "video preprocess {case} pixel max={error} > {tolerance}"
        );
        let axes = input.packed().validate(&spec, 4352)?.packed_axes()?;
        let expected = integers(source, "canvas_position_ids")?;
        ensure!(
            expected.len() == 3 * input.ids.len(),
            "video reference canvas extent {case}"
        );
        ensure!(
            (0..3).all(
                |axis| (0..input.ids.len()).all(|row| i64::from(axes[3 * row + axis])
                    == i64::from(expected[axis * input.ids.len() + row]))
            ),
            "video canvas mismatch {case}"
        );
        ensure!(
            frames == integers(source, "sampled_frame_indices")?.len(),
            "video frame count mismatch {case}"
        );
        println!(
            "PASS minicpm_video_preprocess {case} frames={frames} modality=2 ids={} pixel_max={error} measured_floor={tolerance}",
            input.ids.len()
        );
        Ok(input)
    }

    pub(super) fn packed(&self) -> MultimodalInput<'_> {
        MultimodalInput {
            audio: &[],
            input_ids: &self.ids,
            modality_ids: &self.modalities,
            pixel_values: &self.pixels,
            grids: &self.grids,
        }
    }
}
