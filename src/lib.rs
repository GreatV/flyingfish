#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("features `cuda` and `rocm` are mutually exclusive");

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub device: usize,
    pub runtime_dir: std::path::PathBuf,
    pub verbose: bool,
}

pub mod audio;
pub mod backend;
pub mod bench;
pub mod canary;
pub mod checkpoint;
pub mod config;
pub mod cpu;
pub mod diarization;
pub mod dspark;
pub mod engram;
pub mod eou;
pub mod experts;
pub mod fp8;
pub mod gdn_plan;
pub mod gguf;
pub mod kv;
pub mod linear_plan;
pub mod media;
pub mod minimax_h3;
pub mod model;
pub mod model_spec;
pub mod mtp;
pub mod multitalker;
pub mod music3;
pub mod nemotron;
pub mod op_contract;
pub mod parakeet;
pub mod pixal3d;
pub mod prefill;
pub mod prefix_state;
pub mod serve;
pub mod spec;
pub mod tensors;
pub mod tokenizer;
pub mod trace;
pub mod tree;
pub mod tree_replay;
pub mod verify;
pub mod vision_input;
pub mod work_layout;

pub mod sparse;
