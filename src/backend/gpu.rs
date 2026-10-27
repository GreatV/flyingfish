//! Host objects selected at compile time; operator sources are shared.
#[cfg(feature = "cuda")]
pub use super::cuda::blas;
#[cfg(feature = "cuda")]
pub use super::cuda::{contracts, copy, cubin, decode, ops, state, verification};
#[cfg(feature = "rocm")]
#[path = "hip_blas.rs"]
pub mod blas;
#[cfg(feature = "rocm")]
pub use super::hip::{
    Context as CudaContext, DevicePtr, DevicePtrMut, DeviceRepr, Event as CudaEvent,
    Function as CudaFunction, LaunchConfig, Module as CudaModule, PinnedHostSlice, PushKernelArg,
    Slice as CudaSlice, Stream as CudaStream, ValidAsZeroBits, View as CudaView,
    ViewMut as CudaViewMut,
};
#[cfg(feature = "cuda")]
pub use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut,
    DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PinnedHostSlice, PushKernelArg,
    ValidAsZeroBits,
};

#[cfg(feature = "rocm")]
#[path = "cuda/contracts.rs"]
pub mod contracts;
#[cfg(feature = "rocm")]
#[path = "cuda/copy.rs"]
pub mod copy;
#[cfg(feature = "rocm")]
#[path = "cuda/cubin.rs"]
pub mod cubin;
#[cfg(feature = "rocm")]
#[path = "cuda/decode.rs"]
pub mod decode;
#[cfg(feature = "rocm")]
#[path = "cuda/ops.rs"]
pub mod ops;
#[cfg(feature = "rocm")]
#[path = "cuda/state.rs"]
pub mod state;
#[cfg(feature = "rocm")]
#[path = "cuda/verification.rs"]
pub mod verification;

#[cfg(test)]
#[path = "cuda/dense_tests.rs"]
mod tests;
