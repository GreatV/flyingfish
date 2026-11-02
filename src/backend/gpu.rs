//! Host objects selected at compile time; operator sources are shared.
pub use super::cuda::blas;
pub(crate) use super::cuda::linear;
pub use super::cuda::{
    block, contracts, copy, cubin, decode, flash, ops, sample, state, verification,
};
#[cfg(feature = "rocm")]
pub use super::hip::{
    Context as CudaContext, DevicePtr, DevicePtrMut, DeviceRepr, Event as CudaEvent,
    Function as CudaFunction, Graph as CudaGraph, LaunchConfig, Module as CudaModule,
    PinnedHostSlice, PushKernelArg, Slice as CudaSlice, Stream as CudaStream, ValidAsZeroBits,
    View as CudaView, ViewMut as CudaViewMut,
};
#[cfg(feature = "cuda")]
pub use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaGraph, CudaModule, CudaSlice, CudaStream, CudaView,
    CudaViewMut, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PinnedHostSlice, PushKernelArg,
    ValidAsZeroBits,
};

#[cfg(test)]
#[path = "cuda/dense_tests.rs"]
mod tests;
