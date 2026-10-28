//! Host objects selected at compile time; operator sources are shared.
pub use super::cuda::blas;
pub use super::cuda::{contracts, copy, cubin, decode, ops, state, verification};
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

#[cfg(test)]
#[path = "cuda/dense_tests.rs"]
mod tests;
