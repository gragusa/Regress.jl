# GPU detection based on platform
# - On macOS: try Metal (Apple Silicon GPU)
# - On other platforms: try CUDA (NVIDIA GPU)

const GPU_METHOD = if Sys.isapple()
    try
        using Metal
        Metal.functional() ? :Metal : nothing
    catch
        nothing
    end
else
    try
        using CUDA
        CUDA.functional() ? :CUDA : nothing
    catch
        nothing
    end
end

const GPU_AVAILABLE = GPU_METHOD !== nothing
