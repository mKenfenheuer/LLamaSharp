using System;

namespace LLama.Native;

/// <summary>
/// A LoRA adapter which can be applied to a context for a specific model
/// </summary>
public class LoraAdapter
{
    /// <summary>
    /// The model which this LoRA adapter was loaded with.
    /// </summary>
    public SafeLlamaModelHandle Model { get; }

    /// <summary>
    /// The full path of the file this adapter was loaded from
    /// </summary>
    public string Path { get; }

    /// <summary>
    /// Native pointer of the loaded adapter, will be automatically freed when the model is unloaded
    /// </summary>
    internal IntPtr Pointer { get; }

    /// <summary>
    /// Indicates if this adapter has been unloaded
    /// </summary>
    internal bool Loaded { get; private set; }

    internal LoraAdapter(SafeLlamaModelHandle model, string path, IntPtr nativePtr)
    {
        Model = model;
        Path = path;
        Pointer = nativePtr;
        Loaded = true;
    }

    /// <summary>
    /// Unload this adapter
    /// </summary>
    public void Unload()
    {
        Loaded = false;
        llama_adapter_lora_free(Pointer);
    }

    // Functions to access the adapter's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern int llama_adapter_meta_val_str(IntPtr adapter, string key, byte[] buf, int buf_size);

    // Get the number of metadata key/value pairs
    [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern int llama_adapter_meta_count(IntPtr adapter);

    // Get metadata key name by index
    [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern int llama_adapter_meta_key_by_index(IntPtr adapter, int i, byte[] buf, int buf_size);

    // Get metadata value as a string by index
    [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern int llama_adapter_meta_val_str_by_index(IntPtr adapter, int i, byte[] buf, int buf_size);

    // Get the invocation tokens if the current lora is an alora
    [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern ulong llama_adapter_get_alora_n_invocation_tokens(IntPtr adapter);

    [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern unsafe LLamaToken* llama_adapter_get_alora_invocation_tokens(IntPtr adapter);

    // Manually free a LoRA adapter. loaded adapters will be free when the associated model is deleted
    [DllImport(NativeApi.libraryName, CallingConvention = CallingConvention.Cdecl)]
    static extern void llama_adapter_lora_free(IntPtr adapter);
}