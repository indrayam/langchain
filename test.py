from llama_cpp import Llama
llm = Llama(
    model_path="./ggml-model-q4_0.bin", n_ctx=2048, seed=0, n_threads=None, n_batch=512, use_mmap=True, use_mlock=False, low_vram=False, n_gpu_layers=1)

output = llm("Explain the difference between nuclear fission and fussion", temperature=0.7, top_p=0.9,
             top_k=20, repeat_penalty=1.15, mirostat_mode=0, mirostat_tau=5, mirostat_eta=0.1)
print(output["choices"][0]["text"])
