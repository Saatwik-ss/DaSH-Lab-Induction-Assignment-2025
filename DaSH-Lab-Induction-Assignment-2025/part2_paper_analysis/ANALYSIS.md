# vLLM and PagedAttention

While understanding the paper was not very difficult, trying to truly piece together how the system worked internally was much harder. Reading the paper gave a good conceptual overview, but actually tracing how PagedAttention, the block allocator, and the scheduler tied together in practice was a different challenge altogether. I could understand what each part did individually, like how blocks are allocated or how copy-on-write saves memory, but fitting everything into a mental picture of how vLLM handles one complete request from start to finish took quite a while.

Initially, I planned to go through the source code of vLLM itself to better understand it. However, the codebase was quite complex and modularized to a point that reading one file didn’t give me much clarity without tracing several others. It felt a bit like trying to understand an OS kernel, every function depended on another layer of abstraction. So I focused instead on dissecting the key ideas from the paper and understanding how the memory management works conceptually rather than implementation-wise.

Also used this video for understanding [https://www.youtube.com/watch?v=glyu_nQH0yw](LINK)

---

## PagedAttention

The most interesting part of the whole paper was the PagedAttention mechanism. Traditional attention stores the key-value (KV) cache in contiguous memory, which works fine when the sequence length is known and fixed. But in LLM serving, each user request generates text of unpredictable length, and that creates huge memory fragmentation.

PagedAttention fixes that by using an idea straight out of operating systems, paging. Instead of storing all the key and value vectors of a request in a single contiguous block, it splits them into blocks (or pages) of fixed size. Each of these blocks can be placed anywhere in GPU memory. So instead of reserving memory for the maximum possible sequence length, it only allocates blocks as needed, dynamically.

In simpler terms, each request gets its own little “virtual memory space” for the KV cache. Logical blocks represent the model’s idea of where its keys and values are, while the physical blocks are just wherever they fit in GPU memory. This setup removes internal fragmentation (unused reserved space) and external fragmentation (uneven memory gaps).

This idea alone gives a huge efficiency boost, the memory waste seen in existing systems like Orca and FasterTransformer goes down from around 60–80% to less than 10%.

---

## vLLM Rundown

vLLM is basically a serving engine built around PagedAttention. The architecture has three main parts: the scheduler, the KV cache manager, and the GPU workers.

The scheduler manages which requests are being served and decides when to allocate or free memory blocks. The KV cache manager handles the mapping between logical and physical blocks (kind of like a page table in an OS). GPU workers actually run the attention computations using the PagedAttention kernel.

Each GPU has a block allocator that divides its memory into these small KV blocks. Whenever a new token is generated, the manager decides whether to reuse a block, share it, or allocate a new one.

Another key feature is how vLLM supports memory sharing. For example:
- In parallel sampling, when one prompt is used to generate multiple outputs, all samples share the prompt’s cache and only diverge when needed.
- In beam search, it allows sharing of common prefixes between beams and performs copy-on-write only when beams start differing.

This drastically reduces redundant copies of the same cache and saves even more GPU memory.

The system also supports preemption and swapping. When GPU memory runs low, vLLM can either move less active blocks to CPU RAM (swapping) or simply recompute them later when needed. This is smart because the KV cache can always be regenerated from tokens if required.

---

## Performance

The results shown in the paper are quite impressive. On average, vLLM manages to handle about 2–4× higher throughput compared to Orca and significantly more than FasterTransformer. For smaller models like OPT-13B, this improvement is even more noticeable because memory was the limiting factor.

Interestingly, the authors found that the optimal block size for PagedAttention is around 16 tokens, small enough to prevent fragmentation but large enough for good GPU parallelism.

Even though the PagedAttention kernel has about 20–25% higher latency than traditional attention (due to block table lookups), the overall system performance still improves massively because it allows more requests to fit in GPU memory and increases batching.

---
# Technical Understanding

## a) Problem Analysis

The problem that this paper focuses on is the inefficiency of memory management during large language model serving. When an LLM is deployed for inference, every user request needs to store the key and value tensors from the attention layers in what is called the KV cache. This cache is needed for generating the next token in the sequence since transformers are autoregressive. The size of the KV cache grows with every new token that the model generates. This makes the workload memory-bound rather than compute-bound. The more memory is wasted, the fewer concurrent requests can be batched together, which directly reduces throughput and increases latency.

The issue becomes clearer when looking at how current systems work. In frameworks like PyTorch or TensorFlow, tensors are stored in contiguous memory blocks. So, existing systems that serve LLMs such as Orca or FasterTransformer allocate a single large continuous chunk of memory for every request, usually sized for the maximum possible sequence length. In reality, most user requests generate much shorter sequences. This means that a large portion of that reserved memory is never actually used, leading to internal fragmentation. There is also external fragmentation because each request reserves differently sized chunks of memory, leaving small gaps that cannot be reused. As a result, GPU memory gets filled quickly even when much of it is empty.

The paper shows this problem through measurements on a 13B parameter model running on an NVIDIA A100 GPU. The model weights take up around 65% of GPU memory, and the KV cache takes up about 30%. When the cache is managed poorly, actual usage drops to as low as 20–38% efficiency. Since model weights are fixed, this inefficiency in the cache becomes the main bottleneck. 

Its also talked that the problem becomes worse with longer sequences and advanced decoding methods like beam search or parallel sampling. In these cases, parts of the KV cache could be shared between different beams or samples, but existing systems treat each sequence separately and duplicate memory unnecessarily.

These issues make serving large models very costly. Processing an LLM query is already expensive. So improving throughput by fixing memory waste can have a major impact on both performance and cost. The motivating workloads in the paper include real serving traces from datasets like ShareGPT and Alpaca, which reflect realistic request lengths and variations. They were chosen specifically because they represent actual LLM usage patterns such as chat and instruction following.

---

## b) Proposed Solution

The solution proposed in the paper is PagedAttention, which rethinks how the KV cache is stored. It borrows the concept of paging from operating systems. Instead of keeping the cache in a contiguous memory region, the KV tensors are divided into fixed-size blocks. Each block stores a fixed number of tokens, and these blocks can be scattered across GPU memory. Logical blocks correspond to how the model thinks about memory, while physical blocks correspond to actual GPU memory locations. A mapping between the two is maintained in what the authors call block tables, similar to page tables in an OS.

This idea is implemented inside a system called vLLM. Its architecture has a centralized scheduler, a KV cache manager, and distributed GPU workers. The scheduler coordinates the execution and memory management. The cache manager maintains the block tables and performs allocation, reuse, and freeing of memory. GPU workers perform the attention computation using the PagedAttention kernel that can read and write from non-contiguous memory blocks.

When a request arrives, the scheduler only allocates as many blocks as needed for the current tokens rather than the maximum possible sequence length. As new tokens are generated, more blocks are allocated dynamically. If two requests share the same prefix, as in beam search, they can share the same physical blocks. If a shared block needs modification, vLLM applies a copy-on-write mechanism similar to what operating systems use for process memory. Each block has a reference counter that tracks how many sequences are sharing it. When the count reaches zero, that block can be freed and reused. 

This approach also supports preemption and swapping. When the GPU runs out of blocks, some can be swapped out to CPU memory or recomputed later. This allows the system to handle situations where active requests temporarily exceed GPU memory limits.

The implementation includes several CUDA kernel optimizations. The attention kernel was modified to work with blocks instead of continuous memory. Fused kernels were added for copying, reshaping, and writing blocks efficiently. These optimizations ensure that the paging mechanism does not cause large performance penalties during execution. The authors also tested different block sizes and found that around 16 tokens per block gave the best trade-off between GPU parallelism and memory fragmentation.

---

## c) Evaluation

The experiments were conducted on OPT models of 13B, 66B, and 175B parameters using NVIDIA A100 GPUs. The ShareGPT and Alpaca datasets were used to generate realistic request traces with different average input and output lengths. The authors compared their system, vLLM, with FasterTransformer and three variants of Orca.

The main metric was normalized latency, which measures average latency per token. Throughput was measured in requests per second at a fixed latency threshold. The authors also looked at the number of batched requests that could fit in memory and the amount of memory waste in KV cache storage.

Results showed that vLLM achieved 2 to 4 times higher throughput than Orca and up to 22 times higher than FasterTransformer. In terms of memory, vLLM reduced KV cache waste to less than 10%, while the others wasted between 60% and 80%. For workloads like beam search and parallel sampling, memory sharing saved up to 55% of KV memory. The paper also included ablations showing that block size and block-level sharing directly affect performance. Even though PagedAttention has about 20–25% higher kernel-level latency, the system-level gains far outweigh that overhead.

These results clearly support the claims that vLLM improves throughput and reduces memory waste while keeping latency similar. The findings are consistent across different model sizes and workloads, showing that the method scales.

---

# Critical Analysis

## Strengths

The paper’s main strength is the way it applies a well-known systems concept to a deep learning problem. PagedAttention turns memory management into a paging problem and provides a simple but effective fix for memory fragmentation. The design is easy to reason about conceptually and works well in practice. The implementation is also complete and open source, showing that this is not just a theoretical proposal but a working system.

Another strong point is that the paper does not rely on hardware-specific tricks or quantization. It instead solves the bottleneck at the system design level. The experiments are thorough, using realistic datasets, multiple models, and diverse decoding methods. The analysis includes ablations and microbenchmarks that make the claims convincing. The figures and explanations in the paper are clear and well-organized.

---

## Weaknesses

One limitation is that vLLM’s design assumes a high-speed connection between GPU and CPU memory. Swapping blocks to CPU RAM could become a bottleneck on slower interconnects or multi-node setups. Another limitation is the increase in code complexity. Managing block tables, reference counts, and copy-on-write logic introduces many moving parts that make the system harder to maintain or extend. There is also a small compute overhead in the attention kernel caused by additional block-table lookups.

The evaluation focuses mainly on transformer-based LLMs and assumes autoregressive decoding. The system’s applicability to other architectures or non-text tasks is unclear. While the authors show scaling on larger models, they do not analyze how the paging mechanism interacts with model parallelism beyond tensor parallelism.

---

## Specific Critiques

Are the evaluation metrics appropriate?  
In a pre defined context they are, normalized latency and throughput are suitable because they directly measure how efficiently the system serves requests. These metrics capture the practical goal of LLM serving, which is maximizing tokens generated per second per GPU, but there are lack of other very useful tests which are not mentioned which could have given a clearer picture

Is the baseline comparison fair?  
It is mostly fair, though the Orca system was reimplemented by the authors since the original source code was unavailable. This means that the exact implementation details might differ, even if the core idea is the same.

Are there overlooked edge cases?  
The system assumes that recomputation or swapping will always be possible without major delays. In extremely memory-limited environments, both may become expensive, and it is unclear how vLLM behaves under sustained overload.

Does the solution generalize beyond the tested scenarios?  
It generalizes well to transformer-based autoregressive models.However, It is not yet tested for multi-modal or reinforcement learning settings and thinking models.

Are there hidden costs not discussed?  
The added kernel overhead and the cost of managing the mapping tables are small but real. Also, recomputation or swapping might cause latency spikes during heavy load.


## Additional Critique: Limited Benefit for Single Requests

One limitation that becomes apparent when looking closely at vLLM’s results is that the gains mostly come from batching multiple requests together. The paper’s main improvements arise from how it handles the memory fragmentation that occurs when several requests of different lengths are served simultaneously. By using block-level allocation, vLLM can pack multiple active requests efficiently into GPU memory, which improves overall throughput.  

However, for a single large request, such as running an LLM with an extremely long input context, the benefits of PagedAttention become minimal. When only one sequence is being processed, there is no fragmentation between multiple requests, and hence no wasted space to recover. The memory layout is already continuous and fully occupied by that single sequence. In such a case, vLLM’s paging mechanism introduces only a small amount of overhead due to block management without any meaningful reduction in total memory use.  

This is particularly relevant for anyone hoping to extend the context length of models like LLaMA on consumer-grade GPUs with limited VRAM. While PagedAttention is excellent for improving throughput during concurrent serving of many short or mid-length requests, it does not directly make a single long sequence cheaper to run. The KV cache for a single sequence still grows linearly with sequence length, and paging does not reduce the actual memory footprint per token. It only improves how memory is allocated and reused among requests.

The authors also do not address whether the paging approach could be extended to offload parts of a single sequence’s KV cache to slower memory tiers in real time. Although they mention swapping and recomputation as preemption strategies, those mechanisms are used when GPU memory is full due to multiple requests, not for a single continuous sequence. In practice, this means vLLM does not help users who want to run very long contexts, such as 64k or 100k tokens, on a single GPU.  

This is an important critique because many researchers and developers working on long-context LLMs face exactly this problem. Techniques like context compression, sliding window attention, or streaming KV cache are more relevant for that scenario, but the paper does not make this distinction clear.  

In summary, vLLM is a strong contribution for high-throughput inference but not a solution for increasing the maximum context length of a model on limited hardware. Its focus is on efficient sharing and packing of GPU memory across multiple active requests rather than reducing the absolute per-request memory requirements.
---

## Reflection

This paper stood out because it wasn’t just about algorithmic optimization but more about systems-level design, taking something as low-level as paging from OS concepts and applying it to LLM memory management. It’s rare to see an idea borrowed so directly and effectively from a completely different field.

What really clicked for me is how the authors saw the KV cache problem not as a neural network issue but as a memory allocation issue. Instead of tuning models or quantizing weights, they just redesigned how memory is managed.

The architecture feels scalable and practical, though I do think implementing it from scratch would be extremely challenging because of the amount of coordination needed between CUDA kernels, memory allocators, and schedulers.

Overall, vLLM feels like a bridge between deep learning and operating systems, it uses traditional system design principles to fix what is otherwise seen as a deep learning bottleneck. It’s one of those works where the elegance is in how simple the final idea is once you see it: break the KV cache into pages, allocate on demand, and stop wasting memory. Simple, but genius.


