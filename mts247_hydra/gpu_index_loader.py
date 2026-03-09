"""
gpu_index_loader.py
-------------------
Load a FAISS IVF GPU index from the packed on-disk cache without pulling
inverted-list data into RAM.

Strategy
--------
We load the original .faiss shard with faiss.IO_FLAG_MMAP, which memory-maps
the file instead of reading it into RAM.  The OS only faults in pages that are
actually accessed.  Since the packed-cache fast path in
IVFBase::copyInvertedListsFrom() reads everything from the
gpu_codes_all.{bin,meta} files on disk and never touches the InvertedLists
object's code/id data, those pages are never faulted in — the inverted-list
data costs zero RAM.

Pages that ARE read:
  * Index header / metadata  (a few KB)
  * Coarse quantizer centroids  (~nlist * d * 4 bytes, already tiny)

If the cache is cold (first ever run), the slow path fires and will read the
mmap'd pages to build the cache -- but only once.  All subsequent runs are
cache hits at zero RAM cost.
"""

import os
import struct
import numpy as np
import faiss


# ---------------------------------------------------------------------------
# .meta file layout (mirrors the C++ PackedCacheHeader / writePackedMeta)
# ---------------------------------------------------------------------------
_MAGIC       = 0x4653504B5F4D4554   # "FSPK_MET"
_VERSION     = 1
_HEADER_FMT  = "<QQQQQQ"            # 6 x uint64_t  (little-endian)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _read_meta(meta_path: str, expected_nlist: int):
    """
    Parse the binary .meta file written by IVFBase::writePackedMeta().

    Returns
    -------
    list_sizes      : np.ndarray  shape (nlist,)  dtype int64
    code_offsets    : np.ndarray  shape (nlist,)  dtype int64
    index_offsets   : np.ndarray  shape (nlist,)  dtype int64
    total_gpu_bytes : int
    total_idx_bytes : int
    indices_options : int  (read from file -- do not assume it matches your cloner)
    """
    with open(meta_path, "rb") as f:
        raw = f.read(_HEADER_SIZE)

    magic, version, nlist, idx_opts, total_gpu, total_idx = struct.unpack(_HEADER_FMT, raw)

    if magic != _MAGIC:
        raise ValueError(f"Bad magic in {meta_path}: {magic:#x}")
    if version != _VERSION:
        raise ValueError(f"Unsupported version {version} in {meta_path}")
    if nlist != expected_nlist:
        raise ValueError(f"nlist mismatch: meta has {nlist}, expected {expected_nlist}")

    array_bytes = int(nlist) * 8
    with open(meta_path, "rb") as f:
        f.seek(_HEADER_SIZE)
        list_sizes    = np.frombuffer(f.read(array_bytes), dtype=np.uint64).astype(np.int64)
        code_offsets  = np.frombuffer(f.read(array_bytes), dtype=np.uint64).astype(np.int64)
        index_offsets = np.frombuffer(f.read(array_bytes), dtype=np.uint64).astype(np.int64)

    return list_sizes, code_offsets, index_offsets, int(total_gpu), int(total_idx), int(idx_opts)


def _verify_cache_files(cache_dir: str, total_gpu_bytes: int) -> None:
    """Raise RuntimeError if the codes file is missing or the wrong size."""
    codes_path = os.path.join(cache_dir, "gpu_codes_all.bin")
    if not os.path.exists(codes_path):
        raise RuntimeError(
            f"Cache miss: {codes_path} does not exist. "
            "Run a warmup pass first so the cache is written."
        )
    actual = os.path.getsize(codes_path)
    if actual != total_gpu_bytes:
        raise RuntimeError(
            f"Cache stale: {codes_path} is {actual} bytes, expected {total_gpu_bytes}."
        )


def load_ivf_gpu_index_from_cache(
    res:            faiss.GpuResources,
    gpu_id:         int,
    shard_path:     str,
    nlist:          int,
    cache_dir:      str = "/dev/shm",
    cloner_options: faiss.GpuClonerOptions = None,
) -> faiss.GpuIndexIVF:
    """
    Build a GPU IVF index from the packed disk cache.

    The shard .faiss file is opened with IO_FLAG_MMAP so its inverted-list
    pages are never faulted into RAM on the cache-hit fast path.

    The indices_options value is read directly from the .meta file so it
    always matches what the cache was written with, regardless of what the
    cloner_options specify.

    Parameters
    ----------
    res             : faiss.GpuResources
    gpu_id          : which GPU to use
    shard_path      : path to the original .faiss shard file (mmap'd, ~0 RAM)
    nlist           : number of IVF lists (used to validate the .meta file)
    cache_dir       : directory containing gpu_codes_all.{bin,meta}
                      (must match FAISS_GPU_PACKED_CACHE_PATH)
    cloner_options  : GpuClonerOptions; sensible defaults used if None

    Returns
    -------
    faiss.GpuIndexIVF  ready for search
    """
    # Must be set before C++ reads the env var inside index_cpu_to_gpu.
    os.environ["FAISS_GPU_PACKED_LISTS"]      = "1"
    os.environ["FAISS_GPU_PACKED_CACHE_PATH"] = cache_dir

    # ------------------------------------------------------------------
    # 1. Read the meta file — indices_options comes FROM the file, not
    #    from the caller, so it always matches what was cached.
    # ------------------------------------------------------------------
    meta_path = os.path.join(cache_dir, "gpu_codes_all.meta")
    if not os.path.exists(meta_path):
        raise RuntimeError(
            f"Meta file not found: {meta_path}. "
            "The packed cache has not been written yet -- run a warmup first."
        )

    _, _, _, total_gpu_bytes, _, cached_indices_options = _read_meta(meta_path, nlist)
    _verify_cache_files(cache_dir, total_gpu_bytes)

    # ------------------------------------------------------------------
    # 2. Build cloner options, forcing indicesOptions to match the cache.
    # ------------------------------------------------------------------
    if cloner_options is None:
        cloner_options = faiss.GpuClonerOptions()
    # Override whatever was passed -- must match the cached value or the
    # C++ meta validation will fail and fall through to the slow path.
    cloner_options.indicesOptions = cached_indices_options

    # ------------------------------------------------------------------
    # 3. mmap the shard -- header + quantizer pages are read, inverted-list
    #    pages are never touched on the cache-hit path.
    # ------------------------------------------------------------------
    cpu_index = faiss.read_index(shard_path, faiss.IO_FLAG_MMAP)

    # ------------------------------------------------------------------
    # 4. Transfer to GPU -- hits the packed-cache fast path, uploads from
    #    disk cache, never reads the mmap'd inverted-list pages.
    # ------------------------------------------------------------------
    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index, cloner_options)

    # Release the mmap immediately -- GPU index is fully self-contained.
    del cpu_index
    return gpu_index


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def save_coarse_quantizer(cpu_ivf_index: faiss.IndexIVF, path: str) -> None:
    """Save just the coarse quantizer centroids from a CPU IVF index."""
    faiss.write_index(cpu_ivf_index.quantizer, path)
    print(f"Saved coarse quantizer ({cpu_ivf_index.nlist} centroids, d={cpu_ivf_index.d}) to {path}")


def load_coarse_quantizer(path: str) -> faiss.Index:
    return faiss.read_index(path)
