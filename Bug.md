# Known Bugs

## [OPEN] Dyn-HeiProMap: partition file too short after repeated vertex additions/removals

**Affects:** `amrex_dynmap_video.py` phase 2 — long simulations with dynamic AMR (e.g. `advection_vortex`)

**Symptom:**
```
IndexError: list index out of range
  File "amrex_dynmap_video.py", line 428, in get_partition
    return [raw[self._cell_to_id[cell_key(*cell)]] for cell in curr_nodes]
```
Crashes around frame 475/601 on the vortex simulation (20 s).

**Root cause:**
`DynHeiProMap` assigns vertex IDs monotonically via `_next_id` — IDs are never reused after a `-v` removal. After many add/remove cycles the maximum live ID grows well beyond the number of currently active vertices.

The partition file format specifies one entry per vertex ID (0-based), with inactive slots written as 0, so the file length should always equal `_next_id`. Instead, `Dyn-HeiProMap` writes only the active vertices, producing a file shorter than `_next_id`. Indexing into it with a high ID then raises `IndexError`.

**Fix needed in Dyn-HeiProMap:**
`save-partition` must write entries for all IDs 0 … max_id (inclusive), padding removed/inactive vertices with 0, so the file length equals the total number of IDs ever assigned.
