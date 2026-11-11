# Persistently cache the CMAPSS archive, with a manual override escape hatch

_Status: the CMAPSS example has moved out of this repository, to an application-layer repo (#186). This record is kept as history._

NASA's `data.nasa.gov` legacy URL that serves the CMAPSS zip is unreliable, and the previous code re-downloaded the full archive on every single loader call (three times per pipeline run, every run), turning routine transient failures into a chronic problem. We decided to fetch the archive at most once into a persistent cache — its directory resolved from `STAGEREG_CMAPSS_CACHE_DIR`, defaulting to a stdlib-computed per-user cache location — with retries (3 attempts, exponential backoff) and zip-validity checks around each download and each cache read, so a corrupted cache entry self-heals via re-download rather than poisoning every future call.

Because NASA's server may simply be unreachable regardless of retries, we also added `STAGEREG_CMAPSS_ARCHIVE` (and a matching optional loader parameter) so a user can point at a zip they obtained by other means, bypassing the network entirely and seeding the cache. We considered relying on caching alone, but rejected it: caching only helps once a download has succeeded at least once, and does nothing for a user whose network can't reach NASA's servers at all.

## Considered Options

- **Repo-local cache directory** (e.g. `.cache/` under the repo): rejected because every clone/worktree would re-download independently, and it tangles example data with the repo working tree.
- **A dependency like `pooch` or `platformdirs`** for cache-directory resolution and fetch orchestration: rejected to avoid adding a new dependency for something the standard library can resolve directly (`XDG_CACHE_HOME`/`~/.cache`).
- **Caching without a manual override**: rejected per above — doesn't help a user whose network can't reach NASA at all, only one who's had at least one past success.
