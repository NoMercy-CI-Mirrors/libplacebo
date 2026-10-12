# libplacebo — read-only mirror

Mirror of <https://code.videolan.org/videolan/libplacebo.git>, kept for the [NoMercy FFmpeg](https://github.com/NoMercy-Entertainment/nomercy-ffmpeg) build, which clones its dependencies by tag and has been stopped by that host refusing connections.

- Every upstream branch and tag is here under its own name. Clone with `--branch <tag>` exactly as you would upstream.
- This `ci-mirror` branch is the only thing that is not upstream's: it holds the sync workflow, the upstream URL (`UPSTREAM`) and the time of the last sync (`LAST_SYNC`). It is the default branch so that the schedule runs; a plain clone lands here, so always name the tag you want.
- Synced daily by `.github/workflows/sync.yml`, forced, using this repository's own token. Nothing is edited here by hand; fix things upstream.
