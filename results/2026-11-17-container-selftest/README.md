# The container image, built and self-tested, 2026-11-17

**No model was called and nothing was paid for.** `python3 scripts/container_selftest.py`
was run against a real Docker daemon (Docker 29.9.3, macOS arm64) on the files of
commit `2cb41a2`.

- **The image built.** It is 861 MB.
- **All 18 checks passed.** The script's output is `selftest.txt`.
- **The provider was a stand-in**, a second container. No provider key was involved.

## The checks

| check | result |
| --- | --- |
| image builds | `docker compose build tine` |
| containers start | `docker compose up -d --wait` |
| container healthy | status running, health healthy |
| health | status 200, routing frontier-default, shadow off |
| token needed | 401 with no token, 401 with a wrong one |
| request | status 200, ledger recorded |
| streamed request | status 200, ledger recorded |
| savings | tenant `selftest`, 2 new requests, known answer cost $0.000270 |
| ledger copy | `backup-ledger` and `docker compose cp`: the copy on the host holds 2 of 2 requests |
| savings report page | a page of 11,878 characters, made inside the container from the copy and the policies file |
| runs as uid 10001 | `id -u` printed 10001 |
| read-only except /data, /tmp | `/app`, `/config` and `/var/tmp` not writable; `/data` and `/tmp` writable |
| image holds only the package | `/app` holds `pyproject.toml`, `requirements-dev.lock`, `requirements.txt`, `tine` |
| policy edit, then restart | `/health` reports routing by policy; a request was served by the cheap stand-in at stage 0 |
| ledger kept over a restart | 3 requests in the ledger after the restart, 2 before it |
| stop with a stream open | stopped in 20.6 s (Docker kills at 30), exit code 143 |
| ledger whole after the stop | 4 requests in the ledger after the stop and start |
| clean up | containers, network and test ledger volume removed |

The costs in the savings check are the stand-in's made-up prices.

## Before this

An image of the older Dockerfile, without the lock-file constraints, was built earlier
the same day. Its container served 40 paid requests through a route policy
(`docs/DEPLOY.md`). That is the only time a container of Tine has called a real model.

## Limits

- **One machine.** One build, one run, on one Mac. Not tried on Linux or amd64.
- **A stand-in provider.** No real provider, key, model or price.
- **Not a deployment.** No gateway, no TLS, no customer, no traffic beyond the script's
  own handful of requests.
- **The image of these files has served no paid request.**

## Reproduce

```sh
python3 scripts/container_selftest.py
```

It needs Docker running and nothing else (`docs/DEPLOY.md`, Check the image with one
command).
