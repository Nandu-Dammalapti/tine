# Deploying the proxy in a container

**No Docker daemon has built or run the image yet.** The `Dockerfile` and `docker-compose.yml` were
written on a machine without Docker, and not every step of the Dockerfile has been run.

- Run by hand outside Docker on 2026-11-17, on a Mac: the command that writes `requirements.txt`, a
  `pip` dry-run resolve of its lines for Linux (nothing was installed), the image's command (on
  `127.0.0.1`, not `0.0.0.0`), the health check command, requests and a stop.
- Not run: the base image, the `pip install` itself, the two `COPY` steps, the user-creation step
  (`groupadd`, `useradd`, `chown`, `chmod`), `USER` and `VOLUME`.

The image's command was run from a read-only copy of the package against a stand-in provider.
`docker compose config`, which needs no daemon, accepted the compose files.
[Checked without Docker](#checked-without-docker) lists what that covered. It does not cover the build
itself, Linux, the non-root user, the read-only root filesystem or the ledger volume. Expect to fix
something on the first real build, and treat every `docker` command below as untested.

The quickest first run needs no key and costs nothing: `python3 scripts/container_selftest.py` builds
the image and tests a running container against a stand-in provider. See
[Check the image with one command](#check-the-image-with-one-command).

This page covers packaging only. What the proxy does, its routes and its limits are in
[PROXY.md](PROXY.md). Frozen policies are in [POLICY.md](POLICY.md).

## What is in the image

- `python:3.12-slim-bookworm`, pinned by tag. Pin it by digest too before a real deployment.
- The dependencies listed in `pyproject.toml`, at the versions in `requirements-dev.lock`. The lock is
  given to `pip` as a constraints file, so the image holds the versions the tests ran on. The lock's
  development tools (pytest, ruff) are not installed: a constraints file installs nothing by itself.
- The `tine` package. No tests, results, notebooks, scripts, benchmark data or local artifacts: the
  `.dockerignore` admits only `pyproject.toml`, `requirements-dev.lock` and `tine/`.
- No credential. The image has no `ARG`, and its `ENV` lines set only paths and Python switches.

The container runs `python -m tine serve --host 0.0.0.0 --port 8000` as user 10001, not root. Its
health check calls `GET /health` with the Python standard library. The check ignores `HTTP_PROXY`, so
it still reaches the loopback address when you give the container an egress proxy.

The proxy does not load numpy, pandas, scikit-learn or matplotlib, although the image holds them for
`python -m tine freeze-policy`. Outside Docker it was ready 0.3 seconds after it was started.

## Run it

Start Docker Desktop, then check the commands in a new terminal:

```bash
docker --version
docker compose version
```

If the shell answers `command not found`, Docker Desktop has not put its tools on your `PATH`. Turn on
the system-wide install in Docker Desktop's settings (Advanced), or run this in each terminal:

```bash
export PATH="$HOME/.docker/bin:$PATH"
```

Add the folder, do not call `docker` by its full path alone. The folder also holds
`docker-credential-desktop`, which Docker Desktop normally sets as the credential store, and a build
needs it on `PATH` to pull the base image. On the Mac this was written on, `docker` was not on `PATH`
after the install, and `~/.docker/bin` held `docker`, `docker-compose` and `docker-credential-desktop`.
The pull itself was not tried.

Make the configuration before any `docker compose` command. Until `deploy/tine.env` exists compose
stops with `env file .../deploy/tine.env not found`. That was seen with `docker compose config`;
`build` and `up` load the same file. `docker build -t tine-proxy:dev .` builds the same image without
reading it, and so does the one-command check below.

```bash
cp deploy/tine.env.example deploy/tine.env
mkdir -p deploy/config
cp deploy/checks.example.json deploy/config/checks.json
cp deploy/shadow-policies.example.json deploy/config/shadow-policies.json
```

Edit `deploy/tine.env`. Every value in the example is a placeholder: the keys, the model names and the
prices are made up. The comments in that file describe each variable. `deploy/tine.env` and
`deploy/config/` are ignored by git through `deploy/.gitignore`. Keep credentials out of every other
file.

Four things to do in that file before the first start:

- **Replace the tenant token.** The proxy cannot tell a placeholder from a real token. If
  `TINE_TENANT_KEYS` is left as it is, the proxy starts and the placeholder printed in this repository
  works as a token.
- **Set the provider key, the frontier model and its price.** Put your own key in
  `TINE_ANTHROPIC_API_KEY` and the exact model name in `TINE_ANTHROPIC_FRONTIER_MODEL`. Then rename the
  frontier's entry in `TINE_MODEL_PRICES` to `anthropic:<that model name>` and put your prices in it.
  The example's prices are keyed by the example model names, so if you skip this every cost is
  recorded as unknown. The proxy also starts with the placeholder key; requests then fail at the
  provider.
- **Comment out `TINE_CHECKS` and `TINE_SHADOW_POLICIES`** unless you already have a policy of your
  own. The example policy names the example frontier model. Once you set
  `TINE_ANTHROPIC_FRONTIER_MODEL` to a real model, the proxy refuses to start with the example policy
  still in place. With both lines commented out the proxy sends every request to the frontier model
  you configured and records it.
- **Check who can read the files.** `chmod 600 deploy/tine.env`: only Docker Compose on the host reads
  it. The files in `deploy/config/` are read inside the container by uid 10001, so they must be
  readable by that user (mode 644 is enough, they hold no credential).

```bash
docker compose build
docker compose up -d --wait   # returns when the container is healthy, or says why it is not
docker compose ps
docker compose logs tine
```

`docker compose ps` shows `health: starting` and then `healthy`. By Docker's documentation, Engine 25
and later probe every 5 seconds while a container starts, so expect `healthy` after about 5 seconds;
an older engine waits for the first 30-second interval. Neither was seen here.

If another program already listens on port 8000 of the host, pick another host port. The address stays
`127.0.0.1`:

```bash
TINE_HOST_PORT=8010 docker compose up -d --wait
```

### Check that it answers

`/health` needs no token and answers 200 even with placeholder keys, so it shows only that the process
is up. These four commands show that the container works:

```bash
read -rs TOKEN    # paste the tenant token and press Enter; it is not shown or kept in the history
curl -s http://127.0.0.1:8000/health
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/v1/savings
curl -s -D - http://127.0.0.1:8000/v1/messages \
  -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"anything","max_tokens":16,"messages":[{"role":"user","content":"Say hi"}]}'
curl -s http://127.0.0.1:8000/v1/savings -H "Authorization: Bearer $TOKEN"
```

What to expect, in order:

1. `{"status":"ok","routing":"frontier-default","storage":"sqlite-development","shadow":"off"}`.
2. `401`, because no token was sent.
3. Status 200, with the headers `x-tine-model` (your frontier model) and `x-tine-ledger: recorded`.
   **This is a paid call** on your provider account, of at most 16 output tokens. A 502 means the
   proxy could not reach the provider: the log line says which host (see below).
4. `"requests": 1`, `"unknown_answer_cost_requests": 0` and a `known_answer_cost_total_usd` above 0.
   `unknown_cost_requests` and `unknown_routing_cost_requests` also read 1. That is expected: without
   a route policy the routing cost of a request is always unknown (PROXY.md), so those two count every
   request. A nonzero `unknown_answer_cost_requests` means the model that answered has no entry in
   `TINE_MODEL_PRICES`.

These results are from the image's command run outside Docker against a stand-in provider.
`scripts/container_smoke.py` runs the same sequence for you
([Test the container on your own machine](#test-the-container-on-your-own-machine)).

### When it does not start

The proxy refuses to start when the configuration is invalid. The last line of the log is then
`tine serve: the configuration was refused.` followed by the reason, and the container restarts in a
loop until the configuration is fixed. Read it with `docker compose logs --tail 5 tine`. Reasons you
are likely to meet first, each seen outside Docker:

| Reason in the log | Cause |
| --- | --- |
| `TINE_TENANT_KEYS is not valid JSON (...)` | Quotes around the value, or a missing brace. The same for `TINE_MODEL_PRICES`, `TINE_ALLOWED_MODELS` and `TINE_DENIED_FIELDS`. The value is never printed |
| `TINE_TIMEOUT_SECONDS must be a number, with no unit or other text` | A value such as `30s`. Every numeric variable answers this way |
| `TINE_VERIFIED_CACHE must be 1 or true, or 0 or false` | Any other word in one of the three switches |
| `Configure nonempty TINE_TENANT_KEYS before starting the proxy` | No tenant token, or no env file |
| `<VARIABLE>=<path> could not be loaded as JSON: <ErrorType>: <detail>` | A policy or check file (`TINE_ROUTE_POLICIES`, `TINE_SHADOW_POLICIES`, `TINE_CHECKS`, `TINE_TENANT_CHECKS`) that is missing from `deploy/config/`, unreadable, not UTF-8 or not valid JSON |
| `shadow_policies: A policy's frontier must be the provider's configured frontier model` | The frontier was changed and the example policy left in place |
| `checks: bearer_env 'CI_WEBHOOK_TOKEN' is not set, or is empty, ...` | A webhook check names a token variable the env file does not set |
| `route_policies: ... has agreement stages but no answer reader ...` | A route policy that compares answers has no `answer_pattern`, and `TINE_ANSWER_PATTERN` is unset |

Other refusals: a policy for a tenant that has no token, a named check that is not configured, shadow
sampling without a spending cap. The startup log also has one line per loaded policy, with its tenant,
provider, task type, answer reader and whether the file was edited after freezing.

When the proxy starts but a request gets 502 `Upstream request failed`, the proxy could not reach the
provider. The log has one line for it, with the host and the kind of failure and nothing else:

```text
WARNING:     tine.proxy: no reply from the anthropic upstream at 127.0.0.1:9000 for model 'your-model': ConnectError
```

Two causes are likely in a container:

- **`127.0.0.1` or `localhost` in a base URL is the container itself**, not your machine. For a model
  server on the Docker host use `http://host.docker.internal:PORT` (Docker Desktop), or the service
  name when the server runs in the same compose project.
- **Proxy variables apply to the provider address too.** If you give the container `HTTP_PROXY` or
  `HTTPS_PROXY`, add a private provider host to `NO_PROXY`. Outside Docker, a dead `HTTP_PROXY` with
  the provider on `127.0.0.1` gave this 502, and `NO_PROXY=127.0.0.1` cleared it.

### Stopping and restarting

The proxy reads `deploy/tine.env` and the files in `deploy/config/` only when it starts. After you
change any of them:

```bash
docker compose up -d --force-recreate --wait
curl -s http://127.0.0.1:8000/health
```

Plain `docker compose up -d` is not enough. Compose recreates a container only when the compose
configuration or the image changed. An edited env file counts as a change. An edited file in
`deploy/config/` does not, so `up -d` alone leaves the old policy running. `docker compose restart tine`
rereads the mounted files but not the env file. `--force-recreate` covers both. This is Compose's
documented behaviour; it was not seen here.

When the container is told to stop, the proxy lets requests in flight finish for 20 seconds. Then it
cuts whatever is still open, a stream included, and closes the ledger. The caller of a cut request
sees a broken connection. Measured outside Docker: an idle proxy stopped in 0.1 to 0.3 seconds; one holding
a stream that never ends stopped in 20.1 seconds; the ledger was closed to a single file both times.
Compose waits 30 seconds (`stop_grace_period`) before it kills the container, so the proxy is gone
first. If you want longer requests to finish, raise both: add `--shutdown-seconds N` to the command
in a compose override, and set `stop_grace_period` to at least N plus 10.

After a stop, `docker compose ps -a` will probably show exit code 143. That is a clean stop: the server
shuts down, then ends by the same signal Docker sent (128 + 15). Exit code 137 means Docker had to kill
it. Outside Docker the process ended by the signal after logging `Finished server process`; the code
Docker reports was not seen.

Compose publishes the port on `127.0.0.1` only. Clients reach it through your gateway (see What is
development quality today) with their tenant token.

What the compose file sets, and why:

| Setting | Reason |
| --- | --- |
| `env_file: deploy/tine.env` | Credentials come from a file that is not committed |
| `./deploy/config:/config:ro` | Policy and checks JSON are mounted read-only |
| `tine-ledger:/data` | The ledger is on a named volume and survives the container |
| `read_only: true`, `/tmp` as tmpfs | The app writes only to `/data` and `/tmp` |
| `cap_drop: ALL`, `no-new-privileges` | The app needs no Linux capability |
| `restart: unless-stopped` | Restart after a crash or a host reboot |
| `stop_grace_period: 30s`, `init: true` | The proxy takes up to 20 seconds to stop; signals reach it through an init process |
| Port on loopback | There is no TLS in the container |

If you run the image without compose (plain `docker run`, Kubernetes, ECS), carry the same settings
over by hand. A named Docker volume takes the ownership of `/data` from the image. A bind mount or a
Kubernetes volume does not: make the directory writable by uid 10001 (in Kubernetes, `fsGroup: 10001`).
`docker run --env-file` does not strip quotes from a value as compose does; the env example's format
rules hold for both. Keep the `/tmp` tmpfs with a read-only root: the proxy itself started with a
read-only temp folder outside Docker, but `python -m tine freeze-policy` imports matplotlib, which
needs its cache folder (`MPLCONFIGDIR=/tmp/matplotlib`) to be writable.

## Check the image with one command

```bash
python3 scripts/container_selftest.py
```

It needs Docker running and nothing else: no provider key, no `deploy/tine.env`, no Python packages.
**No model is called and nothing is paid for.** The proxy under test is given a stand-in provider
(`scripts/container_standin.py`) in a second container and a tenant token the script makes up.

It uses `docker-compose.yml` with `docker-compose.selftest.yml` on top, under the project name
`tine-selftest`. So the image and the hardening settings are the real ones, while the env file, the
config folder, the host port (18000, on `127.0.0.1`) and the ledger volume are its own. It does not
touch your own containers, `deploy/tine.env` or `deploy/config/`. The script finds `docker` on `PATH`,
or in `~/.docker/bin` or the Docker Desktop application when it is not, and puts that folder on `PATH`
for the build.

It prints one line per check, PASS or FAIL, and ends with status 0, or 1 if a check failed, or 2 if
Docker could not be used. What it checks:

- the image builds, and both containers reach `healthy`;
- `/health` answers, and a request with no token or a wrong one gets 401;
- a request and a streamed request go to the stand-in and back, are recorded in the ledger and have a
  cost, and `/v1/savings` shows them;
- `scripts/savings_report.py`, fed to the container's Python on standard input, makes the savings page
  from the ledger inside the container;
- the process runs as uid 10001, cannot write to `/app`, `/config` or `/var/tmp`, and can write to
  `/data` and `/tmp`;
- `/app` holds only `pyproject.toml`, `requirements-dev.lock`, `requirements.txt` and `tine/`;
- after the mounted policy file is edited and the container restarted, `/health` shows
  `"routing": "policy"` and a request is served through the policy, and the ledger kept its rows;
- `docker compose stop` ends the container in under 28 seconds while a stream is still open, without
  Docker killing it, and the ledger is whole when the container starts again.

At the end it removes its containers, network and ledger volume. The image stays, tagged
`tine-proxy:dev`, the one `docker compose build` makes. Expect a few minutes for the first build, which
downloads the base image and 35 Python packages. Of the checks, the longest is
the stop with a stream open, which takes 20 seconds by design.

Options: `--port N` for another host port, `--keep` to leave the test containers running for a look
around, `--skip-build` to test the image that already exists, `--wait SECONDS` for a slow machine.

**This script has not been run against a real Docker daemon.** It was run with the Mac's `python3`
(3.9) against a stand-in for the `docker` command that starts the real proxy and the real stand-in
provider as local processes (`tests/test_container_selftest.py`). So its requests, the policy edit and
restart, the report page and the stop are tested; its `docker` commands, the build, uid 10001 and the
read-only filesystem are not. `docker compose config` accepts the two compose files together. These
lines are from that run; with a real daemon the stop takes about 20 seconds, not 3:

```text
image builds                  PASS  docker compose build tine
containers start              PASS  docker compose up -d --wait
container healthy             PASS  status running, health healthy
health                        PASS  status 200, routing frontier-default, shadow off
token needed                  PASS  401 with no token, 401 with a wrong one
request                       PASS  status 200, served by selftest-frontier, route -, ledger recorded
streamed request              PASS  status 200, served by selftest-frontier, route -, ledger recorded
savings                       PASS  tenant 'selftest', 2 new request(s), answer cost unknown for 0, known answer cost $0.000270
savings report page           PASS  a page of 10,123 characters was made inside the container
runs as uid 10001             PASS  id -u printed '10001'
read-only except /data, /tmp  PASS  writable: /app=False /config=False /var/tmp=False /data=True /tmp=True
image holds only the package  PASS  /app holds pyproject.toml, requirements-dev.lock, requirements.txt, tine
policy edit, then restart     PASS  /health routing policy; status 200, served by selftest-cheap, route stage-0, ledger recorded
ledger kept over a restart    PASS  3 request(s) in the ledger after the restart, 2 before it
stop with a stream open       PASS  stream status 200; stopped in 3.2 s (Docker kills at 30), exit code 143
ledger whole after the stop   PASS  4 request(s) in the ledger after the stop and start
clean up                      PASS  containers, network and test ledger volume removed
```

When a check fails, the script prints the last 40 log lines of the proxy container. `image builds FAIL`
shows the build's own output above it. `containers start FAIL` with a message about the port means
something else listens on 18000: pass `--port`. A pass does not test your own configuration, a real
provider, a gateway or load. For those, go on to the next section.

## Test the container on your own machine

For a Mac with Docker Desktop running. Two scripts do the work around the `docker` commands:
`scripts/container_env.py` writes the env file and a tenant token, and `scripts/container_smoke.py`
checks the running container. Both use only the Python standard library, so the Mac's own `python3`
runs them. The first one needs the development environment's Python once you give it a policy
(step 2). Run every command from the repository root.

What has been run and what has not, as of 2026-11-17:

- Both scripts were run with the Mac's `python3` (3.9) against `python -m tine serve`, which is the
  image's command, started outside Docker from the env file the first script wrote. The provider was a
  stand-in on a local port. No real model was called.
- `docker compose config`, which needs no daemon, read that env file to the same values as the
  script's own check (Docker Compose v5.7.0).
- The savings page at the end was made from a ledger that run wrote.
- No `docker` command below was run: not `build`, `up`, `ps`, `logs`, `exec`, `cp` or `down`. The image
  has still not been built.

Run [Check the image with one command](#check-the-image-with-one-command) first. It finds a build or
start problem without a key. This section then tests your own key, model and prices.

### 1. Put the provider key in a file

One line, in a file outside the repository:

```text
ANTHROPIC_API_KEY=your-key
```

`TINE_ANTHROPIC_API_KEY=your-key` is read as well. Do not type the key into a command: it would stay in
the shell history.

### 2. Write the env file and the token

```bash
.tine-dev/venv/bin/python scripts/container_env.py \
  --key-file ~/keys/anthropic.env \
  --frontier YOUR-FRONTIER-MODEL \
  --prices '{"anthropic:YOUR-FRONTIER-MODEL":{"input_per_million":5.0,"output_per_million":25.0}}'
```

Replace the key file, the model name and the two prices with your own. The prices shown are made up.
`--prices` also takes the path of a JSON file.

It writes:

| File | What it holds |
| --- | --- |
| `deploy/tine.env` (mode 600) | The provider key, a new random tenant token, the frontier, the prices |
| `deploy/client-token.txt` (mode 600) | The tenant token alone, for the smoke test |
| `deploy/config/` | The folder compose mounts at `/config`, and a policy file when you gave one |

Git ignores all three. The script never prints the key, and prints only the last four characters of the
token. It leaves an existing `deploy/tine.env` alone unless you add `--force`; `--force` makes a new
token, so recreate the container afterwards (step 3).

Before anything is put in place, the script loads the files into the proxy's own configuration and runs
its start-up validation. A configuration the proxy would refuse is reported and nothing is changed.
That check needs the development environment's Python (`.tine-dev/venv/bin/python`, made by
`bash scripts/setup.sh`). Plain `python3` cannot make it:

- Without a policy, plain `python3` writes the same files and the output says `NOT checked`. Every
  value in them is one the script made or checked itself.
- With `--route-policy` or `--shadow-policy`, plain `python3` writes no file and exits with status 1.
  Only the proxy's own code can say whether it would run a policy. Use the development environment's
  Python.

Options for more than a frontier-only test:

- `--route-policy FILE` serves answers through a frozen policy. `--audit-rate 0.05` holds out that share
  of routable requests to the frontier. Routing changes what callers receive.
- `--shadow-policy FILE --shadow-sample-rate 0.5 --shadow-max-cost-usd 2` runs a frozen policy in shadow
  mode. Without a rate above 0, shadow mode stays off.
- `--tenant NAME` names the tenant. The default is the policy's own tenant, or `local-test`.
- `--answer-pattern REGEX` is for a policy with agreement stages that carries no pattern of its own.

`FILE` is a policy as `python -m tine freeze-policy --tenant NAME` writes it. Give every stage model a
price in `--prices`. A policy that names a judge or webhook check is refused, by either Python, and no
file is written: the script does not write a checks file. For such a policy, write the env file and
`TINE_CHECKS` by hand as in Run it.

### 3. Build and start

```bash
docker compose build
docker compose up -d --wait
docker compose ps
```

- The first build downloads the base image and the Python packages. Expect a few minutes.
- `docker compose up -d --wait` returns when the container is healthy. `docker compose ps` shows
  `health: starting` and then `healthy`. Outside Docker the proxy listened 0.3 seconds after it was
  started.
- A status of `Restarting` means the proxy refused its configuration. Read the logs (step 5).
- If another program already listens on port 8000 of the Mac, such as a `python -m tine serve` left
  running, `docker compose up` fails with an error about the port. Stop the other program, or use
  another host port: `TINE_HOST_PORT=8010 docker compose up -d --wait`, then pass
  `--base-url http://127.0.0.1:8010` to the smoke test.
- After you change `deploy/tine.env` or a policy file, run `docker compose up -d --force-recreate`. The
  proxy reads both only when it starts.

### 4. Run the smoke test

```bash
python3 scripts/container_smoke.py
```

This costs nothing. It waits up to 30 seconds for the proxy to answer, then checks that:

- `GET /health` answers, and prints which routing and shadow mode are on;
- a wrong token, and no token, are refused with 401;
- `GET /v1/savings` answers for the tenant whose token is in `deploy/client-token.txt`;
- `GET /v1/shadow` answers, when shadow mode is sampling.

It prints one line per check and exits with status 1 if any failed. It never prints the token.

To send real requests through the container:

```bash
python3 scripts/container_smoke.py --send 3 --model YOUR-FRONTIER-MODEL --yes-paid
```

**These are paid calls.** The container answers them with the provider key in `deploy/tine.env`, on that
provider account. The script refuses `--send` without `--yes-paid`. Each request asks for one word and
at most 16 output tokens. The last one is streamed when you send two or more. Sending stops at the first
request that fails. `--model` is the name put in the request; the proxy decides which model answers.

For each request it prints the status, the model that served it (`x-tine-model`), the route
(`x-tine-route`: `stage-0` for a policy's first stage, `frontier`, `holdout`, or `-` when no policy
ran), the request id and whether the ledger recorded it. Then it reads `/v1/savings` again and checks
that the requests are there with a cost, and reads `/v1/shadow` if shadow mode is on. These lines are
from a run outside Docker against the stand-in provider, with a route policy, a holdout share of 0.25
and `--send 8`. The ids are shortened, and requests 2 to 5 and 7, which read like request 1, are left
out:

```text
health                 PASS  routing policy, shadow off, storage sqlite-development
wrong token refused    PASS  401 for a wrong token, 401 for no token
savings                PASS  tenant 'rehearsal', 0 request(s), known cost $0.000000, answer cost unknown for 0
shadow report          SKIP  shadow mode is off, so there is no /v1/shadow to read
request 1              PASS  status 200, served by cheap-model, route stage-0, id cd7566b2, ledger recorded
request 6              PASS  status 200, served by frontier-model, route holdout, id 068c3b09, ledger recorded
request 8 (streamed)   PASS  status 200, served by frontier-model, route -, id c61c7106, ledger recorded
savings after          PASS  requests 0 -> 8, known cost $0.000000 -> $0.000432
costs recorded         PASS  every request sent has an answer cost
```

The model names and costs in it are the stand-in's, not a provider's.

What a failure usually means:

| Line | Likely cause |
| --- | --- |
| `health FAIL` | The container is not up, or still restarting. `docker compose ps`, then the logs |
| `savings FAIL` with 401 | `deploy/tine.env` was rewritten after the container started. Recreate it (step 3) |
| a request with status 401 or 404 | The provider refused the key, or does not know the model name |
| `ledger unavailable` | The proxy could not write `/data/ledger.sqlite3`. Check the volume's owner |
| `costs recorded FAIL` | The model that answered has no price in `TINE_MODEL_PRICES` |

The proxy on another port or host: `--base-url http://127.0.0.1:8010`. A token kept elsewhere:
`--token-file PATH`.

### 5. Read the logs

```bash
docker compose logs tine
docker compose logs -f --tail 50 tine
```

The second form follows the log; Ctrl-C stops following and leaves the container running. A healthy
start has one line per loaded policy, then `Uvicorn running on http://0.0.0.0:8000`, then one line per
request. A line starting `tine serve: the configuration was refused.` followed by a restart is a
refused configuration: the message names the setting (When it does not start, above).

Do not paste `docker compose config` output anywhere: it prints the provider key and the tenant token.

### 6. Make the savings report page

The image holds only the `tine` package, not `scripts/`. So the page is made on the Mac, from a copy of
the container's ledger. The copy is taken with SQLite's online backup, which is safe while the proxy
runs (see Back up the ledger):

```bash
mkdir -p artifacts
docker compose exec tine python -c "import os, sqlite3; os.umask(0o077); s = sqlite3.connect('/data/ledger.sqlite3'); d = sqlite3.connect('/data/ledger.backup.sqlite3'); s.backup(d); d.close(); s.close()"
docker compose cp tine:/data/ledger.backup.sqlite3 artifacts/container-ledger.sqlite3
docker compose exec tine python -c "import os; os.remove('/data/ledger.backup.sqlite3')"
.tine-dev/venv/bin/python scripts/savings_report.py \
  --ledger artifacts/container-ledger.sqlite3 --tenant local-test \
  --output artifacts/container-savings.html
open artifacts/container-savings.html
```

- Use the tenant name `container_env.py` printed, if it was not `local-test`.
- Add `--policy FILE` to describe the policy on the page and to check shadowed answers against its
  quality limit. `FILE` is the policy you gave `container_env.py`, or the file the container reads
  (`deploy/config/shadow-policies.json` or `route-policies.json`): the tenant's policy is taken from it.
  Add `--provider anthropic` or `--provider openai` only when the tenant has a policy for both.
- `savings_report.py` needs the development environment's Python. It opens the copy read-only.
- On a machine with Docker and no development environment, make the page inside the container. The
  image has no `scripts/` folder, so the script goes in on standard input:
  `docker compose exec -T tine python - --ledger /data/ledger.sqlite3 --tenant local-test --output /tmp/report.html < scripts/savings_report.py`,
  then `docker compose exec -T tine cat /tmp/report.html > artifacts/container-savings.html`. It reads
  the live ledger without changing it. The script was run this way outside Docker; the two
  `docker compose exec` commands are what the one-command check runs, and were not run by a daemon.
- Do not open the ledger for a report from a volume mounted read-only (`:ro`) in a second container.
  SQLite needs to write beside a ledger in write-ahead mode even to read it, and the report fails with
  `attempt to write a readonly database`.
- The copy holds every prompt and response the container saw. `artifacts/` is ignored by git. Delete the
  copy when you are done, and check that its mode is 600.
- The backup line and `savings_report.py` were run here on a ledger written outside Docker. The
  `docker compose exec` and `docker compose cp` around them were not. [REPORT.md](REPORT.md) explains
  the page.

### 7. Stop and remove

```bash
docker compose stop              # stop the container; it and the ledger stay
docker compose start             # start it again
docker compose down              # remove the container and its network; the ledger volume stays
docker compose down --volumes    # also delete the ledger volume
docker image rm tine-proxy:dev   # remove the image
rm deploy/tine.env deploy/client-token.txt
```

`docker compose down --volumes` deletes every recorded request and cost. Take the copy in step 6 first
if you want them. Remove `deploy/tine.env` and `deploy/client-token.txt` when the test is over: they
hold the provider key and a working token.

### What this test does not show

A pass shows that the image builds, starts under the compose file's settings (uid 10001, read-only
root), can write its ledger volume, answers on the loopback port and records requests. It does not test
a gateway or TLS in front of the proxy, load, a restart under traffic, a backup restore, or the quality
of any policy.

## What leaves your network and what does not

Leaves, to destinations you configure:

- **Prompts and responses go to the model provider.** Each request is forwarded to
  `TINE_ANTHROPIC_BASE_URL` or `TINE_OPENAI_BASE_URL` on your own provider account. By default those
  are the providers' public APIs, so the provider receives the prompt. Not all data stays in-house. If
  a base URL points at a model server on your own hardware, that traffic stays where that server is.
- **Shadow mode and routing send the same prompt to cheaper models** on the same provider account and
  base URL. More models see the prompt than without Tine, all at the provider you chose. Every stage
  model and judge model must be served at that one base URL. The public provider APIs do not serve
  the open models in the example policies; to use those, run your own OpenAI-compatible gateway, such
  as vLLM or LiteLLM, that serves the frontier and every candidate, and point the base URL at it. Do
  not use a third-party router: it would receive your prompts.
- **A judge check sends the request's user text and a cheaper model's response to a Claude model**
  through the Anthropic base URL and key you configured.
- **A webhook check sends the full request and the cheaper model's response to the URL in your checks
  file**, together with the tenant id, the provider, the model name, the proxy's request id and
  whether the proxy is shadowing or routing, and with the bearer token you named. Point it only at your own infrastructure. If you point it somewhere else, that is where the
  data goes. Four rules for `bearer_env`, enforced at startup: it may not name a `TINE_*` variable,
  `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; a webhook that is sent a token must use an `https://` url,
  loopback included; one token variable may be sent for one tenant only; and the variable must be set
  and not empty in the container's environment. The example (`CI_WEBHOOK_TOKEN`, an https url)
  follows all four.
- **Check files are operator-only.** Whoever can edit `checks.json` or the tenant checks file chooses
  which URLs receive tenants' requests and which environment variable is sent as a token. The proxy
  cannot stop a check from naming some other secret in its environment, such as a cloud credential.
  Keep only webhook tokens and the proxy's own settings in the container's environment.

Does not leave:

- **Nothing is sent to Tine.** The proxy code makes outbound calls only to the two provider base URLs
  and to webhook URLs in your checks file. There is no telemetry, licence check or update check in the
  `tine` package. No third-party router receives a prompt.
- **The ledger stays on the volume.** It is a SQLite file holding requests, responses, costs and
  outcomes.

Limits of these statements:

- They describe the `tine` package's own code, read on 2026-11-16. The third-party libraries in the
  image were not audited. Verify with an egress rule: allow the container to reach only your provider
  hosts and your webhook host, and deny everything else.
- Building the image downloads the base image from Docker Hub and packages from PyPI. No customer data
  is involved, but the build host needs that access, or an internal mirror.
- The proxy serves FastAPI's default `/docs`, `/redoc` and `/openapi.json` pages without a tenant
  token. They show the route list, not data. The `/docs` and `/redoc` pages make the viewer's browser
  load scripts from a public CDN. Block these paths at your gateway if you do not want them.
- What a provider does with a prompt is governed by your agreement with that provider, not by Tine.

## What is development quality today

Do not treat this as a production service.

- **SQLite ledger.** One file on one volume. No replication and no point-in-time recovery. Postgres is
  planned and not built. The ledger stores prompts and responses in plain text: protect the volume and
  its backups as you would the prompts themselves, and encrypt the disk.
- **Single process.** One container, one worker. Do not run two containers on the same ledger volume
  and do not scale the service past one replica. There is no load test, so capacity is unknown.
- **No TLS termination.** The container speaks plain HTTP. Put it behind your own gateway or load
  balancer, which terminates TLS, limits request size and rate, and restricts who can connect. Tenant
  tokens travel in a header, so never expose port 8000 beyond the host or a private network.
- **Tenant tokens are static.** They live in an environment variable. Changing or revoking one means
  editing the env file and restarting.
- **Credentials in an env file.** Simple, and readable by anyone who can read that file or inspect the
  container. Use your platform's secret store to inject the same variables if you have one.
- **The ledger fails open by default.** When a row cannot be written the answer is still served, marked
  `x-tine-ledger: unavailable`. Set `TINE_LEDGER_REQUIRED=1` (or `true`) to return 503 instead, so no
  answer goes out unrecorded. It is commented out in `deploy/tine.env.example` under "Ledger".
- **Request limits are basic.** A body over `TINE_MAX_BODY_BYTES` (default 10 MB) gets 413.
  `TINE_DENIED_FIELDS` and `TINE_ANTHROPIC_BETA_ALLOWLIST` are off until set, and refuse listed request
  fields and unlisted `anthropic-beta` values with 400. Nothing refused is stored in the ledger. There
  is no rate limit. Tested with stand-ins only; see Request limits in PROXY.md. They are commented out
  in `deploy/tine.env.example` under "Request limits".
- **No metrics or structured logs.** Only `/health` and the container's log output.
- **Never run on real traffic.** Shadow mode, routing through a policy and named checks have been
  tested with stand-in providers and recorded answers only. See PROXY.md for the limits of each.
- **Streaming requests are passed through** to the frontier (or to a model you allowed by name in
  `TINE_ALLOWED_MODELS`) and are never shadowed or routed by a policy. Their answer cost is recorded
  from the usage the stream carries; OpenAI streams carry it only when the caller sends
  `"stream_options": {"include_usage": true}`, which the proxy does not add. Without usage the cost
  stays unknown. Tested with stand-in streams only.

## Start in shadow mode with a spending cap

Shadow mode changes nothing a caller sees. The frontier answers as before. For a sample of requests the
proxy then runs a frozen policy's cheaper models in the background and records what the policy would
have done. Shadow calls are real calls on your provider account and cost real money.

1. **Run frontier-only first.** Leave `TINE_SHADOW_SAMPLE_RATE=0`. Send some traffic through the
   gateway and read `GET /v1/savings` with the tenant token. `unknown_answer_cost_requests` should be
   0 and `known_answer_cost_total_usd` above 0. If not, fix `TINE_MODEL_PRICES` before going further.
   `unknown_cost_requests` and `unknown_routing_cost_requests` equal `requests` at this stage, with
   the prices right: without a route policy a request's routing cost is always unknown (PROXY.md).
2. **Replace the example policy.** `deploy/shadow-policies.example.json` is fake. Its guarantee says so
   in its own text, and its model names do not exist. Put a policy frozen with
   `python -m tine freeze-policy` from your own graded tasks into `deploy/config/shadow-policies.json`,
   under the tenant id from `TINE_TENANT_KEYS` and the provider it applies to. Its `frontier` must equal
   that provider's frontier model. Freeze it with `--tenant` to name that tenant, and with
   `--answer-pattern` when it has agreement stages ([POLICY.md](POLICY.md), Certify on your own graded
   tasks). `TINE_ANSWER_PATTERN` is the fallback for policies without their own pattern; a route policy
   with agreement stages needs one or the other.
3. **One tenant token per task type.** The proxy does not tell task types apart. Give each kind of
   traffic (ticket labels, reply drafts, code) its own token in `TINE_TENANT_KEYS` and its own policy,
   and send traffic of other kinds on a token with no policy. Check the task type of each loaded
   policy in the startup log.
4. **Check that every model is served.** Every stage model and judge model must answer at the
   provider's base URL (see What leaves your network).
5. **Configure every check the policy names.** An agreement stage needs nothing. Any other check name
   must appear in `deploy/config/checks.json`, or the proxy refuses to start. A configured check must
   do what the measured check did, or the policy's guarantee does not describe what runs. With more
   than one tenant, give each tenant its own webhook check and its own token variable in a second
   file named by `TINE_TENANT_CHECKS` (tenant id, then check name, then steps). The proxy refuses to
   start when a check in `checks.json` with a webhook step is named by the policies of more than one
   tenant, or when one `bearer_env` is used for two tenants.
6. **Price every model.** Add the frontier, each stage model and each judge model to
   `TINE_MODEL_PRICES`. The cap counts known costs only. The proxy refuses to start sampling when a
   stage model or a judge model of a shadow policy has no input and output price.
7. **Set the cap, then the rate.** Set `TINE_SHADOW_MAX_COST_USD` to an amount you are willing to
   lose, then set `TINE_SHADOW_SAMPLE_RATE` to a small share such as `0.05`. The proxy refuses to start
   with a rate above 0, a policy and no cap.
8. **Restart and read the report.** `docker compose up -d --force-recreate` (a changed policy file
   alone does not make plain `up -d` restart the container), then `GET /v1/shadow` with the tenant
   token. `GET /health` shows `"shadow": "sampling"` when it is on. Read each token's acceptance share
   against its policy's estimate, `shadow_coverage`, `streamed_not_shadowed` and `drift_warning`
   before routing anything.

What the cap does and does not do:

- It is per tenant and counts that tenant's known shadow spend. The count is read from the ledger
  once and then kept in memory in the proxy process. It does not reset. Raise it on purpose when you
  want more.
- A run in flight reserves room under the cap: it is assumed to cost what the tenant's most expensive
  run so far did. The run that crosses the cap still completes. Spend can pass the cap by about one
  run, and by more only when runs in flight cost more than any run before them.
- A call can still have an unknown cost, for example one that failed or timed out. Those are counted
  by number, not in dollars. Shadowing stops for a tenant once it has more than
  `TINE_SHADOW_MAX_UNKNOWN_COST_CALLS` of them (default 20) and they are more than
  `TINE_SHADOW_MAX_UNKNOWN_COST_SHARE` (default 0.2) of its shadow calls and checks.
- `TINE_SHADOW_MAX_CONCURRENT` (default 8) limits the shadow runs in flight at once, for the whole
  proxy. `TINE_SHADOW_MAX_CONCURRENT_PER_TENANT` (optional, unset by default) limits them for each
  tenant as well, so one busy tenant cannot take every slot. A sampled request that arrives while a
  limit is full is not shadowed; `GET /v1/shadow` counts it as `skipped_busy`.
- Rate limits belong to the provider organisation, so shadow calls and your real calls share them.
  When a provider answers 429 or 529, on a shadow call or on your real traffic, shadow mode pauses on
  that provider and `GET /v1/shadow` counts the skipped requests as `skipped_rate_limited`. Your real
  requests are passed through as before. `TINE_SHADOW_MAX_CALLS_PER_MINUTE` (optional) also caps the
  shadow calls each tenant may start in a minute. See Shadow mode in PROXY.md.
- It does not limit the frontier calls your traffic makes anyway. It limits only the extra calls.
- Set a budget alert on the provider account as well. The cap is application code that has never run
  against a real bill.

The shadow report's limits are in PROXY.md. The main one: equal answers are evidence only for
exact-answer work, and are not proof that an answer is right.

### Resume shadow mode after it stopped

When `sampling` says "stopped: too many shadow calls of unknown cost", the stop is rebuilt from the
tenant's stored shadow rows at every start. A restart alone does not resume it.

1. Fix the cause the stop reason names. The stage model must be served at the provider base URL, the
   CI webhook must be healthy, and every stage and judge model needs a price.
2. Then either raise `TINE_SHADOW_MAX_UNKNOWN_COST_CALLS` and/or `TINE_SHADOW_MAX_UNKNOWN_COST_SHARE`
   and restart, or back up the ledger, delete that tenant's rows from its `shadow_runs` table and
   restart. Deleted rows leave the report, and their spend leaves the spending cap's count.

A stop at the spending cap ends only when you raise `TINE_SHADOW_MAX_COST_USD` and restart.

Restart here means `docker compose up -d --force-recreate`, so that the change is read whether it was
made in the env file or not.

## Turn routing off

Policy files are read only at startup. Editing or emptying `route-policies.json` while the proxy runs
changes nothing.

1. Unset `TINE_ROUTE_POLICIES` (comment it out in `deploy/tine.env`), or set every policy's `stages` to
   `[]`.
2. Restart: `docker compose up -d --force-recreate`. Plain `docker compose up -d` does not restart
   the container when only the policy file changed, and routing would stay on.
3. Confirm that `GET /health` shows `"routing": "frontier-default"`. Do not skip this step.

New ledger rows then have no `route_stage`. When you change the file instead of removing it, the
`policy_sha256` on new rows changes with it, so the ledger shows which file served each request.

Routing (`TINE_ROUTE_POLICIES`) changes what callers receive. Leave it unset until the shadow report
and PROXY.md have been read. Under live routing the saving is measured only with a frontier holdout:
`TINE_AUDIT_RATE` (default 0, off) sends that share of the requests a policy could serve to the
frontier, and `GET /v1/savings` then compares the two groups' mean cost per request, with a 95%
interval, in `holdout_estimate` (PROXY.md, Measuring the saving under live routing). It is an estimate
from a random sample, it does not measure quality, and held-out requests cost the frontier's price.
While some routed requests have a routing cost that is not known (a cheaper model failing, a webhook,
the route deadline), the figure is only an upper limit on the saving.
Built and tested with stand-ins only. A tenant's provider can have a shadow policy or a route policy,
not both.
`TINE_ROUTE_DEADLINE_SECONDS` is how long a route policy's cheaper stages may take together for one
request before it goes to the frontier. Default: 30 seconds, or `TINE_TIMEOUT_SECONDS` if that is
shorter. Raise it to cover the test run when a stage's check runs tests.

## Back up the ledger

The ledger is `/data/ledger.sqlite3` on the `tine-ledger` volume. It uses SQLite's write-ahead log, so
recent writes can sit in `ledger.sqlite3-wal`. Copying only the main file from a running container can
give an incomplete or broken backup. Use SQLite's online backup, which is safe while the proxy runs:

```bash
docker compose exec tine python -c "import os, sqlite3; os.umask(0o077); s = sqlite3.connect('/data/ledger.sqlite3'); d = sqlite3.connect('/data/ledger.backup.sqlite3'); s.backup(d); d.close(); s.close()"
docker compose cp tine:/data/ledger.backup.sqlite3 ./ledger-$(date +%Y%m%d).sqlite3
docker compose exec tine python -c "import os; os.remove('/data/ledger.backup.sqlite3')"
```

The Python one-liner was run here against a ledger file outside Docker. The `docker compose` commands
around it were not run. The `umask` makes the backup readable by its owner only, like the ledger
itself. Check the mode of the copy on the host as well.

Or stop the container and copy the whole volume:

```bash
docker compose stop tine
docker run --rm -v tine_tine-ledger:/data:ro -v "$PWD":/backup python:3.12-slim-bookworm \
  tar czf /backup/tine-ledger.tar.gz -C /data .
docker compose start tine
```

The volume name carries the compose project name as a prefix (`tine_` when the directory is named
`tine`). Check it with `docker volume ls`.

Notes:

- A backup contains prompts and responses. Store it with the same protection as the ledger.
- The first method needs free space on the volume for a second copy of the ledger.
- To restore: stop the container, put the backup file at `/data/ledger.sqlite3` on the volume, remove
  any `ledger.sqlite3-wal` and `ledger.sqlite3-shm` files left beside it, make sure uid 10001 owns the
  file, and start the container. Restore has not been rehearsed.
- Nothing prunes the ledger. It grows with traffic. Watch the volume's free space: when the ledger
  cannot be written, provider answers are still returned, marked `x-tine-ledger: unavailable`, and
  those requests are missing from the savings record. With `TINE_LEDGER_REQUIRED=1` they get a 503
  instead.

## Upgrade

```bash
git pull
docker compose build
docker compose up -d
```

Back up the ledger first. The ledger upgrades its own tables on start, and there is no downgrade path.
Run one container at a time: PROXY.md asks that all writers restart together on an upgrade, and with
one container that is automatic.

Upgrading a ledger written by an earlier version adds the `baseline_source`,
`policy_sha256`, `route_stage`, `route_skip_reason`, `checks_json` and `holdout` columns to `requests`,
`skip_reason` to `shadow_runs`, and the `outcome_events` and `shadow_counts` tables. Existing baselines get a source
by this rule:

- an estimated baseline becomes `estimated`;
- a baseline on a row whose outcome was never updated becomes `proxy_measured`;
- every other baseline becomes `caller`, because an old ledger cannot show whether the caller replaced
  it.

So verified savings from before the upgrade usually move to the caller-supplied figures. Older rows
have no `policy_sha256`, `route_stage`, `checks_json` or `holdout`, older shadow runs have no check steps,
earlier outcome updates are not in `outcome_events`, and shadow coverage counts start at the upgrade.

## Checked without Docker

`tests/test_deploy_files.py` checks that:

- the env example names exactly the `TINE_*` variables `tine/proxy.py` reads;
- the example env, policy and checks files load into `ProxyConfig` and pass `validate()`, with shadow
  sampling off as shipped and with it on, and that sampling without a cap is refused;
- every commented-out line of the env example validates when it is switched on, with the example
  policy as a route policy and the tenant checks example in place;
- the example policy used as a route policy is refused without an answer reader and accepted with one,
  and a webhook check whose token variable is unset is refused;
- `/health` answers 200 for an app built from the example configuration, and the Dockerfile's health
  check command, run as written against that app on a local port, exits 0, also with `HTTP_PROXY` set
  to a dead address, and exits 1 when nothing listens;
- changing the frontier model with the example policy left in place is refused, and validates again
  with `TINE_CHECKS` and `TINE_SHADOW_POLICIES` switched off;
- no committed deploy file holds a value shaped like a credential. Your own `deploy/tine.env` and
  `deploy/config/` are skipped, not read;
- the Dockerfile has a pinned base, a non-root user, no `ARG`, no secret-named `ENV`, a standard
  library health check that ignores proxy variables, copies only `pyproject.toml`,
  `requirements-dev.lock` and `tine/`, and installs with the lock as constraints; its command that
  writes `requirements.txt`, run through `sh`, gives the dependency lines of `pyproject.toml`, and
  every one of them has a pin in the lock;
- `.dockerignore` excludes `artifacts`, `.tine-dev` and everything else outside the package;
- the compose file has one service with the hardening settings in the table above, publishes on
  `127.0.0.1` only, and waits longer for a stop than `tine serve` takes;
- `tine serve`, called with the Dockerfile's arguments, starts the server with that stop limit, and
  ends with one line and no traceback when the configuration is refused;
- when the `docker` client is installed, `docker compose config` reads the env example to the same
  values as these tests do;
- git ignores `deploy/client-token.txt`, and this page gives the steps of Test the container on your own
  machine in order, with only options the scripts have;
- this page, the Dockerfile and [PROXY.md](PROXY.md) say in the same words which Dockerfile steps were
  run by hand and which were not, and the steps named as not run are the ones the Dockerfile has.

`tests/test_container_env.py` runs `scripts/container_env.py` on a made-up key file in a temporary
folder: the key and the token never reach its output, both files are mode 600, an existing env file is
kept without `--force`, and what it writes loads into `ProxyConfig` and validates. It also runs the
script with a Python that cannot import the package: a policy that names a judge or webhook check is
refused, any other policy is refused as unchecked, and without a policy the files are written.
`tests/test_container_smoke.py` runs `scripts/container_smoke.py` against the real proxy app served on
a local port, with a stand-in provider on a second local port.
`tests/test_container_selftest.py` runs `scripts/container_selftest.py` against a stand-in for the
`docker` command, as described in Check the image with one command, and checks that the two compose
files resolve together without `deploy/tine.env` when the `docker` client is installed.

Checked by hand on 2026-11-17, on a Mac with Docker Desktop installed and its daemon out of reach
(Docker 29.9.3 client, Docker Compose v5.7.0, the development environment's Python 3.12):

- **Compose files.** `docker compose config` accepts `docker-compose.yml` with a copy of the env
  example, reads its JSON values unchanged, publishes `127.0.0.1:8000`, and `127.0.0.1:8010` with
  `TINE_HOST_PORT=8010`. Without `deploy/tine.env` it stops with `env file ... not found`. It accepts
  `docker-compose.yml` with `docker-compose.selftest.yml` when no `deploy/tine.env` exists: one env
  file (the generated one), port 18000 on `127.0.0.1`, and the hardening settings unchanged.
- **Dependencies.** The Dockerfile's command that turns `pyproject.toml` into `requirements.txt` wrote
  the eight dependency lines. `pip install --dry-run` of those lines with `requirements-dev.lock` as
  constraints, from wheels alone, for Python 3.12 on 64-bit Arm and x86 Linux (glibc up to 2.36, as in
  the base image) resolved 35 packages, each at the lock's version. Nothing was installed for Linux.
  A second dry run later that day gave the same 35 packages for both. The dry run has to name every
  glibc tag from `manylinux_2_17` to `manylinux_2_36` with `--platform`: given `manylinux_2_36` alone
  it found no numpy wheel and stopped with `ResolutionImpossible`. `pip` inside the image works out
  its own tags, so this is about repeating the dry run, not about the build.
  Without the constraints, the same lines resolved four packages newer than the lock that day
  (fastapi 0.163.1 against 0.162.0, numpy 2.5.5 against 2.5.4, click and annotated-doc).
- **Start.** The image's command was run from a read-only copy of what `/app` will hold (24 files,
  about 430 KB), with the image's `ENV`, `HOME=/nonexistent`, a read-only temp folder and a stand-in
  provider on a local port. It was bound to `127.0.0.1`, not `0.0.0.0`. With the first-start settings
  of Run it, it was ready in 0.3 seconds. The only files written were the ledger and its `-wal` and
  `-shm` files, mode 600.
- **Import cost.** Importing the proxy took 0.13 to 0.26 seconds and about 50 MB, and loaded none of
  numpy, pandas, scipy, scikit-learn or matplotlib. Before `tine/policy.py` was changed to import them
  only when a policy is frozen, it took 1.8 seconds and 212 MB with matplotlib's cache already built.
- **Health check.** The Dockerfile's command, with the port changed, exited 0 in 0.04 seconds, exited
  0 with `HTTP_PROXY`, `HTTPS_PROXY` and `ALL_PROXY` set to a dead address, and exited 1 with nothing
  listening.
- **Requests.** No token: 401. A request: 200, `x-tine-ledger: recorded`. A streamed request: 200,
  recorded. `/v1/savings` then showed `requests` 2, `unknown_answer_cost_requests` 0,
  `known_answer_cost_total_usd` 0.00054, and `unknown_cost_requests` 2 (no route policy).
- **Stop.** SIGTERM to an idle proxy: gone in 0.1 to 0.3 seconds. With a stream that never ends: gone in
  20.1 seconds, after `Cancel 1 running task(s), timeout graceful shutdown exceeded` and
  `Finished server process`. The ledger was closed to a single file both times.
- **Unreachable provider.** The caller got 502 `Upstream request failed`, and the log had the warning
  line shown in When it does not start.
- **Examples.** The example exactly as shipped started and logged its policy with the answer reader
  `Answer: ([A-D])`. With every commented line of the env example switched on (leaving out the three
  OpenAI lines and the shadow file line), the example policy copied to `route-policies.json` and the
  tenant checks example in place, it started with `"routing": "policy"`.
- **Refusals.** Each of these ended with exit code 1 and the reason on the last line: the webhook
  token line removed; `TINE_TENANT_KEYS` in quotes; `TINE_MODEL_PRICES` missing a brace;
  `TINE_TIMEOUT_SECONDS=30s`; `TINE_VERIFIED_CACHE=yes`; the frontier changed with the example policy
  left; the config folder empty; `TINE_ROUTE_POLICIES` uncommented with no file. With the three
  switches set to `1` it started.
- **One-command check.** `scripts/container_selftest.py`, run with the Mac's `python3` (3.9) against
  the stand-in `docker` command: all 17 lines passed, and the stop with a stream open took 3.2 seconds
  at `--shutdown-seconds 3`.

- **Env script with plain `python3`.** `scripts/container_env.py`, run with the Mac's `python3` (3.9)
  and a made-up key: without a policy it wrote the env file and the token file, both mode 600, and
  said `NOT checked`. With a route or a shadow policy it wrote no file and exited with status 1. With a
  policy that names `judge:x`, `tests` or `tests+judge:x` it exited with status 1 and named the check,
  as it did with the development environment's Python.

Not checked by anything: that the base image pulls, that the image builds, that the packages install
and import on the base image, that the two `COPY` steps and the user-creation step (`groupadd`,
`useradd`, `chown`, `chmod`) run, that the container starts as uid 10001 under a read-only root
filesystem, that the health check passes in a container and when `healthy` first shows, that volume
ownership works as described, what `docker compose up -d` does after a file-only edit, the exit code
Docker reports after a stop, whether macOS asks Docker Desktop for access to this folder on the first
bind mount, and every `docker` command on this page that needs the daemon.
