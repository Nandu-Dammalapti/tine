# Deploying the proxy in a container

**The image has not been built or run by anyone yet.** The `Dockerfile` and `docker-compose.yml` were
written on a machine without Docker and checked by reading them and by the tests in
`tests/test_deploy_files.py`. Those tests load the example configuration into the real proxy
configuration and check the files as text. They do not build an image, start a container or call a
provider. Expect to fix something on the first real build, and treat every command below as untested.

This page covers packaging only. What the proxy does, its routes and its limits are in
[PROXY.md](PROXY.md). Frozen policies are in [POLICY.md](POLICY.md).

## What is in the image

- `python:3.12-slim-bookworm`, pinned by tag. Pin it by digest too before a real deployment.
- The dependencies listed in `pyproject.toml`. They are version ranges, not a lock file, so two builds
  made on different days can contain different versions.
- The `tine` package. No tests, results, notebooks, scripts, benchmark data or local artifacts: the
  `.dockerignore` admits only `pyproject.toml` and `tine/`.
- No credential. The image has no `ARG`, and its `ENV` lines set only paths and Python switches.

The container runs `python -m tine serve --host 0.0.0.0 --port 8000` as user 10001, not root. Its
health check calls `GET /health` with the Python standard library. The check ignores `HTTP_PROXY`, so
it still reaches the loopback address when you give the container an egress proxy.

## Run it

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

Three things to do in that file before the first start:

- **Replace the tenant token.** The proxy cannot tell a placeholder from a real token. If
  `TINE_TENANT_KEYS` is left as it is, the proxy starts and the placeholder printed in this repository
  works as a token.
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
docker compose up -d
docker compose ps          # the health column should reach "healthy"
docker compose logs tine
```

The proxy refuses to start when the configuration is invalid, for example with no tenant token, a
policy for a tenant that has no token, or a named check that is not configured. A policy or check file
(`TINE_ROUTE_POLICIES`, `TINE_SHADOW_POLICIES`, `TINE_CHECKS`, `TINE_TENANT_CHECKS`) that is missing,
unreadable, not UTF-8 or not valid JSON stops startup with
`ValueError: <VARIABLE>=<path> could not be loaded as JSON: <ErrorType>: <detail>`. The error is in the
logs and the container restarts in a loop until the configuration is fixed. The startup log also has
one line per loaded policy, with its tenant, provider, task type, answer reader and whether the file
was edited after freezing.

Compose publishes the port on `127.0.0.1:8000` only. Clients reach it through your gateway (next
section) with their tenant token:

```bash
curl -s http://127.0.0.1:8000/health
```

What the compose file sets, and why:

| Setting | Reason |
| --- | --- |
| `env_file: deploy/tine.env` | Credentials come from a file that is not committed |
| `./deploy/config:/config:ro` | Policy and checks JSON are mounted read-only |
| `tine-ledger:/data` | The ledger is on a named volume and survives the container |
| `read_only: true`, `/tmp` as tmpfs | The app writes only to `/data` and `/tmp` |
| `cap_drop: ALL`, `no-new-privileges` | The app needs no Linux capability |
| `restart: unless-stopped` | Restart after a crash or a host reboot |
| Port on loopback | There is no TLS in the container |

If you run the image without compose (plain `docker run`, Kubernetes, ECS), carry the same settings
over by hand. A named Docker volume takes the ownership of `/data` from the image. A bind mount or a
Kubernetes volume does not: make the directory writable by uid 10001 (in Kubernetes, `fsGroup: 10001`).

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
  file**, together with the tenant id, the provider and the model name, and with the bearer token you
  named. Point it only at your own infrastructure. If you point it somewhere else, that is where the
  data goes. Three rules for `bearer_env`, enforced at startup: it may not name a `TINE_*` variable,
  `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; a webhook that is sent a token must use an `https://` url,
  loopback included; and one token variable may be sent for one tenant only. The example
  (`CI_WEBHOOK_TOKEN`, an https url) follows all three.
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
   gateway and check that `GET /v1/savings` shows requests with known costs. If costs are unknown, fix
   `TINE_MODEL_PRICES` before going further.
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
8. **Restart and read the report.** `docker compose up -d`, then `GET /v1/shadow` with the tenant
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

## Turn routing off

Policy files are read only at startup. Editing or emptying `route-policies.json` while the proxy runs
changes nothing.

1. Unset `TINE_ROUTE_POLICIES` (comment it out in `deploy/tine.env`), or set every policy's `stages` to
   `[]`.
2. Restart: `docker compose up -d`.
3. Confirm that `GET /health` shows `"routing": "frontier-default"`.

New ledger rows then have no `route_stage`. When you change the file instead of removing it, the
`policy_sha256` on new rows changes with it, so the ledger shows which file served each request.

Routing (`TINE_ROUTE_POLICIES`) changes what callers receive. Leave it unset until the shadow report
and PROXY.md have been read. Under live routing the saving cannot be measured: that needs a random
share of requests always sent to the frontier, which is not built (PROXY.md, Measuring the saving
under live routing). A tenant's provider can have a shadow policy or a route policy, not both.
`TINE_ROUTE_DEADLINE_SECONDS` is how long a route policy's cheaper stages may take together for one
request before it goes to the frontier. It defaults to `TINE_TIMEOUT_SECONDS`. Raise it to cover the
test run when a stage's check runs tests.

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
`policy_sha256`, `route_stage` and `route_skip_reason` columns to `requests`, `skip_reason` to
`shadow_runs`, and the `outcome_events` and `shadow_counts` tables. Existing baselines get a source
by this rule:

- an estimated baseline becomes `estimated`;
- a baseline on a row whose outcome was never updated becomes `proxy_measured`;
- every other baseline becomes `caller`, because an old ledger cannot show whether the caller replaced
  it.

So verified savings from before the upgrade usually move to the caller-supplied figures. Older rows
have no `policy_sha256` or `route_stage`, earlier outcome updates are not in `outcome_events`, and
shadow coverage counts start at the upgrade.

## Checked without Docker

`tests/test_deploy_files.py` checks that:

- the env example names exactly the `TINE_*` variables `tine/proxy.py` reads;
- the example env, policy and checks files load into `ProxyConfig` and pass `validate()`, with shadow
  sampling off as shipped and with it on, and that sampling without a cap is refused;
- `/health` answers 200 for an app built from the example configuration;
- changing the frontier model with the example policy left in place is refused, and validates again
  with `TINE_CHECKS` and `TINE_SHADOW_POLICIES` switched off;
- no committed deploy file holds a value shaped like a credential. Your own `deploy/tine.env` and
  `deploy/config/` are skipped, not read;
- the Dockerfile has a pinned base, a non-root user, no `ARG`, no secret-named `ENV`, a standard
  library health check that ignores proxy variables, and copies only `pyproject.toml` and `tine/`;
- `.dockerignore` excludes `artifacts`, `.tine-dev` and everything else outside the package;
- the compose file has one service with the hardening settings in the table above.

Not checked by anything: that the image builds, that dependencies resolve on the base image, that the
container starts under a read-only root filesystem, that the health check passes in a container, that
volume ownership works as described, and every `docker` command on this page.
