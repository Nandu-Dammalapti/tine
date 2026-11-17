# Tine development proxy.
# NOT YET BUILT BY A DOCKER DAEMON, and not every step below has been run.
# Run by hand outside Docker on 2026-11-17, on a Mac: the command that writes
# requirements.txt, a pip dry-run resolve of its lines for Linux (nothing was
# installed), the image's command (on 127.0.0.1, not 0.0.0.0), the health check
# command, requests and a stop.
# Not run: the base image, the pip install itself, the two COPY steps, the
# user-creation step (groupadd, useradd, chown, chmod), USER and VOLUME.
# docs/DEPLOY.md, "Checked without Docker", says what that covers and what it does not.
#
# The base image is pinned by tag. For a real deployment pin it by digest as well
# (python:3.12-slim-bookworm@sha256:...) after you have pulled and reviewed it.
FROM python:3.12-slim-bookworm

# No bytecode files (the root filesystem is meant to be read-only), unbuffered logs,
# and a writable place for matplotlib's cache. The proxy does not import matplotlib;
# "python -m tine freeze-policy" run in this image does.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    MPLCONFIGDIR=/tmp/matplotlib \
    TINE_LEDGER_PATH=/data/ledger.sqlite3

WORKDIR /app

# Dependencies come from pyproject.toml, the single place they are declared. They are
# version ranges, so requirements-dev.lock is given to pip as a constraints file: the
# image then holds the versions the tests ran on. The lock also names the development
# tools (pytest, ruff); a constraints file installs nothing by itself, so they stay out.
COPY pyproject.toml requirements-dev.lock /app/
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('/app/pyproject.toml', 'rb'))['project']['dependencies']))" > /app/requirements.txt \
    && pip install --no-cache-dir -r /app/requirements.txt -c /app/requirements-dev.lock

# Only the package. No artifacts, tests, results, notebooks or scripts.
COPY tine /app/tine

# A fixed non-root user that owns only the ledger directory.
# COPY keeps the permission bits of the build host. The chmod on /app makes the code
# readable by that user even when the checkout was made with a strict umask.
RUN groupadd --gid 10001 tine \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin tine \
    && mkdir -p /data /config \
    && chown 10001:10001 /data \
    && chmod 700 /data \
    && chmod -R a+rX /app

USER 10001:10001

# The SQLite ledger lives here. It holds prompts and responses: mount a volume.
VOLUME ["/data"]

EXPOSE 8000

# Standard library only, so the image needs no curl. The empty ProxyHandler makes the
# check ignore HTTP_PROXY: without it, a container given an egress proxy would send
# this loopback request to the proxy and report itself unhealthy.
# No --start-interval flag: Docker Engine 25 and later already probe every 5 seconds
# during the start period, and an older builder would refuse the flag. On an older
# engine the first probe, and so "healthy", comes 30 seconds after the start.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"]

# Credentials and configuration arrive as environment variables at run time.
# Nothing secret is set in this file.
# On SIGTERM the proxy lets requests in flight finish for up to 20 seconds
# (--shutdown-seconds), then cuts what is left, open streams included.
CMD ["python", "-m", "tine", "serve", "--host", "0.0.0.0", "--port", "8000"]
