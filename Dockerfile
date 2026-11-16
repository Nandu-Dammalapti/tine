# Tine development proxy.
# NOT YET BUILT OR RUN BY ANYONE: this file was written on a machine without Docker.
# See docs/DEPLOY.md before relying on it.
#
# The base image is pinned by tag. For a real deployment pin it by digest as well
# (python:3.12-slim-bookworm@sha256:...) after you have pulled and reviewed it.
FROM python:3.12-slim-bookworm

# No bytecode files (the root filesystem is meant to be read-only), unbuffered logs,
# and a writable place for matplotlib's cache, which the policy code imports.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    MPLCONFIGDIR=/tmp/matplotlib \
    TINE_LEDGER_PATH=/data/ledger.sqlite3

WORKDIR /app

# Dependencies come from pyproject.toml, the single place they are declared.
# They are version ranges, not a lock file, so two builds can differ.
COPY pyproject.toml /app/pyproject.toml
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('/app/pyproject.toml', 'rb'))['project']['dependencies']))" > /app/requirements.txt \
    && pip install --no-cache-dir -r /app/requirements.txt

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
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"]

# Credentials and configuration arrive as environment variables at run time.
# Nothing secret is set in this file.
CMD ["python", "-m", "tine", "serve", "--host", "0.0.0.0", "--port", "8000"]
