"""Check development prerequisites on synthetic data, without fitting models.

This is an environment check, not Tine application code or a quality guarantee.
"""

import asyncio
import os
from io import BytesIO, StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI
from scipy.stats import beta
from sklearn.metrics import pairwise_distances


def main():
    output_dir = Path(os.environ.get("TINE_DEV_DIR", Path(__file__).resolve().parents[1] / ".tine-dev"))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Fifty invented rows; no customer prompts, benchmarks, or API calls.
    frame = pd.DataFrame(
        {
            "task_id": np.arange(50),
            "frontier_cost_usd": np.full(50, 0.02),
            "candidate_cost_usd": np.full(50, 0.01),
        }
    )
    frame = pd.read_csv(StringIO(frame.to_csv(index=False)))
    assert len(frame) == 50
    totals = frame[["frontier_cost_usd", "candidate_cost_usd"]].sum()
    assert np.allclose(totals.to_numpy(), [1.0, 0.5])
    distances = pairwise_distances(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert np.isclose(distances[0, 1], 5.0)
    # One-sided binomial bound only; this does not calibrate a routing policy.
    upper_bound = beta.ppf(0.95, 1, 50)
    assert 0.05 < upper_bound < 0.07

    fig, ax = plt.subplots()
    ax.bar(["Synthetic frontier", "Synthetic candidate"], totals.to_numpy())
    ax.set_ylabel("Synthetic total cost (USD)")
    ax.set_title("Environment check only")
    png = BytesIO()
    fig.savefig(png, format="png")
    plt.close(fig)
    assert png.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    (output_dir / "synthetic-cost-smoke.png").write_bytes(png.getvalue())
    print("CPU data/evaluation tools: 50 synthetic rows, distances, bound, PNG OK")

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok", "kind": "environment-smoke-only"}

    async def check_http():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://smoke"
        ) as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"
            assert (await client.get("/missing")).status_code == 404

    asyncio.run(check_http())
    print("FastAPI/HTTPX request handling: health and missing route OK")


if __name__ == "__main__":
    main()
