# EgressTwin demo-only image (port 7860, the Hugging Face Spaces convention).
#
# Serves the web app in demo-only mode: it replays the recorded real
# muse-spark-1.3 run in data/demo and refuses every POST, so it needs no
# API key and makes no calls to the Meta Model API. The public demo is
# a static export of the same mode (scripts/deploy_space.py), because
# Docker Spaces on free hardware need a PRO plan.
#
#   docker build -t egresstwin-demo .
#   docker run -p 7860:7860 egresstwin-demo
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EGRESSTWIN_DEMO_ONLY=1 \
    META_MOCK=1 \
    MODEL_API_KEY= \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user pyproject.toml README.md LICENSE ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir .

COPY --chown=user data/demo ./data/demo
COPY --chown=user data/samples/spike_plan_hand.png ./data/samples/spike_plan_hand.png
COPY --chown=user docs/verification ./docs/verification
RUN mkdir -p runs && chown -R user:user /home/user/app

USER user
EXPOSE 7860
# --app-dir src makes the copied sources win over the installed wheel,
# so the app finds data/ and docs/ next to them.
CMD ["python", "-m", "uvicorn", "egresstwin.app.api:app", "--app-dir", "src", \
     "--host", "0.0.0.0", "--port", "7860"]
