ARG CSPD_BASE=lmsysorg/sglang:latest
FROM ${CSPD_BASE}

USER root
WORKDIR /opt/CSPD
ENV PATH="/opt/bootstrap/bin:${PATH}"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_MANAGED_PYTHON=1

RUN command -v git >/dev/null || \
    (apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*)
RUN command -v uv >/dev/null || \
    (python3 -m venv /opt/bootstrap && /opt/bootstrap/bin/python -m pip install uv)

COPY . .
RUN bash scripts/setup_blackwell_runtime.sh /opt/gen && \
    uv venv --python 3.11 /opt/score && \
    uv pip install --python /opt/score/bin/python -r experiments/m2rl/requirements.txt '.[data,test]'

RUN git init -q /opt/gorilla && \
    git -C /opt/gorilla fetch --depth 1 https://github.com/ShishirPatil/gorilla.git cd9429ccf3d4d04156affe883c495b3b047e6b64 && \
    git -C /opt/gorilla checkout -q FETCH_HEAD && \
    uv venv --python 3.11 /opt/bfcl && \
    uv pip install --python /opt/bfcl/bin/python \
      -e /opt/gorilla/berkeley-function-call-leaderboard soundfile 'transformers==4.57.1'

# Prefetch only the public LiveCodeBench data used by the offline grader.
RUN HOME=/opt/lcb-cache HF_HOME=/opt/lcb-cache/hf /opt/score/bin/python -c \
    'import json, sys; sys.path.insert(0, "experiments/m2rl/eval"); import score as s; s.git_checkout(s.LCB_REPO, s.LCB_COMMIT); refs = [json.loads(line) for line in open("experiments/m2rl/eval/inputs/lcb_v5/references.jsonl")]; [s._lcb_row(ref) for ref in refs]'
