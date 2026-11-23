# syntax=docker/dockerfile:1
# The base and every Python dependency are immutable.  Rebuild intentionally
# when the pinned LiveCodeBench verifier changes.
FROM python:3.11-slim-bookworm@sha256:2fc9207f64226cb05ac317cee0bab6fa55a9ea311ce5a086baddd4b4a83c2d3c

ARG LCB_COMMIT=28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24

ADD --checksum=sha256:56c7c42aee979def86b69547ecd6d5a877a0e4940af8923bf4325ca0defecbaf \
    https://github.com/LiveCodeBench/LiveCodeBench/archive/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24.tar.gz \
    /tmp/livecodebench.tar.gz

# The upstream package declaration currently includes only the top-level
# lcb_runner module. Overlay the pinned archive so evaluator subpackages are
# present as well.
RUN python -m pip install --no-cache-dir \
        numpy==2.2.5 \
        PyYAML==6.0.3 \
        tqdm==4.67.1 \
    && python -m pip install --no-cache-dir --no-deps /tmp/livecodebench.tar.gz \
    && mkdir /tmp/livecodebench-source \
    && tar -xzf /tmp/livecodebench.tar.gz \
        -C /tmp/livecodebench-source --strip-components=1 \
    && cp -a /tmp/livecodebench-source/lcb_runner/. \
        /usr/local/lib/python3.11/site-packages/lcb_runner/ \
    && printf '%s\n' "$LCB_COMMIT" > /opt/livecodebench-commit \
    && rm -r /tmp/livecodebench-source /tmp/livecodebench.tar.gz

RUN useradd --create-home --uid 65532 verifier
USER verifier
WORKDIR /work
