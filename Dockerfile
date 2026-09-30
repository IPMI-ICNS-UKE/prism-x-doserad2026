FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
ARG IMAGE_TYPE=ct
ARG MODALITY=photon
WORKDIR /opt/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN groupadd -r algo && useradd -m -r -g algo algo && mkdir -p /input /output && chown algo:algo /output
COPY prismx/ ./prismx/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY commissioning/ ./commissioning/
COPY weights/ ./weights/
COPY run.py release-manifest.json ./
ENV PRISMX_MODALITY=${MODALITY} PRISMX_IMAGE_TYPE=${IMAGE_TYPE} PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TMPDIR=/tmp
USER algo
EXPOSE 4743
LABEL org.grand-challenge.api-method="invoke"
ENTRYPOINT ["python", "run.py", "--serve"]
