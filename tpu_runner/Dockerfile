FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:slim

WORKDIR /app
COPY tpu_runner/controller-requirements.txt ./
COPY tpu_runner ./tpu_runner

RUN python3 -m pip install --no-cache-dir --break-system-packages \
    -r controller-requirements.txt

ENTRYPOINT ["python3", "-m", "tpu_runner.cli"]
