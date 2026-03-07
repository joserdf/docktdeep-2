FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
RUN mkdir /data && chmod 777 /data

RUN pip install --no-cache-dir docktdeep==0.2.0

WORKDIR /data
COPY ckpts /ckpts

ENTRYPOINT ["docktdeep", "predict", "--model-checkpoint", "/ckpts/docktdeep-model.ckpt"]
CMD ["--help"]
