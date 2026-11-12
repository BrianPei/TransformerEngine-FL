# Build from the repository root:
# docker build --target ci -f docker/musa/Dockerfile -t harbor.baai.ac.cn/flagos-dev/transformer-engine-fl:v2.17.0-musa-ci .
# The runtime image does not include transformer_engine_musa. This recipe adds
# TE-FL and development/test dependencies; vendor.musa CI remains blocked until
# a native extension compatible with torch_musa 2.9.1 / MUSA 5.2 is supplied.
ARG MUSA_BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-mthreads-musa5.2.0@sha256:1870e059f98d95465009f55db78d723cf77640b211ff1c5f1378d59b8e200c27
FROM ${MUSA_BASE_IMAGE} AS base
ARG MUSA_BASE_IMAGE
ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=2.17.0-musa-ci
LABEL org.opencontainers.image.title="transformer-engine-fl" \
      org.opencontainers.image.description="MUSA 5.2 TE-FL dependency environment; native vendor backend not included" \
      org.opencontainers.image.source="https://github.com/flagos-ai/TransformerEngine-FL" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${MUSA_BASE_IMAGE}" \
      io.flagos.accelerator.backend="musa" \
      io.flagos.image.stage="base" \
      io.flagos.musa.native-ci-ready="false"
ENV VIRTUAL_ENV=/flagos \
    PATH=/flagos/bin:/usr/local/musa/bin:${PATH} \
    PYTHONPATH=/opt/flagtree \
    MUSA_HOME=/usr/local/musa CUDA_HOME=/usr/local/musa \
    LD_LIBRARY_PATH=/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/musa/lib:/usr/local/openmpi/lib \
    PLATFORM=mthreads TE_FL_SKIP_CUDA=1 SKIP_CUDA_BUILD=1 \
    NVTE_WITH_CUDA=0 NVTE_WITH_MACA=0 NVTE_WITH_NCCL_EP=0 NVTE_FRAMEWORK=pytorch \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# Freeze the audited vendor stack so pip cannot replace torch or torch_musa.
RUN mkdir -p /opt/te-image \
    && printf '%s\n' \
        absl-py==2.5.1 \
        annotated-doc==0.0.6 \
        annotated-types==0.8.0 \
        anyio==4.16.0 \
        astroid==3.3.11 \
        black==24.4.2 \
        build==1.6.1 \
        cfgv==3.5.0 \
        clang-format==18.1.6 \
        click==8.5.1 \
        coloredlogs==15.0.1 \
        coverage==7.16.2 \
        cpplint==1.6.0 \
        dill==0.4.1 \
        distlib==0.4.4 \
        einops==0.8.2 \
        exceptiongroup==1.3.1 \
        expecttest==0.3.0 \
        filelock==4.1.0 \
        flag_gems==5.4.0 \
        flagcx==0.14.0+musa5.2.0 \
        flagtree==0.7.0+mthreads3.6 \
        flatbuffers==25.12.19 \
        fsspec==2026.9.0 \
        greenlet==3.5.6 \
        grpcio==1.85.1 \
        h11==0.16.0 \
        hf-xet==1.8.0 \
        httpcore2==2.14.0 \
        httpx2==2.14.0 \
        huggingface_hub==2.4.0 \
        humanfriendly==10.0 \
        identify==2.6.20 \
        idna==3.20 \
        importlib_metadata==9.0.1 \
        iniconfig==2.3.1 \
        intel-cmplr-lib-ur==2024.2.3 \
        intel-openmp==2024.2.3 \
        isort==6.1.0 \
        Jinja2==3.1.6 \
        Markdown==3.10.3 \
        markdown-it-py==4.3.0 \
        MarkupSafe==3.0.4 \
        mccabe==0.7.0 \
        mdurl==0.1.2 \
        mkl==2024.0.0 \
        ml_dtypes==0.5.4 \
        mpmath==1.3.0 \
        mypy_extensions==1.2.0 \
        networkx==3.4.2 \
        ninja==1.13.0 \
        nodeenv==1.12.0 \
        numpy==1.26.4 \
        nvdlfw_inspect==0.2.2 \
        onnx==1.23.2 \
        onnx-ir==1.1.0 \
        onnxruntime==1.23.2 \
        onnxruntime_extensions==0.15.2 \
        onnxscript==0.7.2 \
        packaging==26.3 \
        pathspec==1.1.1 \
        pillow==12.3.0 \
        platformdirs==4.12.4 \
        pluggy==1.7.0 \
        pre_commit==4.6.2 \
        protobuf==7.36.2 \
        psutil==7.2.2 \
        pybind11==3.0.3 \
        pydantic==2.14.3 \
        pydantic_core==2.50.4 \
        Pygments==2.21.0 \
        pylint==3.3.4 \
        pyproject_hooks==1.3.3 \
        pytest==8.2.1 \
        pytest-cov==7.1.0 \
        python-discovery==1.9.5 \
        PyYAML==6.0.1 \
        regex==2026.9.29 \
        rich==15.0.0 \
        safetensors==0.8.0 \
        setuptools-scm==9.2.2 \
        shellingham==1.5.4 \
        six==1.17.0 \
        SQLAlchemy==2.0.57 \
        sympy==1.14.0 \
        tbb==2021.13.1 \
        tensorboard==2.21.0 \
        tensorboard-data-server==0.7.2 \
        tokenizers==0.23.3 \
        tomli==2.5.0 \
        tomlkit==0.15.2 \
        torch==2.9.1+musa5.2.0 \
        torch_musa==2.9.1 \
        torchao==0.13.0 \
        torchaudio==2.9.1+musa5.2.0 \
        torchvision==0.24.1.post1+musa5.2.0 \
        tqdm==4.70.1 \
        transformers==5.20.2 \
        truststore==0.10.5 \
        typer==0.29.1 \
        typing-inspection==0.4.5 \
        typing_extensions==4.16.0 \
        vermin==1.9.2 \
        virtualenv==21.15.6 \
        Werkzeug==3.1.9 \
        zipp==4.1.2 > /opt/te-image/constraints.txt
ENV PIP_CONSTRAINT=/opt/te-image/constraints.txt
RUN python -c 'import importlib.metadata as m; expected={"torch":"2.9.1+musa5.2.0","torch_musa":"2.9.1","flagtree":"0.7.0+mthreads3.6","flag_gems":"5.4.0","flagcx":"0.14.0+musa5.2.0"}; actual={k:m.version(k) for k in expected}; assert actual == expected, actual' \
    && python -c 'import pathlib,sys,sysconfig; target=pathlib.Path(sysconfig.get_config_var("BINDIR"))/"python3-config"; assert target.is_file(); pathlib.Path("/flagos/bin/python3-config").symlink_to(target)'
WORKDIR /workspace
CMD ["bash"]

FROM base AS release
LABEL io.flagos.image.stage="release"
ARG TE_WHEEL_URL=https://resource.flagos.net/repository/flagos-pypi-thead/packages/transformer-engine/2.17.0+fl.0.3.0/transformer_engine-2.17.0+fl.0.3.0-py3-none-any.whl#sha256=d6abfe990299995bafcc70cbd4c28f9cad176f34064e4b887c3e0c051e74d778
RUN python -m pip install --no-cache-dir \
        pydantic==2.14.3 importlib_metadata==9.0.1 einops==0.8.2 \
        onnx==1.23.2 onnxscript==0.7.2 nvdlfw_inspect==0.2.2 "${TE_WHEEL_URL}" \
    && python -m pip check

FROM release AS dev
LABEL io.flagos.image.stage="dev"
# mcc selects GCC 12 on this base, which lacks libstdc++-12-dev initially.
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++-12 \
    && rm -rf /var/lib/apt/lists/*
RUN command -v gcc g++ cmake git ninja python3-config \
    && python -m pip install --no-cache-dir \
        build==1.6.1 black==24.4.2 clang-format==18.1.6 pre_commit==4.6.2 \
        vermin==1.9.2 cpplint==1.6.0 pylint==3.3.4 six==1.17.0 \
        regex==2026.9.29 psutil==7.2.2 tensorboard==2.21.0 \
    && python -m pip check

FROM dev AS ci
LABEL io.flagos.image.stage="ci"
RUN python -m pip install --no-cache-dir \
        pytest==8.2.1 expecttest==0.3.0 coverage==7.16.2 pytest-cov==7.1.0 \
        onnxruntime==1.23.2 onnxruntime_extensions==0.15.2 \
        transformers==5.20.2 torchao==0.13.0 \
    && python -m pip check \
    && python -m coverage --version \
    && python -c 'import importlib.metadata as m; assert m.version("transformer_engine")=="2.17.0+fl.0.3.0"'
# Existing .github/scripts/setup_musa.sh additionally requires
# transformer_engine_musa and transformer_engine_musa_torch. It must remain a
# failing gate until a supported native artifact is installed and validated.
