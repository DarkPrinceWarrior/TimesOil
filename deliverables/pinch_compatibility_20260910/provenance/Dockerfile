FROM openporousmedia/opmreleases:2026.04_amd64@sha256:db8865d7c80440513c8c73df7ed385a3b7d2e055a0ef95f7662ec06ef6a6b3a9
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran git libopm-grid-dev=2026.04-1~noble libboost-test-dev \
    && rm -rf /var/lib/apt/lists/*
COPY opm-grid /src/opm-grid
COPY 935.patch /src/935.patch
RUN cd /src/opm-grid && git apply --check /src/935.patch && git apply /src/935.patch \
    && cmake -S . -B /build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTING=ON -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build /build --target opmgrid test_minpvprocessor -j 8 \
    && /build/bin/test_minpvprocessor --log_level=test_suite \
    && cp -a /build/lib/libopmgrid.so* /usr/local/lib/ && ldconfig
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/x86_64-linux-gnu
