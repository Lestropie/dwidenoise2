ARG MAKE_JOBS="1"
ARG DEBIAN_FRONTEND="noninteractive"

FROM debian:bookworm-slim AS base
FROM buildpack-deps:bookworm AS base-builder

FROM base-builder AS mrtrix3-builder

# Git commitish from which to build MRtrix3.
# This is branch "dev" as at 2025-10-21
ARG MRTRIX3_GIT_COMMITISH="26965d57b374a733ac0c583d3b92bad17923128a"

RUN apt-get -qq update \
    && apt-get install -yq --no-install-recommends \
        cmake \
        libfftw3-dev \
        ninja-build \
        pkg-config \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone the version of MRtrix3 required
ARG MAKE_JOBS
WORKDIR /src/mrtrix3
RUN git clone https://github.com/MRtrix3/mrtrix3.git . \
    && git checkout $MRTRIX3_GIT_COMMITISH

# As support for external modules compiling against cmake MRtrix3 is not yet committed,
#   just insert the code for this tool into the MRtrix3 directory
#   prior to cmake configuration
COPY cpp/cmd/dwidenoise2.cpp /src/mrtrix3/cpp/cmd/dwidenoise2.cpp
COPY cpp/cmd/dwi2noise.cpp /src/mrtrix3/cpp/cmd/dwi2noise.cpp
COPY cpp/core/denoise /src/mrtrix3/cpp/core/denoise

RUN cmake -Bbuild -GNinja -DMRTRIX_BUILD_GUI=OFF --preset=release \
    && cmake --build build --target dwidenoise2 dwi2noise mrcalc mrcat mrconvert

# Build final image.
FROM base AS final

# Install runtime system dependencies.
RUN apt-get -qq update \
    && apt-get install -yq --no-install-recommends \
        less \
        libfftw3-single3 \
        libfftw3-double3 \
        libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

#COPY --from=mrtrix3-builder /opt/mrtrix3 /opt/mrtrix3
COPY --from=mrtrix3-builder /src/mrtrix3/build /opt/mrtrix3

# Technically not required if it's the entrypoint
ENV PATH="/opt/mrtrix3/bin:$PATH"

ENV LD_LIBRARY_PATH="/opt/mrtrix3/cpp/core/"

# TODO Define an entrypoint for the container
#ENTRYPOINT ["/opt/mrtrix3/bin/dwidenoise2"]
