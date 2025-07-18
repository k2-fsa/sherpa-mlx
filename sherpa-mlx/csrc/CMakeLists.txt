set(sources
  parse-options.cc
  text-utils.cc
)

# Always static build
add_library(sherpa-mlx-core STATIC ${sources})
target_include_directories(
  sherpa-mlx-core
  PUBLIC
    ${PROJECT_SOURCE_DIR}
)

set_target_properties(
    sherpa-mlx-core
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
)
