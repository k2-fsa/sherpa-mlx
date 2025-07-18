include_directories(${PROJECT_SOURCE_DIR})

set(sources
  parse-options.cc
  text-utils.cc
)

# Always static build
add_library(sherpa-mlx-core STATIC ${sources})

set_target_properties(
    sherpa-mlx-core
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
)
