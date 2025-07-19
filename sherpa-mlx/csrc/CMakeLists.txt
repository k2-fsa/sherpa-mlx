set(sources
  circular-buffer.cc
  file-utils.cc
  parse-options.cc
  silero-vad-model-config.cc
  silero-vad-model.cc
  text-utils.cc
  vad-model-config.cc
  vad-model.cc
  voice-activity-detector.cc
  wave-reader.cc
  wave-writer.cc
)

# Always static build
add_library(sherpa-mlx-core STATIC ${sources})
target_include_directories(
  sherpa-mlx-core
  PUBLIC
    ${PROJECT_SOURCE_DIR}
    $ENV{MLX_CMAKE_DIR}/include
)

set_target_properties(
    sherpa-mlx-core
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
)

target_link_libraries(
  sherpa-mlx-core
  PUBLIC
  -L$ENV{MLX_CMAKE_DIR}/lib
  -lmlx
  )

add_executable(sherpa-mlx-vad sherpa-mlx-vad.cc)
target_link_libraries(sherpa-mlx-vad sherpa-mlx-core)

set(srcs_to_check)
foreach(s IN LISTS sources)
  list(APPEND srcs_to_check ${CMAKE_CURRENT_LIST_DIR}/${s})
endforeach()

# For clang-tidy
add_custom_target(
  clang-tidy-check
  clang-tidy -p ${CMAKE_BINARY_DIR}/compile_commands.json --config-file ${PROJECT_SOURCE_DIR}/.clang-tidy ${srcs_to_check}
  DEPENDS ${sources})

add_custom_target(check DEPENDS clang-tidy-check)
