project(lightning_utils LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_library(lightning_utils INTERFACE)
target_include_directories(lightning_utils INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                                     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/DS>
)

if ("$ENV{USE_OPENBLAS}" OR "${USE_OPENBLAS}")
    target_compile_options(lightning_utils INTERFACE "-D_ENABLE_BLAS=1")
    target_link_libraries(lightning_utils INTERFACE openblas)
endif()