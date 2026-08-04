# Build the unit tests of this module.
#
# These are plain executables, each with its own main() and its own check() helpers, whose non-zero
#   exit status indicates failure. They are NOT registered with MRtrix3's GoogleTest harness: doing
#   so would mean editing a CMakeLists.txt that this repository does not own, and converting every
#   test to TEST() macros. Building them from a directory of our own keeps the whole arrangement
#   inside this module, at the cost of running them through the wrapping target below rather than
#   through ctest.
#
# The Dockerfile grafts this directory into an MRtrix3 checkout as "dwidenoise2_tests/" and appends
#   one add_subdirectory() line to that checkout's top-level CMakeLists.txt; see Dockerfile_debug's
#   "unit-tests" stage, which is also what runs them.
#
# mrtrix::common must be linked, not merely mrtrix::core: it carries the compile definitions of the
#   active preset, and the debug preset's STL debug flags change the ABI. A test compiled without
#   them would link against a differently-shaped standard library than the core it is testing.

file(GLOB DWIDENOISE2_UNIT_TEST_SRCS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

find_package(Threads REQUIRED)

# Not part of ALL: an ordinary build of the commands should not pay for the tests.
add_custom_target(dwidenoise2-unit-tests)

foreach(src IN LISTS DWIDENOISE2_UNIT_TEST_SRCS)
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} EXCLUDE_FROM_ALL ${src})
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/unit_tests
    )
    target_link_libraries(${name} PRIVATE
        mrtrix::core
        mrtrix::executable-version
        mrtrix::common
        Threads::Threads
    )
    add_dependencies(dwidenoise2-unit-tests ${name})
endforeach()
