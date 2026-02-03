# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Rust C API bindings library
add_library(zvec_rust_binding STATIC
    shim.cc
)

target_include_directories(zvec_rust_binding
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${PROJECT_ROOT_DIR}/src/include
        ${PROJECT_ROOT_DIR}/src
)

target_link_libraries(zvec_rust_binding
    PRIVATE
        zvec_db
        zvec_core
        zvec_common
)

set_target_properties(zvec_rust_binding PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)
