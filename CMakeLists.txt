cmake_minimum_required(VERSION 2.8)
project (blesdr)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

add_subdirectory(blesdr-tools)

# iq2pcap target (no LimeSuite required)
add_executable(iq2pcap
    iq2pcap.cpp
    lib/BLESDREncoder.cpp
    lib/BLESDRDecoder.cpp
)
target_include_directories(iq2pcap PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# If your build needs math:
find_library(M_LIB m)
if(M_LIB)
  target_link_libraries(iq2pcap PRIVATE ${M_LIB})
endif()
