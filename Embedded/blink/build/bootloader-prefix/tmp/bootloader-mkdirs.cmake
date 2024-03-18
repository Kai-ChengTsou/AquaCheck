# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/Users/hamzakamel/esp/esp-idf/components/bootloader/subproject"
  "/Users/hamzakamel/Desktop/ECE 695 - I2I/AquaCheck/Embedded/blink/build/bootloader"
  "/Users/hamzakamel/Desktop/ECE 695 - I2I/AquaCheck/Embedded/blink/build/bootloader-prefix"
  "/Users/hamzakamel/Desktop/ECE 695 - I2I/AquaCheck/Embedded/blink/build/bootloader-prefix/tmp"
  "/Users/hamzakamel/Desktop/ECE 695 - I2I/AquaCheck/Embedded/blink/build/bootloader-prefix/src/bootloader-stamp"
  "/Users/hamzakamel/Desktop/ECE 695 - I2I/AquaCheck/Embedded/blink/build/bootloader-prefix/src"
  "/Users/hamzakamel/Desktop/ECE 695 - I2I/AquaCheck/Embedded/blink/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/hamzakamel/Desktop/ECE 695 - I2I/AquaCheck/Embedded/blink/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/hamzakamel/Desktop/ECE 695 - I2I/AquaCheck/Embedded/blink/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
