IF (BUILD_FUNCTIONAL_TESTS)
    add_subdirectory(functional_tests)
ENDIF()

IF (BUILD_SOS_TESTS)
    add_subdirectory(sos_tests)
ENDIF()

IF (BUILD_UNIT_TESTS)
    add_subdirectory(unit_tests)
ENDIF()