

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/rdorent/SlicerTumorSegmentator")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Reuben Dorent (Harvard University)")
set(EXTENSION_DESCRIPTION "Fully automatic Tumor Surface Segmentation, using TumorSegmentator model.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/lassoan/SlicerTotalSegmentator/main/TotalSegmentator.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/lassoan/SlicerTotalSegmentator/main/Screenshot01.jpg")
set(EXTENSION_DEPENDS "PyTorch")


project(SlicerTumorSegmentator)

#-----------------------------------------------------------------------------
set(MODULE_NAME SlicerTumorSegmentator)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  )


#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
