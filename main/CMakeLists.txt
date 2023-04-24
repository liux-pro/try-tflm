idf_component_register(
        SRC_DIRS "."
        INCLUDE_DIRS "."
)

#add_custom_command(OUTPUT model_tflite.h
#        COMMAND bmp2h -i ${COMPONENT_DIR}/logo.bmp -o tflite_model.h
#        DEPENDS ${COMPONENT_DIR}/logo.bmp
#        VERBATIM)
#
#add_custom_target(logo DEPENDS logo.h)
#add_dependencies(${COMPONENT_LIB} logo)
#
#set_property(DIRECTORY "${COMPONENT_DIR}" APPEND PROPERTY
#        ADDITIONAL_MAKE_CLEAN_FILES tflite_model.h)