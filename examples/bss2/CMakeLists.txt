add_custom_target(BSS2)
set_target_properties(BSS2 PROPERTIES FOLDER Examples)

set(LLVM_LINK_COMPONENTS
   Support
   )

macro(add_compiler name)
  add_dependencies(BSS2 ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_compiler name)

add_compiler(bss2c
    mlir/Dialect.cpp
    bss2c.cpp
    )

include_directories(include/)
target_link_libraries(bss2c
  PRIVATE
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
)
