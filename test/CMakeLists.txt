set(ProjectName test)

set(AllFiles 
	"test.cpp"
)


add_definitions(-DCMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_executable(${ProjectName} ${AllFiles})
target_link_libraries(${ProjectName} "${TORCH_LIBRARIES}")