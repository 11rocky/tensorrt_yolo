add_subdirectory(plugin)

add_executable(yolo_trt
    main.cpp
    utils.cpp
    calibrator.cpp
    yolo_loader.cpp
    yolo_engine.cpp
    yolo_infer.cpp
)

target_link_libraries(yolo_trt
    ${OpenCV_LIBS}
    CUDA::cudart
    nvinfer
    fmt
    pystring
    plugin
)

