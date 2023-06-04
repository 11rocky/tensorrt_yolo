# tensorrt_yolo
yolov3/v4 on tensorrt, support features:
+ datatype: fp32, fp16, int8
+ dynamic batch
+ image pre-process on gpu
+ yolo layer and mish activation on gpu with plugin

# benchmark
+ test machine  
    - **CPU**: 5700X
    - **GPU**: 3060Ti 8G
    - **DDR**: DDR4 3600MHz 8GX2
    - **OS**: Ubuntu 22.04
    - **CUDA**: 11.7
    - **TensorRT**: 8.6.1
    - **GCC**: 11.3

+ profiling data  
    | network     | fp32   | fp16   | int8   |
    |-------------|--------|--------|--------|
    | yolov3-tiny | 1.12ms | 0.61ms | 0.51ms |
    | yolov3      | 7.42ms | 2.88ms | 1.89ms |
    | yolov4      | 8.48ms | 5.02ms | 4.18ms |
    |             |        |        |        |

+ remark  
    - test input is 1x3x416x416
    - test with inference person.jpg for 100 times
    - the cost time contains image pre-process and network inference, no post-process
    - yolov3-tiny with int8 loss percision

# usage
+ requiremments:
    - git
    - GCC
    - cmake
    - opencv
    - CUDA
    - TensorRT
+ download this project to your computer, and cd into the project dir
+ init with run script: `./prepare.sh`
+ build
    - change the `CMAKE_CUDA_ARCHITECTURES` value in `CMakeLists.txt`
    - run cmd: `cd build & cmake .. & make -j`
    - then the executable was generated in `build/bin`
+ run
    - generate engine
    ```
    Usage: ./bin/yolo_trt build [--help] --cfg VAR --weights VAR --output VAR [--type VAR] [--batch VAR] [--calibration_path VAR] [--calibration_table VAR] [--dump VAR...]

    build tensorrt engine

    Optional arguments:
        -h, --help            shows help message and exits 
        -c, --cfg             yolo cfg file [required]
        -w, --weights         yolo weights file [required]
        -o, --output          output file name [required]
        -t, --type            datatype for infer: fp32, fp16, int8 [default: "fp32"]
        -b, --batch           max inference batch [default: 1]
        --calibration_path    calibration images path for int8 [default: ""]
        --calibration_table   calibration table file [default: ""]
        --dump                dump tensors [nargs: 1 or more] [default: {}]
    ```
    - inference
    ```
    Usage: infer [--help] --engine VAR --inputs VAR... [--output VAR] [--names VAR] [--thresh VAR] [--nms VAR] [--repeat VAR]

    inference with inputs

    Optional arguments:
        -h, --help    shows help message and exits 
        -e, --engine  engine file [required]
        -i, --inputs  input files [nargs: 1 or more] [required]
        -o, --output  output dir [default: "."]
        --names       class names file [default: "coco.names"]
        --thresh      thresh value [default: 0.5]
        --nms         nms value [default: 0.45]
        --repeat      repeat times [default: 1]
    ```

# depends
+ [pystring](https://github.com/imageworks/pystring.git)
+ [fmt](https://github.com/fmtlib/fmt.git)
+ [argparse](https://github.com/p-ranav/argparse.git)
