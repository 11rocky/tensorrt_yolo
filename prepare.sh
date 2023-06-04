#!/bin/bash

script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)
third_party_dir=$script_dir/3rdparty
build_dir=$script_dir/build
install_dir=$script_dir/install

function compile_lib() {
    local lib=$1
    local copy_header=$2
    local opts=$3
    mkdir $lib
    cd $lib
    cmake $third_party_dir/$lib -DCMAKE_INSTALL_PREFIX=$install_dir $opts
    make -j & make install
    cd $build_dir
    rm -rf $lib
    if [ $copy_header ]; then
        mkdir -p $install_dir/include/$lib
        cp $third_party_dir/$lib/*.h $install_dir/include/$lib
    fi
}

git submodule update --init

if [ ! -d $build_dir ]; then
    mkdir -p $build_dir
fi

cd $build_dir
compile_lib fmt false -DFMT_MASTER_PROJECT=OFF
compile_lib argparse false
compile_lib pystring true
