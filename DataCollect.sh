#!/usr/bin/env bash

# script to download the data
unzip_from_link() {
  local link=$1
  local dir=$2
  echo "Downloading Data"
  curl -L -o "${dir}/tmp.tar.gz" ${link}
  echo "unziping..."
  tar -xvf tmp.tar.gz
  $(rm "${dir}/tmp.tar.gz")
}

$(mkdir -p "data/")
unzip_from_link "http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz" "data"
