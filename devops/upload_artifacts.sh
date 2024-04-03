#!/bin/bash

# DIR=./artifacts
# GCS_BUCKET=caraml-icl-bucket

gsutil -m cp -n -r ./artifacts gs://caraml-icl-bucket/artifacts
