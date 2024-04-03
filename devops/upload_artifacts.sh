#!/bin/bash

# DIR=./artifacts
# GCS_BUCKET=quaild-icl-bucket

gsutil -m cp -n -r ./artifacts gs://quaild-icl-bucket/artifacts
