#!/bin/bash

# DIR=./artifacts
# GCS_BUCKET=quaild-icl-bucket

gsutil -m cp -n -r ./checkpoints gs://quaild-icl-bucket/checkpoints
