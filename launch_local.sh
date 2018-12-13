#!/usr/bin/env bash

DATASET_FILE=./data/BACH10/MSYNC-bach10.tfrecord
DATASET_AUDIO_DIR=./data/BACH10/Audio/
LOG_DIR=./logs/
EPOCHS=50

gcloud ml-engine local train --package-path trainer \
                             --module-name trainer.msync \
                             -- \
                             --dataset_file $DATASET_FILE \
                             --dataset_audio_dir $DATASET_AUDIO_DIR \
                             --logdir $LOG_DIR \
                             --epochs $EPOCHS
