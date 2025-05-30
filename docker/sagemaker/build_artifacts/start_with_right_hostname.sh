#!/usr/bin/env bash

if [[ "$1" = "train" ]]; then
     CURRENT_HOST=$(jq .current_host  /opt/ml/input/config/resourceconfig.json)
     sed -ie "s/PLACEHOLDER_HOSTNAME/$CURRENT_HOST/g" /usr/local/bin/changehostname.c
     gcc -o changehostname.o -c -fPIC -Wall /usr/local/bin/changehostname.c
     gcc -o libchangehostname.so -shared -export-dynamic changehostname.o -ldl
     CWD=$(pwd)
     LD_PRELOAD=$CWD/libchangehostname.so train
else
     "$@"
fi