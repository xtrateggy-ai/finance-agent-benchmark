#!/bin/bash
# run.sh

# source the environment/alias setup
#source ./setalias.sh

#KMP_DUPLICATE_LIB_OK=TRUE

# now python alias or env vars are visible
python launcher.py --num_tasks 5 --env secrets/secrets.env
