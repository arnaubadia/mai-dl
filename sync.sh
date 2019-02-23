#!/bin/bash 

rsync -r -e ssh dt01:/home/nct01/nct01058/* /Users/arnaubadiasampera/Documents/mai/mai1.2/dl/bsc

rsync -r -e ssh /Users/arnaubadiasampera/Documents/mai/mai1.2/dl/bsc/* dt01:/home/nct01/nct01058/
