#!/bin/bash 

rsync -au --progress --exclude 'segment_enumeration_dataset' -e ssh dt01:/home/nct01/nct01058/* /Users/arnaubadiasampera/Documents/mai/mai1.2/dl/bsc

rsync -au --progress --exclude 'segment_enumeration_dataset' -e ssh /Users/arnaubadiasampera/Documents/mai/mai1.2/dl/bsc/* dt01:/home/nct01/nct01058/
