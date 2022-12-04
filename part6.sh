#!/bin/bash

for i in 0.0 0.2 0.4 0.6 0.8 1.0
do
  file="$i.predict"
  echo "---SCORING FOR $i---"
  perl score.pl $file gold.trial
done