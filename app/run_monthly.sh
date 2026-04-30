#!/bin/bash

# Get previous month in YYYY-MM format
AS_OF_MONTH=$(date -d "$(date +%Y-%m-01) -1 month" +%Y-%m)

echo "Running for as_of_month=$AS_OF_MONTH"

curl -X POST "http://api:8000/predict/all/${AS_OF_MONTH}"

echo "Done"