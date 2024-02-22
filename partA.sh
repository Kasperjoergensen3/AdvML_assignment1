#!/bin/bash


# Define an array of configuration files
configs=("partA_FlowPrior.yaml" "partA_VampPrior.yaml" "partA_GaussianPrior.yaml" "partA_GMMPrior.yaml")
version="v1"
#remove reports/partA_metrics.csv if it exists
if [ -f "reports/partA_metrics_${version}.csv" ]; then
    rm reports/partA_metrics_${version}.csv
fi

#make list of random seed numbers between 100 and 10000 (hardcoded for reproducibility)
seeds=(109 231 456 789 1113 1234 5678 9012 3456 7890)


# Outer loop to run the whole process 10 times
for i in {1..10}; do
    echo "Iteration $i"
    # Loop through the array of configs
    for config in "${configs[@]}"; do
        python src/train_PartA.py --config "src/configs/${config}" --seed "${seeds[$i-1]}" --version version
    done
done
