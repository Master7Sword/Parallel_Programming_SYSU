OUTPUT_FILE="mpi_matrix_mul_results_$(date +%Y%m%d_%H%M%S).csv"

MATRIX_SIZES=(128 256 512 1024 2048)
PROCESSES=(1 2 4 8 16)

echo "Matrix Size,Processes,Time (s)" > "$OUTPUT_FILE"

for size in "${MATRIX_SIZES[@]}"; do
    for np in "${PROCESSES[@]}"; do
        echo "Testing size=$size with $np processes..."

        time_output=$(mpirun -np "$np" -host "localhost:$np" ./MPI "$size" "$size" "$size" 2>&1)
        
        exec_time=$(echo "$time_output" | grep "Matrix multiplication time" | awk '{print $4}')
        echo "$size,$np,$exec_time" >> "$OUTPUT_FILE"
        
        sleep 1
    done
done

echo "All tests completed! Results saved to $OUTPUT_FILE"