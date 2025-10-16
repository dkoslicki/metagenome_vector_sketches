# metagenome_vector_sketches
Repository with code to sketch genomic data with random projection

## Usage

Create projected vectors from fracminhash data into the index folder:
``` shell
../project_everything <input_folder> <index_folder> [-t threads] [-d dimension] [-s strategy]
```

Example:

```shell
../project_everything toy toy_index/ -t 8 -d 1024 -s 0
```

Use the vectors to create FAISS index:

``` shell
python3 ../jaccard.py index toy_index -t 8
```

Use the vectors to create pairwise matrix:

``` shell
../pairwise_comp_optimized --vectors toy_index/vectors.bin --dimension 1024 --output_folder toy_index/ --max_memory_gb 12 --num_threads 8
```

Then, to query using the AVA matrix:

``` shell
 ../query_ava_matrix --matrix_folder toy_index/shard_0/ --query_ids 10
```
