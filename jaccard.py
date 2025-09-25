import faiss
import numpy as np
import scipy
import os
import argparse
import glob
import time
import shutil
import matplotlib.pyplot as plt
import random

def index_vectors(output_dir):
    identifiers = []
    vectors = []

    # Ensure output_dir exists and contains only vectors.txt
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        files = os.listdir(output_dir)
        allowed = {"vectors.txt"}
        for f in files:
            if f not in allowed:
                os.remove(os.path.join(output_dir, f))
    
    output_index = os.path.join(output_dir, "faiss.index")
    input_file = os.path.join(output_dir, "vectors.txt")
    output_pos = os.path.join(output_dir, "positions.bin")

    with open(input_file, "r") as f, open(output_pos, "wb") as pos_f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) != 2:
                continue
            identifier = parts[0].strip()
            vec_str = parts[1].strip()
            vec = np.fromstring(vec_str, sep=" ")
            identifiers.append(identifier)
            vectors.append(vec)
            pos_f.write(pos.to_bytes(8, byteorder="little"))

    vectors = np.stack(vectors).astype("float32")
    dim = vectors.shape[1]

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dim)  # Use inner product metric
    index.add(vectors)
    faiss.write_index(index, output_index)
    print(f"Indexed {len(identifiers)} vectors of dimension {dim} into {output_index}.")

def search_index(index_folder, query_file, j):
    index = faiss.read_index(index_folder+"faiss.index")
    queries = []

    with open(query_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vec = np.fromstring(line.split(":")[1], sep=" ")
            queries.append(vec)

    queries = np.stack(queries).astype("float32")
    query_norms = np.linalg.norm(queries, axis=1)
    faiss.normalize_L2(queries)
    minimum_required_inner_product = 2*j/(1+j)
    
    # Start with a reasonable number of neighbors, increase until all above threshold are found
    nb_searches = 10
    found_all = False
    while not found_all:
        D, I = index.search(queries, nb_searches)
        # Check if all neighbors above threshold are included for each query
        found_all = True
        for i in range(D.shape[0]):
            if np.any(D[i] > minimum_required_inner_product):
                # If there are neighbors above threshold, check if the last returned is still above threshold
                if D[i][-1] > minimum_required_inner_product:
                    # Might need to fetch more neighbors
                    found_all = False
                    break
        if not found_all:
            nb_searches *= 2  # Double the number of neighbors and try again
    
    positions_file = os.path.join(index_folder, "positions.bin")
    vectors_file = os.path.join(index_folder, "vectors.txt")

    with open(positions_file, "rb") as pos_f, open(vectors_file, "r") as vec_f:
        pos_f.seek(0, os.SEEK_END)
        num_vectors = pos_f.tell() // 8
        pos_f.seek(0)
        positions = [int.from_bytes(pos_f.read(8), byteorder="little") for _ in range(num_vectors)]

        recovered_vectors = []
        for idx in I.flatten():
            if idx < 0 or idx >= len(positions):
                recovered_vectors.append(None)
                continue
            vec_f.seek(positions[idx])
            line = vec_f.readline().strip()
            recovered_vectors.append(line)

    for i, neighbors in enumerate(I):
        results = []
        for rank, idx in enumerate(neighbors):
            neighbor_line = recovered_vectors[rank]
            neighbor_id = neighbor_line.split(":")[0]
            neighbor_vec = np.fromstring(neighbor_line.split(":")[1], sep=" ")
            neighbor_norm = np.linalg.norm(neighbor_vec)
            query_norm = query_norms[i]
            inner_product = D[i, rank]

            jaccard_index = inner_product * query_norm * neighbor_norm / (neighbor_norm**2 + query_norm**2 - inner_product * query_norm * neighbor_norm)

            if jaccard_index > j:
                results.append((neighbor_id, jaccard_index, inner_product, neighbor_norm, query_norm))

        # Sort results by jaccard_index descending
        results.sort(key=lambda x: x[1], reverse=True)

        print(f"Query {i}:")
        for rank, (neighbor_id, jaccard_index, inner_product, neighbor_norm, query_norm) in enumerate(results):
            print(f"  Neighbor {rank}: {neighbor_id} (jaccard: {jaccard_index:.4f}), inner_product: {inner_product:.4f} {neighbor_norm} {query_norm}")

def test():

    index_folder = "/home/roland-faure/Documents/postdoc/penn_state/pairwise_comp/faiss_db"

    vectors_file = os.path.join(index_folder, "vectors.txt")
    with open(vectors_file, "r") as f:
        num_vectors = sum(1 for line in f if line.strip())
    print(f"Number of vectors in {vectors_file}: {num_vectors}")

    # Draw 20 random vector indices
    random_indices = random.sample(range(num_vectors), 20)

    # Read all lines into memory for easy access
    with open(vectors_file, "r") as f:
        all_lines = [line.strip() for line in f if line.strip()]

    output_file = os.path.join(index_folder, "random_neighbors.txt")
    results = []

    # Load FAISS index and positions
    index = faiss.read_index(os.path.join(index_folder, "faiss.index"))
    positions_file = os.path.join(index_folder, "positions.bin")
    with open(positions_file, "rb") as pos_f:
        pos_f.seek(0, os.SEEK_END)
        num_vectors = pos_f.tell() // 8
        pos_f.seek(0)
        positions = [int.from_bytes(pos_f.read(8), byteorder="little") for _ in range(num_vectors)]

    # For each random query, search for neighbors using the index
    all_necessary_vectors = set()
    for idx in random_indices:
        query_line = all_lines[idx]
        query_id, query_vec_str = query_line.split(":")
        all_necessary_vectors.add(query_id)
        query_vec = np.fromstring(query_vec_str, sep=" ").astype("float32").reshape(1, -1)
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 20 :
            print("this one is ", query_norm)
            continue
        faiss.normalize_L2(query_vec)
        # Search for a reasonable number of neighbors
        nb_searches = 30
        D, I = index.search(query_vec, nb_searches)
        # Recover neighbor info
        with open(vectors_file, "r") as vec_f:
            for rank, neighbor_idx in enumerate(I[0]):
                if neighbor_idx < 0 or neighbor_idx >= len(positions):
                    continue
                vec_f.seek(positions[neighbor_idx])
                neighbor_line = vec_f.readline().strip()
                neighbor_id, neighbor_vec_str = neighbor_line.split(":")
                neighbor_vec = np.fromstring(neighbor_vec_str, sep=" ").astype("float32")
                neighbor_norm = np.linalg.norm(neighbor_vec)

                if neighbor_norm < 20 :
                    continue
                if neighbor_norm == 0 or query_norm == 0:
                    continue
                actual_inner_product = np.dot(query_vec.flatten(), neighbor_vec)
                jaccard = actual_inner_product / (neighbor_norm**2 + query_norm**2 - actual_inner_product)
                if jaccard > 0.05 and neighbor_id != query_id:
                    results.append((query_id, neighbor_id, jaccard))
                    all_necessary_vectors.add(neighbor_id)

    print("queried the 20 vecotrs")

    # Load all hashes from "hashes.txt"
    hashes_file = "all_hashes.txt"
    hashes_dict = {}
    if os.path.exists(hashes_file):
        with open(hashes_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                sample_name = parts[0].strip()
                if sample_name in all_necessary_vectors :
                    hash_list = parts[1].strip().split()
                    hashes_dict[sample_name] = hash_list
        print(f"Loaded hashes for {len(hashes_dict)} samples from {hashes_file}")
    else:
        print(f"Hashes file {hashes_file} not found.")

    # For every pair with jaccard > 0.05, compute their actual jaccard from the hashes
    actual_jaccard_results = []
    for query_id, neighbor_id, jaccard in results:
        hashes1 = set(hashes_dict.get(query_id, []))
        hashes2 = set(hashes_dict.get(neighbor_id, []))
        if not hashes1 or not hashes2:
            continue
        intersection = hashes1 & hashes2
        union = hashes1 | hashes2
        if union:
            actual_jaccard = len(intersection) / len(union)
            actual_jaccard_results.append((query_id, neighbor_id, jaccard, actual_jaccard))

    # Optionally print or save the results
    for query_id, neighbor_id, jaccard, actual_jaccard in actual_jaccard_results:
        print(f"{query_id} vs {neighbor_id}: vector_jaccard={jaccard:.4f}, hash_jaccard={actual_jaccard:.4f}")

    # Scatter plot: x = true jaccard, y = estimated jaccard
    xs = [actual_jaccard for _, _, _, actual_jaccard in actual_jaccard_results]
    ys = [jaccard for _, _, jaccard, _ in actual_jaccard_results]
    plt.figure(figsize=(6,6))
    plt.scatter(xs, ys, alpha=0.5)

def main():

    test()
    sys.exit()

    parser = argparse.ArgumentParser(description="FAISS indexer and searcher.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_index = subparsers.add_parser("index", help="Index vectors from a file.")
    parser_index.add_argument("output_index", type=str, help="Path to output FAISS index folder.")

    parser_search = subparsers.add_parser("search", help="Search vectors in a FAISS index.")
    parser_search.add_argument("index_folder", type=str, help="Path to FAISS index folder.")
    parser_search.add_argument("query_file", type=str, help="Path to query file (one vector per line).")
    parser_search.add_argument("-j", type=float, default=0.1, help="Retrieve all datasets with higher Jaccard index")

    args = parser.parse_args()

    if args.command == "index":
        index_vectors(args.output_index)
    elif args.command == "search":
        if args.index_folder[-1] != "/":
            args.index_folder += "/"
        search_index(args.index_folder, args.query_file, args.j)

if __name__ == "__main__":
    main()