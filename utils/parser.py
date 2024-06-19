import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Argument parser with multiple groups")

    # Sparse expansion specific parameters group
    sparse_group = parser.add_argument_group('Sparse Expansion Specific Parameters')
    sparse_group.add_argument('--model', choices=['llama', 'pythia'], required=True, help="Name of the model (llama/pythia)")
    sparse_group.add_argument('--model_size', type=str, required=True, help="Size of the model")
    sparse_group.add_argument('--sparsity', type=float, required=True, help="Sparsity level (float)")
    sparse_group.add_argument('--quantize', action='store_true', help="Whether to quantize the model")
    sparse_group.add_argument('--bits', type=int, help="Number of bits for quantization (if quantize is true)")

    # Calibration dataset group
    calibration_group = parser.add_argument_group('Calibration Dataset')
    calibration_group.add_argument('--dataset', choices=['wikitext2', 'c4'], required=True, help="Which dataset to use (wikitext2, c4)")
    calibration_group.add_argument('--size', type=int, required=True, help="Size of the dataset in terms of number of sequences")

    # PCA reduction group
    pca_group = parser.add_argument_group('PCA Reduction')
    pca_group.add_argument('--do_PCA', type=bool, default=True, help="Whether to perform PCA reduction (default: True)")
    pca_group.add_argument('--PCA_reduction_factor', type=float, help="PCA reduction factor (optional if do_PCA is false)")

    # KMeans group
    kmeans_group = parser.add_argument_group('KMeans')
    kmeans_group.add_argument('--num_clusters', type=int, required=True, help="Number of clusters for KMeans")

    return parser

def validate_args(args):
    # Ensure valid model_size for the selected model
    pythia_sizes = ['14M', '31M', '70M', '160M', '410M', '1B', '1.4B', '2.8B', '6.9B', '12B']
    llama_sizes = ['7B', '8B', '13B']

    if args.model == 'pythia' and args.model_size not in pythia_sizes:
        raise argparse.ArgumentTypeError(f"Invalid model_size for pythia. Choose from {pythia_sizes}")
    if args.model == 'llama' and args.model_size not in llama_sizes:
        raise argparse.ArgumentTypeError(f"Invalid model_size for llama. Choose from {llama_sizes}")

    # Ensure bits is provided if quantize is true
    if args.quantize and args.bits is None:
        raise argparse.ArgumentTypeError("--bits is required if --quantize is specified")

    # Ensure PCA_reduction_factor is provided if do_PCA is true
    if args.do_PCA and args.PCA_reduction_factor is None:
        print(args.do_PCA, args.PCA_reduction_factor)
        raise argparse.ArgumentTypeError("--PCA_reduction_factor is required if --do_PCA is specified")

# if __name__ == "__main__":
#     parser = get_parser()
#     args = parser.parse_args()
#     validate_args(args)
#     print("Parsed arguments:")
#     print(args)
