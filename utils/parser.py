import argparse
import re


def sparsity_type(value):
    # Check if the value is a float
    try:
        return float(value)
    except ValueError:
        pass

    # Check if the value is in the format N:M where N and M are integers and N <= M
    match = re.match(r"^(\d+):(\d+)$", value)
    if match:
        n, m = int(match.group(1)), int(match.group(2))
        # print("Regex match", n,m)
        if n <= m:
            return (n, m)

    raise argparse.ArgumentTypeError(
        "Sparsity must be a float or in the format N:M where N and M are integers and N <= M"
    )


def get_parser():
    parser = argparse.ArgumentParser(
        description="Argument parser with multiple groups"
    )

    # Sparse expansion specific parameters group
    sparse_group = parser.add_argument_group(
        "Sparse Expansion Specific Parameters Arguments"
    )
    sparse_group.add_argument(
        "-m",
        "--model",
        choices=["llama", "pythia"],
        required=True,
        help="Name of the model (llama/pythia)",
    )
    sparse_group.add_argument(
        "-ms", "--model_size", type=str, required=True, help="Size of the model"
    )
    sparse_group.add_argument(
        "-cd",
        "--cache_dir",
        type=str,
        default="storage",
        help="Cache directory for the model",
    )
    sparse_group.add_argument(
        "-s",
        "--sparsity",
        type=sparsity_type,
        required=True,
        help="Sparsity level (float or N:M format)",
    )
    sparse_group.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        help="Whether to quantize the model",
    )
    sparse_group.add_argument(
        "-b",
        "--bits",
        type=int,
        help="Number of bits for quantization (if quantize is true)",
    )
    sparse_group.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose logs"
    )

    # Calibration dataset group
    calibration_group = parser.add_argument_group(
        "Calibration Dataset Arguments"
    )
    calibration_group.add_argument(
        "-d",
        "--dataset",
        choices=["wikitext2", "c4"],
        required=True,
        help="Which dataset to use (wikitext2, c4)",
    )
    calibration_group.add_argument(
        "-ds",
        "--dataset-size",
        type=int,
        required=True,
        help="Size of the dataset in terms of number of sequences",
    )

    # PCA reduction group
    pca_group = parser.add_argument_group("PCA Reduction Arguments")
    pca_group.add_argument(
        "-npca",
        "--no_PCA",
        action="store_true",
        help="Do not perform PCA reduction",
    )
    pca_group.add_argument(
        "-pcrf",
        "--PCA_reduction_factor",
        type=int,
        help="PCA reduction factor (required if no_PCA is false)",
    )

    # KMeans group
    kmeans_group = parser.add_argument_group("Clustering Arguments")
    kmeans_group.add_argument(
        "-nc",
        "--num_clusters",
        type=int,
        required=True,
        help="Number of clusters for KMeans",
    )

    return parser


def validate_args(args):
    # Ensure valid model_size for the selected model
    pythia_sizes = [
        "14M",
        "31M",
        "70M",
        "160M",
        "410M",
        "1B",
        "1.4B",
        "2.8B",
        "6.9B",
        "12B",
    ]
    llama_sizes = ["7B", "8B", "13B", "1.1B"]

    if args.model == "pythia" and args.model_size not in pythia_sizes:
        raise argparse.ArgumentTypeError(
            f"Invalid model_size for pythia. Choose from {pythia_sizes}"
        )
    if args.model == "llama" and args.model_size not in llama_sizes:
        raise argparse.ArgumentTypeError(
            f"Invalid model_size for llama. Choose from {llama_sizes}"
        )

    # make model_size upper case
    args.model_size = args.model_size.upper()

    # Ensure bits is provided if quantize is true
    if args.quantize and args.bits is None:
        raise argparse.ArgumentTypeError(
            "--bits is required if --quantize is specified"
        )

    # Ensure PCA_reduction_factor is not provided if no_PCA is true
    if args.no_PCA and args.PCA_reduction_factor is not None:
        raise argparse.ArgumentTypeError(
            "--PCA_reduction_factor should not be specified if --no_PCA is set"
        )

    # Ensure PCA_reduction_factor is provided if no_PCA is false
    if not args.no_PCA and args.PCA_reduction_factor is None:
        raise argparse.ArgumentTypeError(
            "--PCA_reduction_factor is required if --no_PCA is not set"
        )
