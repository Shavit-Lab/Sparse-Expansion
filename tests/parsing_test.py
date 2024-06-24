import subprocess
import pytest


def run_command(args):
    result = subprocess.run(
        ["python", "main.py"] + args.split(), capture_output=True, text=True
    )
    return result.returncode, result.stdout, result.stderr


@pytest.mark.parametrize(
    "args, expected_exit_code",
    [
        # Passing commands
        (
            "--model pythia --model_size 12B --sparsity 0.5 --dataset wikitext2 --dataset-size 256 --num_clusters 16 --PCA_reduction_factor 32",
            0,
        ),
        (
            "--model pythia --model_size 70M --sparsity 0.5 --dataset wikitext2 --dataset-size 256 --num_clusters 16 --PCA_reduction_factor 32",
            0,
        ),
        (
            "--model llama --model_size 7B --sparsity 0.2 --dataset c4 --dataset-size 512 --num_clusters 4 --no_PCA",
            0,
        ),
        (
            "--model llama --model_size 7B --sparsity 0.2 --dataset wikitext2 --dataset-size 256 --num_clusters 4 --PCA_reduction_factor 4",
            0,
        ),
        (
            "--model llama --model_size 7B --sparsity 2:4 --dataset wikitext2 --dataset-size 32 --num_clusters 5 --no_PCA",
            0,
        ),
        # Failing commands
        (
            "--model pythia --model_size 10B --sparsity 0.5 --dataset wikitext2 --dataset-size 1000 --num_clusters 10 --PCA_reduction_factor 16",
            1,
        ),  # Invalid model_size for pythia
        (
            "--model llama --model_size 6B --sparsity 0.2 --dataset c4 --dataset-size 500 --num_clusters 5 --PCA_reduction_factor 10",
            1,
        ),  # Invalid model_size for llama
        (
            "--model pythia --model_size 1B --sparsity 0.5 --dataset wikitext2 --dataset-size 1000 --num_clusters 10 --quantize",
            1,
        ),  # Missing bits for quantization
        (
            "--model pythia --model_size 1B --sparsity 0.5 --dataset wikitext2 --dataset-size 1000 --num_clusters 10 --no_PCA --PCA_reduction_factor 32",
            1,
        ),  # PCA_reduction_factor should not be specified if no_PCA is set
        (
            "--model pythia --model_size 6.9B --sparsity 0.5 --dataset wikitext2 --dataset-size 1000 --num_clusters 10",
            1,
        ),  # Missing PCA_reduction_factor when no_PCA is not set
        (
            "--model pythia --model_size 2.8B --sparsity 4:2 --dataset wikitext2 --dataset-size 1000 --num_clusters 10 --PCA_reduction_factor 4",
            2,
        ),  # Invalid sparsity format (N > M)
        (
            "--model pythia --model_size 12B --sparsity 1:2:3 --dataset wikitext2 --dataset-size 1000 --num_clusters 10 --PCA_reduction_factor 16",
            2,
        ),  # Invalid sparsity format
        (
            "--model pythia --model_size 1B --sparsity 1:2 --dataset wikitext2 --dataset-size 1000 --num_clusters 10 --no_PCA --PCA_reduction_factor 16",
            1,
        ),  # PCA_reduction_factor should not be specified if no_PCA is set
    ],
)
def test_argparse(args, expected_exit_code):
    exit_code, stdout, stderr = run_command(args)
    assert (
        exit_code == expected_exit_code
    ), f"Expected exit code {expected_exit_code} but got {exit_code}. Args: {args}\nStdout: {stdout}\nStderr: {stderr}"


if __name__ == "__main__":
    pytest.main()
