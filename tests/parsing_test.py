import subprocess
import pytest

def run_command(args):
    result = subprocess.run(['python', 'main.py'] + args.split(), capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

@pytest.mark.parametrize("args, expected_exit_code", [
    ("--model pythia --model_size 1B --sparsity 0.5 --dataset wikitext2 --size 1000 --num_clusters 10", 0),
    ("--model llama --model_size 7B --sparsity 0.2 --dataset c4 --size 500 --num_clusters 5 --do_PCA False", 0),
    ("--model pythia --model_size 10B --sparsity 0.5 --dataset wikitext2 --size 1000 --num_clusters 10", 2),
    ("--model llama --model_size 6B --sparsity 0.2 --dataset c4 --size 500 --num_clusters 5", 2),
    ("--model pythia --model_size 1B --sparsity 0.5 --dataset wikitext2 --size 1000 --num_clusters 10 --quantize", 2),
    ("--model pythia --model_size 1B --sparsity 0.5 --dataset wikitext2 --size 1000 --num_clusters 10 --do_PCA True", 2),
])
def test_argparse(args, expected_exit_code):
    exit_code, stdout, stderr = run_command(args)
    assert exit_code == expected_exit_code, f"Expected exit code {expected_exit_code} but got {exit_code}. Args: {args}\nStdout: {stdout}\nStderr: {stderr}"

if __name__ == "__main__":
    pytest.main()
