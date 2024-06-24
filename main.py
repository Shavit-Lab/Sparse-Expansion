import utils.parser as args_parser
from utils.datautils import *
from utils.modelutils import *
import warnings


def main():
    parser = args_parser.get_parser()
    args = parser.parse_args()
    args_parser.validate_args(args)

    # return 0

    # seqlen is 4096 if args.model is llama and args.model_size == 8B else 2048
    if args.model == 'llama' and args.model_size == '8B':
        CONTEXT_LENGTH = 4096
    else:
        CONTEXT_LENGTH = 2048
    
    dataloader, testloader = get_loaders(
        args.dataset,args.model_size, args.dataset_size, seed=0, 
        model=args.model, seqlen=CONTEXT_LENGTH,
        # cache_dir="/storage2/projects/input-clustering/models/Pythia"
        cache_dir="/storage2/projects/input-clustering/models/llama/llama-chkp"
    )

    assert len(dataloader) == args.dataset_size, "Dataset size mismatch!"

    # check for suitable number of devices for Sparse Expansion
    if torch.cuda.device_count() < 2:
        warnings.warn("Less than 2 GPUs detected! Sparse Expansion might get OOM!!")
        dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        dev = torch.device('cuda:1')
    

    if args.model == 'llama':
        model = get_llama(args.model_size,
                          cache_dir="/storage2/projects/input-clustering/models/llama/llama-chkp")
        model.seqlen = CONTEXT_LENGTH
        from llama import llama_sequential
        
        mean_nll, ppl = llama_sequential(
            model,
            args.sparsity,
            args.quantize,
            args.bits,
            dataloader,
            len(dataloader),
            testloader,
            dev,
            args.no_PCA,
            args.PCA_reduction_factor,
            args.num_clusters,
            args.verbose
        )
    
    elif args.model == 'pythia':
        model = get_pythia(args.model_size,
                           cache_dir="/storage2/projects/input-clustering/models/Pythia")
        model.seqlen = CONTEXT_LENGTH
        from pythia import pythia_sequential
        mean_nll, ppl = pythia_sequential(
            model,
            args.sparsity,
            args.quantize,
            args.bits,
            dataloader,
            len(dataloader),
            testloader,
            dev,
            args.no_PCA,
            args.PCA_reduction_factor,
            args.num_clusters,
            args.verbose
        )

    print(f"Mean NLL: {mean_nll}, PPL: {ppl}")
    
    
if __name__ == "__main__":
    main()