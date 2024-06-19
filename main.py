import utils.parser as args_parser

def main():
    parser = args_parser.get_parser()
    args = parser.parse_args()
    args_parser.validate_args(args)

    print("Parsed arguments in main script:")
    print(args)

if __name__ == "__main__":
    main()