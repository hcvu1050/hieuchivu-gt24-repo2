import argparse

def main():
    parser = argparse.ArgumentParser(description="Example script with command-line arguments")
    parser.add_argument('--arg1', type=int, help='An integer argument')
    parser.add_argument('--arg2', type=str, help='A string argument')

    args = parser.parse_args()

    arg1_value = args.arg1
    arg2_value = args.arg2

    print("arg1:", arg1_value)
    print("arg2:", arg2_value)

if __name__ == "__main__":
    main()