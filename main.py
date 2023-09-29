from src.Modelling.Tests import tests_keras_functional_RnnGAT, tests_keras_subclass_RnnGAT
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--functional", action="store_true")
    parser.add_argument("--model", action="store_true")
    args = parser.parse_args()
    functional = args.functional
    model = args.model

    if functional:
        tests_keras_functional_RnnGAT.main()

    if model:
        tests_keras_subclass_RnnGAT.main()
