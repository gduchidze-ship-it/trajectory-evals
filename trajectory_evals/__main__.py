"""Run the command line: python -m trajectory_evals <command> <agent> ..."""

from dotenv import load_dotenv

from trajectory_evals.cli import main

if __name__ == "__main__":
    load_dotenv()
    raise SystemExit(main())
