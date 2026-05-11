from src.data.load_runs import load_all_runs
from src.data.run_durations import get_run_durations_with_logging
from src.features.embeddings import get_word_embeddings
from src.context.prediction_windows import create_word_prediction_window


def run_pipeline():

    print("Starting pipeline...")

    # placeholder paths
    data_path = "data/"
    
    # load runs
    runs = load_all_runs(data_path)

    # load durations
    run_durations = get_run_durations_with_logging(data_path)

    print("Pipeline structure initialized.")


if __name__ == "__main__":
    run_pipeline()