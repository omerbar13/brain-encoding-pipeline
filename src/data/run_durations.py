# src/data/run_durations.py

import logging
import os
from nilearn.image import load_img

logger = logging.getLogger(__name__)

# Run 4 is never loaded but its duration still counts toward cumulative time.
# Volume count is hard-coded as in the notebook (488 volumes × 2s TR = 976s).
_RUN4_VOLUMES  = 488
_RUN4_DURATION = 976.0


def get_run_durations_with_logging(
    base_dir: str,
    num_runs: int = 8,
    tr_duration: float = 2.0,
) -> list[float]:
    """
    Load each fMRI run and compute its duration in seconds.

    Run 4 is never loaded from disk — its volume count is hard-coded
    (488 volumes, 976 s) matching the notebook. All other runs are loaded
    and their duration is derived from shape[-1] * tr_duration.

    Parameters
    ----------
    base_dir : str
        Directory containing the hyperaligned NIfTI files.
    num_runs : int
        Total number of runs including run 4 (default 8).
    tr_duration : float
        TR length in seconds (default 2.0).

    Returns
    -------
    list of float
        Duration in seconds for all runs including run 4 at index 3.
        Length is always num_runs.
    """
    run_durations  = []
    cumulative_time = 0.0

    logger.info("=== Run Duration Analysis ===")
    logger.info("TR: %.1f seconds", tr_duration)

    for run_num in range(1, num_runs + 1):

        run_path = os.path.join(
            base_dir,
            f"group_avg_movie_run_{run_num}_hyperaligned.nii"
        )
        logger.info("Run %d | path: %s", run_num, run_path)

        try:
            if run_num == 4:
                # Run 4 is skipped in all processing but its duration
                # must be included so cumulative start times stay correct
                duration = _RUN4_DURATION
                run_durations.append(duration)
                cumulative_time += duration
                logger.info(
                    "Run 4 (skipped) | hard-coded volumes=%d | duration=%.1fs",
                    _RUN4_VOLUMES, duration
                )

            else:
                run_img     = load_img(run_path)
                n_timepoints = run_img.shape[-1]
                duration    = n_timepoints * tr_duration
                run_durations.append(duration)

                logger.info(
                    "Run %d | shape=%s | volumes=%d | duration=%.1fs | "
                    "time range: %.1fs → %.1fs",
                    run_num, run_img.shape, n_timepoints, duration,
                    cumulative_time, cumulative_time + duration,
                )
                cumulative_time += duration

        except Exception as e:
            logger.error("Error loading run %d: %s", run_num, e)
            run_durations.append(0.0)

    logger.info("=== Run Duration Summary ===")
    for i, dur in enumerate(run_durations, 1):
        logger.info("Run %d: %.1fs", i, dur)
    logger.info(
        "Total time (excl. run 4): %.1fs | Total (incl. run 4): %.1fs",
        cumulative_time - _RUN4_DURATION, cumulative_time
    )

    return run_durations