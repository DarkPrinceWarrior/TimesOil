"""New conversion proposal with bounded feedback; preserve the rejected first run."""
from run_z_guarded_policy import run


if __name__ == '__main__':
    run(conversion_search=True, output_name='timesfm-conversion-z-repair-20260910')
