import pytest

from tests.helpers.run_if import RunIf
from tests.helpers.run_sh_command import run_sh_command

startfile = "src/train.py"
overrides = ["logger=[]"]


@RunIf(sh=True)
@pytest.mark.slow
def test_experiments(tmp_path):
    """Test running all available experiment configs with fast_dev_run=True."""
    command = [
        startfile,
        "-m",
        "experiment=glob(*)",
        "hydra.sweep.dir=" + str(tmp_path),
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_hydra_sweep(tmp_path):
    """Test default hydra sweep."""
    command = [
        startfile,
        "-m",
        "hydra.sweep.dir=" + str(tmp_path),
        "batch_size=4,8",
        "++trainer.fast_dev_run=true",
    ] + overrides

    run_sh_command(command)


@RunIf(sh=True)
@pytest.mark.slow
def test_optuna_sweep(tmp_path):
    """Test optuna sweep."""
    command = [
        startfile,
        "-m",
        "hparams_search=setfit_optuna",
        "hydra.sweep.dir=" + str(tmp_path),
        "hydra.sweeper.n_trials=5",
        "hydra.sweeper.sampler.n_startup_trials=5",
        "++trainer.fast_dev_run=true",
    ] + overrides
    run_sh_command(command)
