from packaging import version
import optuna
from fastai.callback.core import CancelFitException
from fastai.callback.tracker import TrackerCallback

step = 0


class FastAIMultiValPruningCallback(TrackerCallback):
    def __init__(
        self, trial: optuna.Trial, monitor: str = "valid_loss", min_step: int = 0
    ):
        super().__init__(monitor=monitor)
        self.trial = trial
        self.min_step = min_step

    def after_epoch(self) -> None:
        super().after_epoch()
        # self.idx is set by TrackTrackerCallback
        global step
        self.trial.report(self.recorder.final_record[self.idx], step=step)
        step += 1
        if step > self.min_step:
            if self.trial.should_prune():
                raise CancelFitException()

    def after_fit(self) -> None:
        super().after_fit()
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial was pruned at epoch {self.epoch}.")
