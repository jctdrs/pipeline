from typing import Literal

from pydantic import BaseModel
from pydantic import PositiveInt
from pydantic import model_validator


class Config(BaseModel):
    mode: Literal["Single Pass", "Monte-Carlo", "Analytic"]

    # In the case where the user does not define an 'niter' value
    # it means they only need to run the pipeline once in 'Single Pass' mode.
    # This validation shall be done in the 'preprocess_niter_based_on_error'
    # method that follows.
    niter: PositiveInt = 1

    @model_validator(mode="after")
    def niter_based_on_error(self):
        one_iter_errors = {"Single Pass", "Analytic"}
        mode = self.mode
        niter = self.niter

        # In case 'niter' is defined in 'Single Pass' or 'Analytic'
        if niter > 1 and mode in one_iter_errors:
            print(f"[WARNING] Ignoring 'niter' field for '{mode}' mode.")
            self.niter = 1
        return self

    @model_validator(mode="after")
    def error_based_on_niter(self):
        mode = self.mode
        niter = self.niter

        # In case 'niter=1' in 'Monte-Carlo'
        if niter == 1 and mode == "Monte-Carlo":
            print("[WARNING] Ignoring 'mode' field for 'niter=1'.")
            self.mode = "Single Pass"
        return self
