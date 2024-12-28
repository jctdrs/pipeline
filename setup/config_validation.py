from pydantic import BaseModel, PositiveInt, model_validator
from typing import Literal


class Config(BaseModel):
    mode: Literal["Single Pass", "Monte-Carlo", "Automatic Differentiation"]

    # In the case where the user does not define an 'niter' value
    # it means they only need to run the pipeline once in 'Single Pass' mode.
    # This validation shall be done in the 'preprocess_niter_based_on_error'
    # method that follows.
    niter: PositiveInt = 1

    @model_validator(mode="after")
    def niter_based_on_error(self):
        one_iter_errors = {"Single Pass", "Automatic Differentiation"}
        mode = self.mode
        niter = self.niter

        # In case 'niter' is defined in 'Single Pass' or 'Automatic Differentiation'
        if niter > 1 and mode in one_iter_errors:
            print(f"[WARNING] Ignoring 'niter' field for '{mode}' mode.")
            self.niter = 1
        return self
