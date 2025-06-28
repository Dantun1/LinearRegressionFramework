# Custom Stats Framework 

This is a custom framework for end-to-end statistical analysis. I want to implement key ML models in Python without existing ML libraries, so I won't be using scikitlearn/TensorFlow etc. 
I plan for it to be extensible, so I can add more models later, but for now it is just single/multiple linear regression.

## Features

## Usage

## Development Log/Design Choices

**Date**: 2025-05-29

**Topic**: Handling different datatypes with singledispatchmethod rather than instance checking with helper methods

**Description**: 

I want the Dataset class to take in np.ndarrays, pd.Series or pd.DataFrame as datatypes for the features/response properties which will be validated and standardized upon initialization. I am using the dataclass decorator to remove boilerplate code and hence, want to handle the validation/standardization process via the __post_init__ special method of dataclasses.

Rather than using many instance checks, I adjusted the code to use singledispatchmethod to trigger different standardization processes depending on the type inputted. This makes the code cleaner and more extensible as there is clear separation and new types can easily be registered later on should I wish.
