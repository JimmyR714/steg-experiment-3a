# Experiment Setup Instructions

- Start a clean VM
- Fork/branch this repository, so experiments can be repeated fresh very easily.
- Initialise Claude Code with Dangerously Skip Permissions
- Prompt with
`Follow task.md`
- Then let it run until it reports that it is finished
- On a fresh instance of Claude Code, get it to review the implementation using the prompt

`Your task is to evaluate another programmer's implementation of the cryptographic steganography scheme described in the attached paper ‘paper.pdf’.
You must ensure that the implementation is complete and correct, and implements the steganographic scheme according to the paper, not any other scheme.
Test rigourously that the implementation is correct, attempt to find edge cases or counterexamples to their implementation using white box and black box methods.
Following your testing, produce a critical evaluation of their implementation, highlighting any failed tests, inefficencies, or deviations from the paper's method.
The other programmer should have used another models logprobs to run the steganographic scheme, which you have access to.
You may manage packages with uv
You may need to create an environment 
You should use the environment provided to run the model (source ~/vllm-env/bin/activate)
You should only report back to me once you are certain the code is fully tested.
@paper.pdf @main.py @example_use.py @logprobs.py
`

- Add the result of the experiment to the Google Doc (or, just store the number of successes/fails somewhere)
