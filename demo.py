

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    from stochprocq import MPS, HiddenMarkovModel, get_uniform_renewal, get_bio_renewal
    from stochprocq.hmm import IOHiddenMarkovModel
    return IOHiddenMarkovModel, MPS, get_bio_renewal, get_uniform_renewal, np


@app.cell
def _(MPS, get_bio_renewal, get_uniform_renewal):
    model = get_uniform_renewal(1)
    model2 = get_bio_renewal(1)
    mps = MPS.from_hmm(model)
    return model, model2, mps


@app.cell(disabled=True)
def _(model):
    model.classical_bd(4)
    return


@app.cell
def _(mps):
    mps.to_unitary()
    return


@app.cell
def _(IOHiddenMarkovModel, model, model2, np):
    io_model = np.array([model.tensor,model2.tensor])

    iohmm = IOHiddenMarkovModel(io_model)
    return (iohmm,)


@app.cell
def _(iohmm):
    iohmm.tensor
    return


@app.cell
def _(iohmm, np):
    tensor = np.sqrt(iohmm.tensor)
    pair = np.einsum('ijkl, mnop -> ikmo jlnp', tensor, tensor.conj())

    np.einsum('ijij klkl', pair)
    return


@app.cell
def _(iohmm, np):
    bell_in = np.eye(iohmm.input_alphabet_size*iohmm.dim).reshape([iohmm.input_alphabet_size,iohmm.dim]*2)
    bell_out = np.eye(iohmm.output_alphabet_size*iohmm.dim).reshape([iohmm.output_alphabet_size,iohmm.dim]*2)
    return


@app.cell(disabled=True)
def _():
    from stochprocq._utility import generate_partitions

    morphs = generate_partitions(2, set(range(4)))

    for morph in morphs:
        for m in morph:
            print(m)
        print('---')
    return


if __name__ == "__main__":
    app.run()
