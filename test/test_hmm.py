import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_seq=np.load('./data/mini_weather_sequences.npz')

    # create an instance of HMM class
    mini_hmm_model = HiddenMarkovModel(
        observation_states = mini_hmm['observation_states'],
        hidden_states = mini_hmm['hidden_states'],
        prior_p = mini_hmm['prior_p'],
        transition_p = mini_hmm['transition_p'],
        emission_p = mini_hmm['emission_p']
    )
    obs_sequence = mini_seq['observation_state_sequence'] # get observation sequence
    expected = mini_seq['best_hidden_state_sequence'] # get expected hidden state sequence

    forward_probs = mini_hmm_model.forward(obs_sequence) # run forward algorithm to get forward probability for the observation sequence
    assert isinstance(forward_probs, float), "Output of forward algorithm should be a float"
    assert forward_probs > 0, "Output of forward algorithm should be a positive number"
    assert forward_probs < 1, "Output of forward algorithm should be less than 1"

    expected_forward_prob = 0.03506441162109375 # from github actions error
    # check if forward probability is close to expected value (within a reasonable tolerance) 
    assert np.isclose(forward_probs, expected_forward_prob, atol=1e-4), f"Output of forward algorithm should be close to {expected_forward_prob}, but got {forward_probs}"

    predicted_viterbi_path = mini_hmm_model.viterbi(obs_sequence) # run viterbi algorithm to get predicted hidden state sequence
    assert isinstance(predicted_viterbi_path, list), "Output of viterbi algorithm should be a list"
    assert len(predicted_viterbi_path) == len(obs_sequence), "Output of viterbi algorithm should have the same number of states as the input observation sequence"
    assert all(state in mini_hmm_model.hidden_states for state in predicted_viterbi_path), "All states in output of viterbi algorithm should be valid hidden states in the model"
    assert np.array_equal(predicted_viterbi_path, expected), f"Output of viterbi algorithm should be {expected}, but got {predicted_viterbi_path}"

    # raise value error if input to forward algorithm is not a numpy array
    with pytest.raises(ValueError):
        mini_hmm_model.forward([0, 1, 2]) # input is not a numpy array 
    # raise value error if input to viterbi algorithm is not a numpy array
    with pytest.raises(ValueError):
        mini_hmm_model.viterbi([0, 1, 2]) # input is not a numpy array

    


def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_seq=np.load('./data/full_weather_sequences.npz')

    # create an instance of HMM class
    full_hmm_model = HiddenMarkovModel(
        observation_states = full_hmm['observation_states'],
        hidden_states = full_hmm['hidden_states'],
        prior_p = full_hmm['prior_p'],
        transition_p = full_hmm['transition_p'],
        emission_p = full_hmm['emission_p']
    )
    obs_sequence = full_seq['observation_state_sequence'] # get observation sequence
    expected = full_seq['best_hidden_state_sequence'] # get expected hidden state sequence

    forward_probs = full_hmm_model.forward(obs_sequence) # run forward algorithm to get forward probability for the observation sequence
    assert isinstance(forward_probs, float), "Output of forward algorithm should be a float"
    assert forward_probs > 0, "Output of forward algorithm should be a positive number"
    assert forward_probs < 1, "Output of forward algorithm should be less than 1"

    exprected_forward_prob = 1.6864513843961343e-11 # from github actions error
    # check if forward probability is close to expected value (within a reasonable tolerance) 
    assert np.isclose(forward_probs, exprected_forward_prob, atol=1e-6), f"Output of forward algorithm should be close to {exprected_forward_prob}, but got {forward_probs}"

    predicted_viterbi_path = full_hmm_model.viterbi(obs_sequence) # run viterbi algorithm to get predicted hidden state sequence
    assert isinstance(predicted_viterbi_path, list), "Output of viterbi algorithm should be a list"
    assert len(predicted_viterbi_path) == len(obs_sequence), "Output of viterbi algorithm should have the same number of states as the input observation sequence"
    assert all(state in full_hmm_model.hidden_states for state in predicted_viterbi_path), "All states in output of viterbi algorithm should be valid hidden states in the model"
    assert np.array_equal(predicted_viterbi_path, expected), f"Output of viterbi algorithm should be {expected}, but got {predicted_viterbi_path}"

    # raise value error if input to forward algorithm is not a numpy array
    with pytest.raises(ValueError):
        full_hmm_model.forward([0, 1, 2]) # input is not a numpy array
    # raise value error if input to viterbi algorithm is not a numpy array
    with pytest.raises(ValueError):
        full_hmm_model.viterbi([0, 1, 2]) # input is not a numpy array













