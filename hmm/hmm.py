import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Validate input
        if not isinstance(input_observation_states, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        # Step 1. Initialize variables
        
        # create a table to store forward probabilities: rows = time steps, columns = hidden states
        forward_table = np.zeros((len(input_observation_states), len(self.hidden_states)))
        
        # initialize first row of forward table
        for i in range(len(self.hidden_states)):
            obs_index = self.observation_states_dict[input_observation_states[0]] # get index of first observed state in observation states dictionary
            forward_table[0, i] = self.prior_p[i] * self.emission_p[i, obs_index] # prior probability of hidden state i * emission probability for hidden state i emitting the first observed state

       
        # Step 2. Calculate probabilities
        for t in range(1, len(input_observation_states)): # for each time step starting from the second time step
            for j in range(len(self.hidden_states)): # for each hidden state at time t
                sum_prob = 0 # initialize sum of probabilities
                for i in range(len(self.hidden_states)): # for each hidden state at time t-1
                    sum_prob += forward_table[t-1, i] * self.transition_p[i, j] # forward probability at time t-1 for hidden state i * transition probability from hidden state i to hidden state j
                obs_index = self.observation_states_dict[input_observation_states[t]] # get index of observed state at time t in observation states dictionary
                forward_table[t, j] = sum_prob * self.emission_p[j, obs_index] # sum of probabilities * emission probability for hidden state j emitting the observed state at time t

        # Step 3. Return final probability 
        return np.sum(forward_table[-1, :]) # sum of forward probabilities for all hidden states at final time step
        


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Validate input
        if not isinstance(decode_observation_states, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        # Step 1. Initialize variables
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states))) # for storing probabilities of hidden state at each step 
        best_path = np.zeros((len(decode_observation_states), len(self.hidden_states)), dtype=int)  # for storing index of best previous hidden state    
        
       
       # Step 2. Calculate Probabilities
        for i in range(len(self.hidden_states)): # for each hidden state at time 0
            # prior probability of hidden state i * emission probability for hidden state i emitting the first observed state
            viterbi_table[0, i] = self.prior_p[i] * self.emission_p[i, self.observation_states_dict[decode_observation_states[0]]] 
            
        # Step 3. Traceback 
        for t in range(1, len(decode_observation_states)): # for each time step, starting from the second time step
            for j in range(len(self.hidden_states)): # for each hidden state at time t
                max_prob = 0 # set at 0 for first iteration, will be updated to max probability for hidden state j at time t
                max_state_index = 0 # set at 0 for first iteration, will be updated to index of hidden state at time t-1 that gives max probability for hidden state j at time t
                for i in range(len(self.hidden_states)): # for each hidden state at time t-1
                    obs_index = self.observation_states_dict[decode_observation_states[t]] # get index of observed state at time t in observation states dictionary
                    # viterbi probability at time t-1 for hidden state i * transition probability from hidden state i to hidden state j * emission probability for hidden state j emitting the observed state at time t
                    prob = viterbi_table[t-1, i] * self.transition_p[i, j] * self.emission_p[j, obs_index] 
                    if prob > max_prob: # if we find a higher probability for hidden state j at time t
                        max_prob = prob # update max probability
                        max_state_index = i # update index of hidden state at time t-1 that gives max probability
                viterbi_table[t, j] = max_prob # store max probability in viterbi table
                best_path[t, j] = max_state_index # store index of best previous state for state j at time t

        # Step 4. Return best hidden state sequence 
        best_hidden_state_sequence = [] # list to store best hidden state sequence
        current_state = np.argmax(viterbi_table[-1, :]) # get index of hidden state with highest probability at final time step
        best_hidden_state_sequence.append(current_state) # add index of hidden state at final time step to best hidden state sequence
        for t in reversed(range(1, len(decode_observation_states))): # iterate backwards through time steps to get best hidden state sequence
            current_state = int(best_path[t, current_state]) # get index of previous state that led to current state
            best_hidden_state_sequence.append(current_state) # add index of hidden state 
        best_hidden_state_sequence.reverse() # reverse best hidden state sequence to get correct order
        best_hidden_state_sequence = [self.hidden_states[int(idx)] for idx in best_hidden_state_sequence] # convert indices to hidden state names (to match expected output format)
        return best_hidden_state_sequence
    
# %%