import numpy as np

# Input sequence
sequence = "HHHHHTTTTT"

states = ('fair', 'bias')

start_probability = {'fair': 0.5, 'bias': 0.5}

transition_probability = {
    'bias': {'bias': 0.9, 'fair': 0.1},
    'fair': {'bias': 0.1, 'fair': 0.9}
}

emission_probability = {
    'fair': {'H': 0.5, 'T': 0.5},
    'bias': {'H': 0.75, 'T': 0.25}
}


def viterbi(obs, states, start_p, trans_p, emit_p):

    V = [{y:(start_p[y] * emit_p[y][obs[0]]) for y in states}]
    path = {y:[y] for y in states}

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            (prob, state) = max((V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath

    table = np.zeros((2, len(V)))
    for i, column in enumerate(V):
        table[0, i] = column.get('fair')
        table[1, i] = column.get('bias')

    # backtracking
    opt = []
    max_prob = 0.0
    best_st = None
    for st, data in V[-1].items():
        if data > max_prob:
            max_prob = data
            best_st = st
    opt.append(best_st)
    for t in range(len(V) - 2, -1, -1):
        previous = max(V[t + 1], key=lambda key: V[t + 1][key])
        opt.insert(0, previous)

    print("Most probable path [" + ", ".join(opt) + "] with probability of %.8f" % max_prob)


def forward_algorithm(obs, states, start_p, emit_p, trans_p):
    T = len(obs)
    N = len(states)

    # Initialize the forward table
    forward = np.zeros((N, T))

    # Initialize the first column of the forward table
    for s in range(N):
        forward[s][0] = start_p[states[s]] * emit_p[states[s]][obs[0]]

    # Fill in the rest of the forward table
    for t in range(1, T):
        for s in range(N):
            prob = 0
            for s_prev in range(N):
                prob += forward[s_prev][t - 1] * trans_p[states[s_prev]][states[s]]
            forward[s][t] = prob * emit_p[states[s]][obs[t]]

    # Calculate the probability of the sequence
    p_sequence = np.sum(forward[:, T - 1])

    return forward, p_sequence


def backward_algorithm(obs, states, trans_p, emit_p):
    T = len(obs)
    N = len(states)

    # Initialize the backward table
    backward = np.zeros((N, T))

    # Initialize the last column of the backward table
    for s in range(N):
        backward[s][T - 1] = 1

    # Fill in the rest of the backward table
    for t in range(T - 2, -1, -1):
        for s in range(N):
            prob = 0
            for s_next in range(N):
                prob += trans_p[states[s]][states[s_next]] * emit_p[states[s_next]][
                    obs[t + 1]] * backward[s_next][t + 1]
            backward[s][t] = prob

    return backward


# debugger was used to grab images of all non-outputted values (tables)
viterbi(list(sequence), states, start_probability, transition_probability, emission_probability)

forward_table, p_x = forward_algorithm(sequence, states, start_probability, emission_probability, transition_probability)

backward_table = backward_algorithm(sequence, states, transition_probability, emission_probability)

t_7_biased = forward_table[1][6] * backward_table[1][6] / p_x
print("\nPosterior Probability (P(T7=BX)):")
print(t_7_biased)
