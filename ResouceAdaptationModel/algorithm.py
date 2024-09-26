import math

def resource_idle_rate(P_i, M_i):
    S_i = 2 * P_i * M_i / (P_i + M_i)
    return S_i

def availability_index(a, S_i, r_i):
    U_i = (1 + math.pow(a, 2)) * S_i / r_i / (math.pow(a, 2) / r_i + S_i)
    return U_i

def calculation_rate(forward_size, total_size):
    c_j = forward_size / total_size
    return c_j

def occupation_index(b, c_j, m_ij):
    O_ij = (1 + math.pow(b, 2)) * c_j * m_ij / (math.pow(b, 2) * m_ij + c_j)
    return O_ij

def judge_availability(U_i, O_ij):
    if U_i >= O_ij:
        print("Branch j can be install on ECU i.")
        return 1
    else:
        print("Branch j can not be install on ECU i.")
        return 0

def whole_algorithm(P_i, M_i, a, r_i, forward_size, total_size, b, m_ij):
    S_i = resource_idle_rate(P_i, M_i)
    # print("Resource idle rate: ", S_i)

    U_i = availability_index(a, S_i, r_i)
    # print("Availability index: ", U_i)

    c_j = calculation_rate(forward_size, total_size)
    print("Calculation rate: ", c_j)

    O_ij = occupation_index(b, c_j, m_ij)
    print("Occupation index: ", O_ij)
    return judge_availability(U_i, O_ij)

if __name__ == '__main__':
    processor_idle_rate = 0.6
    memory_idle_rate = 0.5
    alpha = 1
    risk_index = 4
    forward_size = 0.08
    total_size = 0.09
    beta = 2
    memory_ecu = 1    
    memory_occupied_rate = total_size / memory_ecu

    whole_algorithm(processor_idle_rate, memory_idle_rate, alpha, risk_index, forward_size, total_size, beta, memory_occupied_rate)


