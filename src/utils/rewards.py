import re

def extract_initial_number(s):
    match = re.match(r'\s*(\d+)', s)
    return int(match.group(1)) if match else -1234

def ParityReward(data_source, solution_str, ground_truth, extra_info=None):
    ground_truth_list = ground_truth.split()
    length = len(ground_truth_list)

    try:
        solution_list = solution_str.split()[:length]
    except:
        return 0
    
    n_equal = 0
    for i in range(0, length, 2):
        yi = int(ground_truth_list[i])
        zi = int(ground_truth_list[i+1])
        try:
            hat_yi = int(solution_list[i])
            hat_zi = int(solution_list[i+1])
        except:
            # If the solution is not in the expected format, return 0
            return 0
        if yi ^ zi != hat_yi ^ hat_zi:
            # If the parity does not match, return 0
            n_equal += 1
    
    # the model gets 0.1 reward if the length of the solution is equal to the length of the ground truth and 
    # all tokens are integers (hopefully they are 0 or 1)
    reward = 0.1 * (length==len(solution_str.split())) + 0.9 * (n_equal == 0)

    return reward



def CumulativeParityReward(data_source, solution_str, ground_truth, extra_info=None):
    ground_truth_list = ground_truth.split()
    length = len(ground_truth_list)

    try:
        solution_list = solution_str.split()[:length]
    except:
        return 0
    
    n_equal = 0
    for i in range(0, length, 2):
        yi = int(ground_truth_list[i])
        zi = int(ground_truth_list[i+1])
        try:
            hat_yi = int(solution_list[i])
            hat_zi = int(solution_list[i+1])
        except:
            # If the solution is not in the expected format, return 0
            return 0
        
        if i == 0:
            zi1 = 0
            hat_zi1 = 0
        else:
            zi1 = int(ground_truth_list[i-1])
            hat_zi1 = int(solution_list[i-1])
        
        if yi ^ zi ^ zi1 != hat_yi ^ hat_zi ^ hat_zi1:
            # If the parity does not match, return 0
            n_equal += 1

    
    # the model gets 0.1 reward if the length of the solution is equal to the length of the ground truth and 
    # all tokens are integers (hopefully they are 0 or 1)
    reward = 0.1 * (length==len(solution_str.split())) + 0.9 * (n_equal == 0)

    return reward


def TensorGSMReward(data_source, solution_str, ground_truth, extra_info=None):
    ground_truth_list = ground_truth.split(' and ')

    try:
        solution_list = solution_str.split('####')[1:]
    except:
        return 0
    
    if len(solution_list) < len(ground_truth_list):
        return 0
    
    for i in range(len(solution_list)):
        try:
            if extract_initial_number(solution_list[i]) != int(ground_truth_list[i].strip()):
                return 0.1
        except:
            return 0.1
    return 1