import numpy as np
import math

def ExploreThenCommit(bandit, n, m, seed=541):
    """
    Implementation of the explore-then-commit algorithm
    -----
    bandit: the bandit instance
    n: time horizon
    m: exploration time of each arm
    """

    rng = np.random.default_rng(seed)
    
    K = bandit.K()

    reward_history = {}

    # explore
    for a in range(K):
            reward_history[a] = [bandit.pull(a)]
    for t in range(m - 1):
        for a in range(K):
            reward_history[a].append(bandit.pull(a))

    # then
    est_best_arm = -1
    est_best_arm_score = -np.inf
    for a in range(K):
        candidate_score = np.mean(reward_history[a])
        if candidate_score == est_best_arm_score:
                est_best_arm = rng.choice([est_best_arm, a])
        elif candidate_score > est_best_arm_score:
                est_best_arm = a
                est_best_arm_score = candidate_score

    # commit
    for t in range(m * K, n):
        # pull the leader arm
        reward_history[est_best_arm].append(bandit.pull(est_best_arm))

def FindOptimalM(n, gap):
    """
    Find the optimal value of m in equation 6.5 in Lattimore (2018)
    -----
    n: time horizon
    gap: the suboptimal gap
    """

    return max(1, math.ceil(4 / gap ** 2 * math.log(n * gap ** 2 / 4)))

def FindUpperBound(n, gap):
    """ 
    Find the upper bound on the regret in equation 6.6 in Lattimore (2018)
    """

    return min(n * gap, gap + 4 / gap * (1 + max(0, math.log(n * gap ** 2 / 4))))

def FollowTheLeader(bandit, n, seed=541):
    # implement the Follow-the-Leader algorithm by replacing
    # the code below that just plays the first arm in every round

    rng = np.random.default_rng(seed)
    
    K = bandit.K()

    reward_history = {}

    # pull each arm once
    for a in range(K):
        reward_history[a] = [bandit.pull(a)]

    for t in range(K, n):

        # find the leader arm 
        leader = -1
        leader_score = -np.inf
        for a in range(K):
            candidate_score = np.mean(reward_history[a])
            if candidate_score == leader_score:
                leader = rng.choice([leader, a])
            elif candidate_score > leader_score:
                leader = a
                leader_score = candidate_score

        # pull the leader arm
        reward_history[leader].append(bandit.pull(leader))