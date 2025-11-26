# POLICY ITERATION ALGORITHM

## AIM
Implement policy iteration algorithm to find optimal policy by iteratively maximizing the value function.

## PROBLEM STATEMENT
The aim of this experiment is to find optimal policy for the mdp using policy iteration. Policy iteration includes policy evaluation and policy improvement where evaluation function is used to find optimal value function of each state and then improvement function is used to find best policy by comparing all the action value function as well as policy.

## POLICY ITERATION ALGORITHM
# Step 1:
Import required libraries.
# Step 2:
Load the frozen lake environment.
# Step 3:
Define the value evaluation, value improvement and value iteration functions.
# Step 4: 
Run the functions and display the results.

## POLICY IMPROVEMENT FUNCTION
### Name: VINOTH M P
### Register Number: 212223240182
```python
def policy_improvement(V,P,gamma=1.0):
  Q=np.zeros((len(P),len(P[0])),dtype=np.float64)
  for s in range(len(P)):
    for a in range(len(P[s])):
      for prob, next_state, reward, done in P[s][a]:
        Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
      new_pi=lambda s: {s:a for s,a in enumerate(np.argmax(Q, axis=1))}[s]
  return new_pi
```
## POLICY ITERATION FUNCTION
### Name: VINOTH M P
### Register Number: 212223240182
```python
def policy_iteration(P,gamma=1.0,theta=1e-10):
  random_actions=np.random.choice(tuple(P[0].keys()),len(P))
  pi=lambda s: {s:a for s, a in enumerate(random_actions)}[s]
  while True:
    old_pi={s: pi(s) for s in range(len(P))}
    V=policy_evaluation(pi,P,gamma,theta)
    pi=policy_improvement(V,P,gamma)
    if old_pi=={s:pi(s) for s in range(len(P))}:
      break
  return V,pi

```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
<img width="475" height="99" alt="image" src="https://github.com/user-attachments/assets/62b06255-2ddb-400d-b991-4aca36ae9fff" />
<img width="419" height="111" alt="image" src="https://github.com/user-attachments/assets/426bc0b0-2fad-44e1-990a-562e0922baca" />


### 2. Policy, Value function and success rate for the Improved Policy
<img width="443" height="74" alt="image" src="https://github.com/user-attachments/assets/1b2f5f0f-42aa-4155-b155-11d3fabfffc8" />

<img width="464" height="108" alt="image" src="https://github.com/user-attachments/assets/a63bf37d-0f0e-497a-beb7-261ee0783ec4" />



### 3. Policy, Value function and success rate after policy iteration
<img width="396" height="80" alt="image" src="https://github.com/user-attachments/assets/0ec4998f-6948-461e-b3da-a1c2adba2847" />

<img width="459" height="134" alt="image" src="https://github.com/user-attachments/assets/a982323c-3252-4f0c-aaf4-85ba3bad4a13" />





## RESULT:

Therefore, policy iteration algorithm to find optimal policy by iteratively maximizing the value function is successfully implemented.
