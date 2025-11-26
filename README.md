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
<img width="583" height="158" alt="image" src="https://github.com/user-attachments/assets/fc5f9e13-6acf-4467-9e94-aef896263d46" />
<img width="721" height="176" alt="image" src="https://github.com/user-attachments/assets/c70a6b5a-8f09-42d9-81a2-0f307ee4868b" />
<img width="947" height="38" alt="image" src="https://github.com/user-attachments/assets/4b607b60-8894-4a06-9264-ae5ef80e0826" />


### 2. Policy, Value function and success rate for the Improved Policy
<img width="767" height="171" alt="image" src="https://github.com/user-attachments/assets/7ed6e56a-b7f5-4d9f-9e88-e416c79f994d" />
<img width="875" height="192" alt="image" src="https://github.com/user-attachments/assets/46df37d2-7a47-428d-bda1-7463e944c8ad" />
<img width="932" height="46" alt="image" src="https://github.com/user-attachments/assets/b42b4698-74c0-453f-8f29-d97b5334595e" />



### 3. Policy, Value function and success rate after policy iteration
<img width="696" height="168" alt="image" src="https://github.com/user-attachments/assets/60ac7590-d328-480e-920a-0f4ea2f0da13" />
<img width="767" height="201" alt="image" src="https://github.com/user-attachments/assets/ee9093ed-0019-457d-9043-52709525e92c" />
<img width="922" height="52" alt="image" src="https://github.com/user-attachments/assets/5bd74fd5-995a-4af1-8cbf-b6ec4f0205a3" />



## RESULT:

Therefore, policy iteration algorithm to find optimal policy by iteratively maximizing the value function is successfully implemented.
