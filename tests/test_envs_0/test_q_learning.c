#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>  // For DBL_MAX
#include <time.h>   // Include for time function

#define NUM_STATES 100  // Update according to your environment
#define NUM_ACTIONS 4   // Update according to your environment
#define GAMMA 0.9
#define ALPHA 0.1
#define EPSILON 0.1
#define MAX_EPISODES 1000

// Assuming these functions are defined in the library
extern void* secret_env_0_new();
extern void secret_env_0_delete(void* env_instance);
extern int* secret_env_0_available_actions(void* env_instance, int* count);
extern void secret_env_0_step(void* env_instance, int action);
extern double secret_env_0_score(void* env_instance);
extern int secret_env_0_is_game_over(void* env_instance);
extern int secret_env_0_state_id(void* env_instance);
extern int secret_env_0_is_forbidden(void* env_instance, int action);

double Q[NUM_STATES][NUM_ACTIONS];  // Q-table for storing values

// Initialize the Q-table
void initialize_Q() {
    for (int i = 0; i < NUM_STATES; i++) {
        for (int j = 0; j < NUM_ACTIONS; j++) {
            Q[i][j] = 0.0;
        }
    }
}

int choose_action(int state, int* available_actions, int num_available_actions) {
    double max_value = -DBL_MAX;
    int action = 0; // Default to a safe action if no valid actions are available

    if (num_available_actions > 0) {
        action = available_actions[rand() % num_available_actions];  // Default random action
        if ((double)rand() / RAND_MAX < EPSILON) {
            // Exploration: Randomly choose from available actions
            action = available_actions[rand() % num_available_actions];
        } else {
            // Exploitation: Choose the best action based on the Q-table
            for (int i = 0; i < num_available_actions; i++) {
                int act = available_actions[i];
                if (act < NUM_ACTIONS && Q[state][act] > max_value) {
                    max_value = Q[state][act];
                    action = act;
                }
            }
        }
    }

    return action;
}

// Update the Q-table
void update_Q(int state, int action, int next_state, double reward) {
    double max_next = -DBL_MAX;
    for (int a = 0; a < NUM_ACTIONS; a++) {  // Assuming action space is consistent
        if (Q[next_state][a] > max_next) {
            max_next = Q[next_state][a];
        }
    }
    Q[state][action] += ALPHA * (reward + GAMMA * max_next - Q[state][action]);
}

// Run a single episode
void run_episode() {
    void* env_instance = secret_env_0_new();
    int state = secret_env_0_state_id(env_instance);
    int action, next_state;
    double reward;

    printf("Starting new episode with initial state: %d\n", state);

    while (!secret_env_0_is_game_over(env_instance)) {
        int num_available_actions;
        int* available_actions = secret_env_0_available_actions(env_instance, &num_available_actions);
        if (num_available_actions <= 0) {
            printf("No available actions at state %d\n", state);
            break;
        }
        printf("Available actions at state %d (count: %d): ", state, num_available_actions);
        for (int i = 0; i < num_available_actions; i++) {
            printf("%d ", available_actions[i]);
        }
        printf("\n");

        action = choose_action(state, available_actions, num_available_actions);

        // Check if the action is forbidden
        if (secret_env_0_is_forbidden(env_instance, action)) {
            printf("Chosen action %d is forbidden at state %d\n", action, state);
            continue; // Skip this iteration and choose another action
        }

        secret_env_0_step(env_instance, action);
        next_state = secret_env_0_state_id(env_instance);
        reward = secret_env_0_score(env_instance);
        printf("State: %d, Action: %d, Reward: %.2f, Next State: %d\n", state, action, reward, next_state);
        update_Q(state, action, next_state, reward);
        state = next_state;
    }

    printf("Episode completed.\n");
    secret_env_0_delete(env_instance);
}

int main() {
    srand((unsigned int)time(NULL));  // Seed for random number generation
    initialize_Q();

    for (int i = 0; i < MAX_EPISODES; i++) {
        run_episode();
    }

    return 0;
}
