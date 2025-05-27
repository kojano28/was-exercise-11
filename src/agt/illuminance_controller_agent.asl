//illuminance controller agent

/*
* The URL of the W3C Web of Things Thing Description (WoT TD) of a lab environment
* Simulated lab WoT TD: "https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl"
* Real lab WoT TD: Get in touch with us by email to acquire access to it!
*/

/* Initial beliefs and rules */

// the agent has a belief about the location of the W3C Web of Thing (WoT) Thing Description (TD)
// that describes a lab environment to be learnt
learning_lab_environment("https://raw.githubusercontent.com/Interactions-HSG/example-tds/was/tds/interactions-lab.ttl").

// the agent believes that the task that takes place in the 1st workstation requires an indoor illuminance
// level of Rank 2, and the task that takes place in the 2nd workstation requires an indoor illumincance 
// level of Rank 3. Modify the belief so that the agent can learn to handle different goals.
task_requirements([1, 2]).



/* Initial goals */
!start. // the agent has the goal to start

/* 
 * Plan for reacting to the addition of the goal !start
 * Triggering event: addition of goal !start
 * Context: the agent believes that there is a WoT TD of a lab environment located at Url, and that 
 * the tasks taking place in the workstations require indoor illuminance levels of Rank Z1Level and Z2Level
 * respectively
 * Body: (currently) creates a QLearnerArtifact and a ThingArtifact for learning and acting on the lab environment.
*/
@start
+!start : learning_lab_environment(Url) 
  & task_requirements([Z1Level, Z2Level]) <-

  .print("Hello world");
  .print("I want to achieve Z1Level=", Z1Level, " and Z2Level=",Z2Level);

  // creates a QLearner artifact for learning the lab Thing described by the W3C WoT TD located at URL
  makeArtifact("qlearner", "tools.QLearner", [Url], QLArtId);

  // creates a ThingArtifact artifact for reading and acting on the state of the lab Thing
  makeArtifact("lab", "org.hyperagents.jacamo.artifacts.wot.ThingArtifact", [Url], LabArtId);
  .print("Attempting to read from lab artifact with ID: ", LabArtId);

  // Task 2.2: Calculate Q matrices for the desired goal state
  .print("Agent: Starting Q-Learning for goal [", Z1Level, ",", Z2Level, "]");
  
  // Parameters for Q-Learning: episodes, alpha, gamma, epsilon, reward
  calculateQ([Z1Level, Z2Level], 100, 0.3, 0.9, 0.05, 100)[artifact_id(QLArtId)];
  .print("Q-Learning completed for goal [", Z1Level, ",", Z2Level, "]");

  .wait(2000);
  // Task 2.3: Try to Achieve the goal based on the learned Q values
  !achieve_goal([Z1Level, Z2Level]);
  .

// Task 2.3: Plan to achieve the goal state
+!achieve_goal(Goal) <-
  .print("Working to achieve goal state: ", Goal);
  !read_and_act(Goal, LabArtId); 
  .

// Task 2.3: TODO Read the current state and take appropriate action
+!read_and_act(Goal, LabId) : task_requirements([Z1Level, Z2Level]) <-
  
  .print("Reading current state...");
  readProperty("https://example.org/was#Status", CurrentTags, CurrentState);
  .print("Environment status - Tags: ", CurrentTags, ", State: ", CurrentState);

  getActionFromState([Z1Level,Z2Level], CurrentTags, CurrentState, NextTag, NextPayloadTags, NextPayload);
  .print("Selected action - Type: ", NextTag, ", Tags: ", NextPayloadTags, ", Value: ", NextPayload);

  .print("Goal light levels: Z1=", Z1Level, ", Z2=", Z2Level);
  invokeAction(NextTag, NextPayloadTags, NextPayload);

  .wait(2000);

  readProperty("https://example.org/was#Status", NextTags, NextState);

  targetReached([Z1Level,Z2Level], NextTags, NextState, Reached);
  
  if (Reached == true) {
    .print("Target  reached: [", Z1Level, ",", Z2Level, "]");
  } else {
    !read_and_act(Goal, LabId);
    .print("try again")
  }.
  
